#!/usr/bin/env python3
"""TIDE U32: dense local multimodal field mechanism pilot."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_melt import MLLM, sanitized_cohort
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times, timed_text, valid_chunks
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl

ARMS = ("u32_joint", "u32_visual", "u32_text", "u32_unmasked_joint")
POLICY = (
    "Hateful content is an asserted or endorsed attack, dehumanization, threat, "
    "exclusion, or hateful stereotype directed at a person or protected/social "
    "group. Quotation, neutral reporting, counterspeech, satire, and condemnation "
    "are not hateful unless the speaker endorses the hostility."
)


def carrier(images, focus, visible):
    """Identical 32-cell carrier; `visible` controls temporal evidence only."""
    canvas = Image.new("RGB", (896, 448), (127, 127, 127))
    draw = ImageDraw.Draw(canvas)
    for i in range(len(images)):
        x, y = (i % 8) * 112, (i // 8) * 112
        if i in visible:
            canvas.paste(images[i].convert("RGB").resize((112, 112)), (x, y))
        draw.text((x + 3, y + 3), str(i), fill="yellow", stroke_width=2, stroke_fill="black")
    x, y = (focus % 8) * 112, (focus // 8) * 112
    draw.rectangle((x + 2, y + 2, x + 109, y + 109), outline="red", width=5)
    return canvas


def prompt(index, records, arm):
    scope = (
        "The red cell is the only temporal cell to classify; only its immediate "
        "temporal neighbors are supplied as local context."
        if arm != "u32_unmasked_joint" else
        "The red numbered cell is the only temporal cell to classify; all other "
        "cells and speech are potentially contaminating global context."
    )
    speech = ("No transcript is supplied." if arm == "u32_visual" else
              f"Timestamp-aligned speech records: {json.dumps(records, ensure_ascii=False)}.")
    return (f"{POLICY} {scope} {speech} Does temporal cell {index} itself contain "
            "the complete asserted or endorsed hateful event evidence? Answer Yes or No only:")


@torch.inference_mode()
def score_batches(model, images, prompts, batch_size=8):
    output = []
    for start in range(0, len(images), batch_size):
        output.extend(model.binary_logits(images[start:start + batch_size],
                                          prompts[start:start + batch_size]).tolist())
    return np.asarray(output, float)


def bins_to_curve(values, duration):
    n = max(1, math.floor(duration * 4))
    times = (np.arange(n) + .5) / 4
    ids = np.minimum(len(values) - 1, (times / duration * len(values)).astype(int))
    return np.asarray(values, float)[ids]


def positive_intervals(z, duration):
    active = np.asarray(z) > 0
    bounds = np.flatnonzero(np.diff(np.r_[False, active, False])).reshape(-1, 2)
    return [Interval(a / len(active) * duration, b / len(active) * duration,
                     float(1 / (1 + np.exp(-np.mean(z[a:b]))))) for a, b in bounds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--bins", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated subset of the frozen U32 arms")
    args = ap.parse_args()
    if args.bins != 32:
        raise ValueError("TIDE U32 carrier requires --bins 32")
    selected_arms = tuple(x.strip() for x in args.arms.split(",") if x.strip())
    if not selected_arms or len(set(selected_arms)) != len(selected_arms) or not set(selected_arms) <= set(ARMS):
        raise ValueError(f"invalid --arms: {selected_arms}")
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    cohort = sanitized_cohort(args.cohort)
    if args.limit:
        cohort = cohort[:args.limit]
    asr = transcript_rows(); model = MLLM(args.model)
    for row in cohort:
        dataset, video_id = row["dataset"], row["video_id"]
        duration = float(row["duration"]); before = model.calls
        try:
            chunks, rejected = valid_chunks(asr.get((dataset, video_id), []), duration)
            speech = timed_text(chunks, duration, nbins=args.bins)
            # Freeze one common ASR budget before deriving any arm.
            remaining = 12000
            budgeted = []
            for text in speech:
                value = text[:remaining] if remaining > 0 else ""
                budgeted.append(value); remaining -= len(value)
            speech = budgeted
            records = [{"bin": i, "start": round(i * duration / args.bins, 3),
                        "end": round((i + 1) * duration / args.bins, 3), "text": text}
                       for i, text in enumerate(speech) if text]
            centers = (np.arange(args.bins) + .5) / args.bins * duration
            frames, times, fallback = frames_at_times(row["video_path"], centers)
            if len(frames) != args.bins:
                raise RuntimeError(f"decoded {len(frames)}/{args.bins} center frames")
            pending = []
            for arm in selected_arms:
                arm_before = model.calls
                actual_images, null_images, actual_prompts, null_prompts = [], [], [], []
                for i in range(args.bins):
                    local_ids = {max(0, i - 1), i, min(args.bins - 1, i + 1)}
                    visible = (set(range(args.bins)) if arm == "u32_unmasked_joint"
                               else set() if arm == "u32_text" else local_ids)
                    actual_images.append(carrier(frames, i, visible))
                    null_images.append(carrier(frames, i, set()))
                    if arm == "u32_visual":
                        local_records = []
                    elif arm == "u32_unmasked_joint":
                        local_records = records
                    else:
                        local_records = [r for r in records if r["bin"] in local_ids]
                    actual_prompts.append(prompt(i, local_records, arm))
                    null_prompts.append(prompt(i, [], arm))
                arm_batch = min(args.batch_size, 2) if arm == "u32_unmasked_joint" else args.batch_size
                raw = score_batches(model, actual_images, actual_prompts, arm_batch)
                null = score_batches(model, null_images, null_prompts, arm_batch)
                corrected = raw - null
                probability = 1 / (1 + np.exp(-np.clip(corrected, -30, 30)))
                curve = bins_to_curve(probability, duration)
                intervals = positive_intervals(corrected, duration)
                pending.append(Prediction(
                    arm, dataset, video_id, duration, score_curve=curve.tolist(),
                    intervals=intervals, calls=model.calls - arm_before,
                    modality_evidence={"raw_log_odds": raw.tolist(),
                                       "content_free_log_odds": null.tolist(),
                                       "corrected_log_odds": corrected.tolist(),
                                       "speech_bins": speech,
                                       "invalid_asr_spans_rejected": rejected,
                                       "ffmpeg_fallback_frames": fallback},
                    raw={"bins": args.bins, "batch_size": args.batch_size,
                         "arm": arm, "semantic_queries": 2 * args.bins,
                         "gt_access": False}))
            for prediction in pending:
                append_jsonl(args.out, prediction)
            print(json.dumps({"dataset": dataset, "video_id": video_id,
                              "calls": model.calls - before}), flush=True)
        except Exception as exc:
            print(json.dumps({"dataset": dataset, "video_id": video_id,
                              "failure": f"{type(exc).__name__}: {exc}"}), flush=True)
            raise


if __name__ == "__main__":
    main()
