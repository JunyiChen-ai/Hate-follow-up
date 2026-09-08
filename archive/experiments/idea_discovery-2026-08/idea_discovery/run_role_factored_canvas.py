#!/usr/bin/env python3
"""Single-MLLM role-factored temporal canvas localization.

The MLLM must separately judge whether a protected-target hostile proposition
is present and whether the video's current author/speaker endorses it.  Their
noncompensatory minimum is the hate-localization score.  This prevents vivid
quoted hate from compensating for absent endorsement and prevents generic
hostility from compensating for an absent protected target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import rows, transcript_rows
from scripts.idea_discovery.run_visual_temporal_canvas import (
    dense_bins, infer, make_canvas, timed_text, valid_chunks, video_frames,
)
from scripts.label_free_adapt.mechanisms import parse_json_object
from scripts.label_free_adapt.schema import Prediction, append_jsonl


def valid_axis(value: object) -> bool:
    return (isinstance(value, list) and len(value) == 16 and
            all(isinstance(x, (int, float)) and math.isfinite(float(x))
                and 0 <= float(x) <= 100 for x in value))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, max_pixels=896 * 800)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="cuda:0", local_files_only=True,
        attn_implementation="sdpa").eval()
    transcript = transcript_rows()
    cohort = rows(args.cohort)
    if args.limit:
        cohort = cohort[:args.limit]
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    with args.out.open("w") as handle:
        for row in cohort:
            dataset, video_id = row["dataset"], row["video_id"]
            duration = float(row["duration"])
            chunks, rejected = valid_chunks(
                transcript.get((dataset, video_id), []), duration)
            speech = timed_text(chunks, duration)
            speech_records = [
                {"bin": i, "start": round(i * duration / 16, 2),
                 "end": round((i + 1) * duration / 16, 2), "text": value}
                for i, value in enumerate(speech) if value
            ]
            images, times, fallback = video_frames(row["video_path"], duration)
            canvas = make_canvas(images, times, None, "", 16)
            prompt = (
                "The image is a chronological 16-bin video timeline. Timestamp-aligned "
                "speech is " + json.dumps(speech_records, ensure_ascii=False) + ". "
                "Judge every bin locally using its frame, its timestamped speech, and temporal "
                "context. Return two independent 0..100 axes. target_attack means evidence that "
                "the bin communicates an attack, dehumanization, threat, exclusion, slur, or "
                "hateful stereotype against a protected person or group. Generic profanity, "
                "ordinary violence, personal disputes, and non-protected hostility must score "
                "low. endorsement means evidence that the current author or speaker asserts, "
                "adopts, celebrates, or supports that protected-target hostility. Quotation, "
                "news reporting, documentary display, rebuttal, condemnation, and counterspeech "
                "must have low endorsement even when target_attack is visually or lexically vivid. "
                "Do not copy one video-wide verdict to all bins; use graded local evidence. "
                "Return JSON only, with exactly 16 numeric entries per axis: "
                '{"target_attack":[...],"endorsement":[...]}.')
            attempts = []
            parsed = None
            for attempt in range(2):
                retry = "" if attempt == 0 else (
                    " Previous output was invalid. Return exactly the two requested arrays, "
                    "each with 16 finite numbers in [0,100].")
                raw = infer(canvas, prompt + retry, processor, model)
                attempts.append(raw)
                try:
                    obj = parse_json_object(raw)
                    attack, endorsement = obj.get("target_attack"), obj.get("endorsement")
                    if valid_axis(attack) and valid_axis(endorsement):
                        parsed = (np.asarray(attack, dtype=float) / 100.0,
                                  np.asarray(endorsement, dtype=float) / 100.0)
                        break
                except (TypeError, ValueError):
                    pass
            if parsed is None:
                raise RuntimeError(
                    f"invalid role axes for {dataset}/{video_id}: {attempts!r}")
            attack, endorsement = parsed
            bottleneck = np.minimum(attack, endorsement)
            n = max(1, int(np.floor(duration * 4.0)))
            dense_attack = dense_bins(attack, n)
            dense_endorsement = dense_bins(endorsement, n)
            dense_score = dense_bins(bottleneck, n)
            prediction = Prediction(
                "role_factored_canvas_bottleneck_v1", dataset, video_id, duration,
                score_curve=dense_score.tolist(), intervals=[], calls=len(attempts),
                modality_evidence={
                    "target_attack_bins": attack.tolist(),
                    "endorsement_bins": endorsement.tolist(),
                    "dense_target_attack": dense_attack.tolist(),
                    "dense_endorsement": dense_endorsement.tolist(),
                    "invalid_asr_spans_rejected": rejected,
                    "ffmpeg_fallback_frames": fallback,
                },
                raw={
                    "gt_access": False,
                    "module": "single_mllm_role_factored_canvas",
                    "fusion": "noncompensatory_minimum_of_target_attack_and_endorsement",
                    "numeric_weights": 0,
                    "dataset_parameters": 0,
                    "label_selected_parameters": 0,
                    "responses": attempts,
                    "code_sha256": code_hash,
                },
            )
            handle.write(json.dumps(prediction.to_dict(), separators=(",", ":")) + "\n")
            print(json.dumps({"dataset": dataset, "video_id": video_id,
                              "attack": attack.tolist(),
                              "endorsement": endorsement.tolist()}), flush=True)


if __name__ == "__main__":
    main()
