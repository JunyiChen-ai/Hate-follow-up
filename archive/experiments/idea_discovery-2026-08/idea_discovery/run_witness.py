#!/usr/bin/env python3
"""WITNESS: fixed-budget minimal context-invariant witness search.

The inference path is label-free.  It never opens video labels or temporal GT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_melt import MLLM, sanitized_cohort
from scripts.idea_discovery.run_tide_u32 import POLICY, bins_to_curve, score_batches
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times, valid_chunks
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl

METHODS = (
    "witness",
    "witness_no_extraction",
    "witness_no_zoom",
    "hash_zoom_pair",
    "score_zoom_pair",
    "uniform24_pair",
)
VARIANTS = ("joint", "text_only", "visual_only", "frame_shuffle", "asr_shift")
GRID = 128
COARSE = 8
REFINE_PARENTS = 4
CHILDREN = 4


def interval_text(chunks, start, end, cap=1500):
    pieces = []
    for row in chunks:
        a, b = map(float, row["span"])
        if max(a, start) < min(b, end):
            pieces.append(str(row.get("text", "")))
    return " ".join(pieces)[:cap]


def nearest_frame(frames, normalized_time):
    index = min(len(frames) - 1, max(0, int(normalized_time * len(frames))))
    return frames[index]


def paired_carriers(frames, start, end):
    """Return matched narrow/exposed carriers for normalized [start,end)."""
    width = end - start
    focus_pos = [start + width * q for q in (.2, .4, .6, .8)]
    # Duplicate the exact focus anchors in the narrow arm.  The exposed arm
    # changes only these redundant slots to real outside-neighborhood frames;
    # the four target-evidence cells remain bit-identical between arms.
    narrow_context = list(focus_pos)
    exposed_context = [
        max(0.0, start - .75 * width),
        max(0.0, start - .25 * width),
        min(1.0 - 1e-9, end + .25 * width),
        min(1.0 - 1e-9, end + .75 * width),
    ]

    def render(context):
        positions = context[:2] + focus_pos + context[2:]
        canvas = Image.new("RGB", (896, 140), "white")
        draw = ImageDraw.Draw(canvas)
        for i, pos in enumerate(positions):
            x = i * 112
            image = nearest_frame(frames, pos).convert("RGB").resize((112, 112))
            canvas.paste(image, (x, 0))
            role = "F" if 2 <= i <= 5 else "C"
            draw.text((x + 3, 3), role, fill="yellow", stroke_width=2,
                      stroke_fill="black")
            if role == "F":
                draw.rectangle((x + 2, 2, x + 109, 109), outline="red", width=4)
        draw.text((4, 116), "F=target interval; C=context only", fill="black")
        return canvas

    return render(narrow_context), render(exposed_context)


def query_prompt(start, end, duration, speech):
    return (
        f"{POLICY} Red F cells are the only target interval; C cells are context "
        "and must not themselves be classified. The target spans "
        f"[{start * duration:.3f}, {end * duration:.3f}) seconds. "
        f"Timestamp-aligned target speech: {json.dumps(speech, ensure_ascii=False)}. "
        "Does the target interval contain the complete asserted or endorsed "
        "hateful event evidence? Answer Yes or No only:"
    )


def pair_score(model, frames, chunks, intervals, duration, batch_size,
               visual_enabled=True, speech_enabled=True):
    images, prompts = [], []
    for start, end in intervals:
        narrow, exposed = paired_carriers(frames, start, end)
        if not visual_enabled:
            blank = Image.new("RGB", narrow.size, (127, 127, 127))
            narrow = blank; exposed = blank.copy()
        speech = (interval_text(chunks, start * duration, end * duration)
                  if speech_enabled else "")
        prompt = query_prompt(start, end, duration, speech)
        images.extend((narrow, exposed)); prompts.extend((prompt, prompt))
    logits = score_batches(model, images, prompts, batch_size).reshape(-1, 2)
    return {interval: (float(z[0]), float(z[1]))
            for interval, z in zip(intervals, logits)}


def coarse_intervals():
    return [(i / COARSE, (i + 1) / COARSE) for i in range(COARSE)]


def split(interval):
    start, end = interval; width = (end - start) / CHILDREN
    return [(start + i * width, start + (i + 1) * width)
            for i in range(CHILDREN)]


def witness_order(scores):
    def key(interval):
        narrow, exposed = scores[interval]
        discordant = (narrow > 0) != (exposed > 0)
        paired_positive = narrow > 0 and exposed > 0
        return (int(discordant), int(paired_positive), abs(narrow - exposed),
                max(narrow, exposed), -interval[0])
    return sorted(scores, key=key, reverse=True)


def score_order(scores):
    return sorted(scores, key=lambda x: (scores[x][0], -x[0]), reverse=True)


def hash_order(scores, dataset, video_id):
    def key(interval):
        token = f"{dataset}\0{video_id}\0{interval[0]:.8f}\0{interval[1]:.8f}"
        return hashlib.sha256(token.encode()).hexdigest()
    return sorted(scores, key=key)


def leaves(parents, selected):
    chosen = set(selected)
    output = []
    for interval in parents:
        output.extend(split(interval) if interval in chosen else [interval])
    return output


def minimal_witness_partition(parents, selected, scores):
    """Use children when any is a witness, otherwise fall back to the parent."""
    chosen = set(selected); output = []
    for parent in parents:
        if parent not in chosen:
            output.append(parent); continue
        children = split(parent)
        child_positive = any(scores[x][0] > 0 and scores[x][1] > 0
                             for x in children)
        parent_positive = scores[parent][0] > 0 and scores[parent][1] > 0
        output.extend(children if child_positive or not parent_positive else [parent])
    return output


def active_intervals(leaves_, scores, paired=True):
    active = []
    for interval in leaves_:
        narrow, exposed = scores[interval]
        if narrow > 0 and (exposed > 0 if paired else True):
            active.append(interval)
    active.sort()
    merged = []
    for start, end in active:
        if merged and abs(merged[-1][1] - start) < 1e-10:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged


def dense_leaf_scores(leaves_, scores, duration, paired=True):
    n = max(1, math.floor(duration * 4)); times = (np.arange(n) + .5) / (4 * duration)
    values = np.zeros(n, float)
    for interval in leaves_:
        start, end = interval; narrow, exposed = scores[interval]
        value = min(narrow, exposed) if paired else narrow
        mask = (times >= start) & (times < end)
        values[mask] = 1 / (1 + np.exp(-np.clip(value, -30, 30)))
    return values.tolist()


def prediction(method, row, leaves_, scores, paired, rejected, fallback,
               shared_video_calls, semantic_queries, logical_batch_forwards):
    duration = float(row["duration"])
    intervals = active_intervals(leaves_, scores, paired=paired)
    objects = []
    for start, end in intervals:
        covered = [scores[x] for x in leaves_ if x[0] >= start and x[1] <= end]
        margin = min(min(pair) if paired else pair[0] for pair in covered)
        objects.append(Interval(start * duration, end * duration,
                                float(1 / (1 + np.exp(-margin)))))
    return Prediction(
        method, row["dataset"], row["video_id"], duration,
        score_curve=dense_leaf_scores(leaves_, scores, duration, paired),
        intervals=objects, calls=logical_batch_forwards,
        modality_evidence={
            "leaf_windows": [[a, b] for a, b in leaves_],
            "narrow_log_odds": [scores[x][0] for x in leaves_],
            "exposed_log_odds": [scores[x][1] for x in leaves_],
            "paired_required": paired,
            "invalid_asr_spans_rejected": rejected,
            "ffmpeg_fallback_frames": fallback,
        },
        raw={
            "gt_access": False,
            "semantic_queries": semantic_queries,
            "shared_runner_batch_forwards": shared_video_calls,
            "grid_frames": GRID,
            "decision_threshold": 0.0,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--variant", choices=VARIANTS, default="joint")
    args = parser.parse_args()
    selected_methods = tuple(x.strip() for x in args.methods.split(",") if x.strip())
    if not selected_methods or not set(selected_methods) <= set(METHODS):
        raise ValueError(f"invalid methods: {selected_methods}")
    if len(set(selected_methods)) != len(selected_methods):
        raise ValueError("duplicate methods")
    partial = args.out.with_name(args.out.name + ".partial")
    if args.out.exists() or partial.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    cohort = sanitized_cohort(args.cohort)
    if args.limit:
        cohort = cohort[:args.limit]
    asr = transcript_rows(); model = MLLM(args.model)
    expected = set()
    try:
        for row in cohort:
            before = model.calls; duration = float(row["duration"])
            chunks, rejected = valid_chunks(asr.get((row["dataset"], row["video_id"]), []), duration)
            if args.variant == "asr_shift":
                shifted = []
                for chunk in chunks:
                    item = dict(chunk); a, b = map(float, item["span"])
                    width = b - a; start = (a + duration / 2) % duration
                    if start + width <= duration:
                        item["span"] = [start, start + width]; shifted.append(item)
                    else:
                        first = dict(item); first["span"] = [start, duration]
                        second = dict(item); second["span"] = [0.0, start + width - duration]
                        shifted.extend((first, second))
                chunks = shifted
            centers = (np.arange(GRID) + .5) / GRID * duration
            frames, _, fallback = frames_at_times(row["video_path"], centers)
            if len(frames) != GRID:
                raise RuntimeError(f"decoded {len(frames)}/{GRID} grid frames")
            if args.variant == "frame_shuffle":
                token = f"{row['dataset']}\0{row['video_id']}\0frame_shuffle"
                seed = int(hashlib.sha256(token.encode()).hexdigest()[:16], 16)
                order = np.random.default_rng(seed).permutation(GRID)
                frames = [frames[int(i)] for i in order]
            visual_enabled = args.variant != "text_only"
            speech_enabled = args.variant != "visual_only"
            coarse = coarse_intervals()
            needs_coarse = any(name != "uniform24_pair" for name in selected_methods)
            all_scores = (pair_score(model, frames, chunks, coarse, duration,
                                     args.batch_size, visual_enabled, speech_enabled)
                          if needs_coarse else {})

            witness_selected = (witness_order(all_scores)[:REFINE_PARENTS]
                                if {"witness", "witness_no_extraction"} & set(selected_methods) else [])
            score_selected = (score_order(all_scores)[:REFINE_PARENTS]
                              if "score_zoom_pair" in selected_methods else [])
            hash_selected = (hash_order(all_scores, row["dataset"], row["video_id"])[:REFINE_PARENTS]
                             if "hash_zoom_pair" in selected_methods else [])
            requested_parents = []
            if {"witness", "witness_no_extraction"} & set(selected_methods):
                requested_parents += witness_selected
            if "hash_zoom_pair" in selected_methods:
                requested_parents += hash_selected
            if "score_zoom_pair" in selected_methods:
                requested_parents += score_selected
            needed_children = sorted(set(sum((split(x) for x in requested_parents), [])))
            if needed_children:
                all_scores.update(pair_score(model, frames, chunks, needed_children,
                                             duration, args.batch_size,
                                             visual_enabled, speech_enabled))

            uniform = [(i / 24, (i + 1) / 24) for i in range(24)]
            uniform_scores = {}
            if "uniform24_pair" in selected_methods:
                uniform_scores = pair_score(model, frames, chunks, uniform, duration,
                                            args.batch_size, visual_enabled, speech_enabled)
            actual_calls = model.calls - before
            adaptive_calls = math.ceil(16 / args.batch_size) + math.ceil(32 / args.batch_size)
            configs = {}
            for name in selected_methods:
                if name == "witness":
                    configs[name] = (minimal_witness_partition(coarse, witness_selected, all_scores),
                                     all_scores, True, 48, adaptive_calls)
                elif name == "witness_no_extraction":
                    configs[name] = (leaves(coarse, witness_selected), all_scores,
                                     False, 48, adaptive_calls)
                elif name == "witness_no_zoom":
                    configs[name] = (coarse, all_scores, True, 16,
                                     math.ceil(16 / args.batch_size))
                elif name == "hash_zoom_pair":
                    configs[name] = (leaves(coarse, hash_selected), all_scores,
                                     True, 48, adaptive_calls)
                elif name == "score_zoom_pair":
                    configs[name] = (leaves(coarse, score_selected), all_scores,
                                     True, 48, adaptive_calls)
                elif name == "uniform24_pair":
                    configs[name] = (uniform, uniform_scores, True, 48,
                                     math.ceil(48 / args.batch_size))
            suffix = "" if args.variant == "joint" else f"_{args.variant}"
            pending = [prediction(name + suffix, row, *configs[name][:3], rejected,
                                  fallback, actual_calls, configs[name][3], configs[name][4])
                       for name in selected_methods]
            for item in pending:
                append_jsonl(partial, item)
                expected.add((row["dataset"], row["video_id"], item.method))
            print(json.dumps({"dataset": row["dataset"], "video_id": row["video_id"],
                              "variant": args.variant, "batched_calls": actual_calls,
                              "semantic_queries_unique": 2 * ((len(coarse) if needs_coarse else 0) +
                                                               len(needed_children) +
                                                               len(uniform_scores))}), flush=True)
        observed = []
        with partial.open() as handle:
            for line in handle:
                row = json.loads(line); observed.append((row["dataset"], row["video_id"], row["method"]))
        if len(observed) != len(expected) or len(set(observed)) != len(observed):
            raise RuntimeError(f"output cardinality/uniqueness failure: {len(observed)} rows, {len(expected)} expected")
        os.replace(partial, args.out)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


if __name__ == "__main__":
    main()
