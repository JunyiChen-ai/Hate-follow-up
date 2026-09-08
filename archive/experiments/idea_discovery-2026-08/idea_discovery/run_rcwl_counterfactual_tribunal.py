#!/usr/bin/env python3
"""RCWL pilot: typed, paired counterfactual arbitration of LESS intervals.

Inference is label-free.  LESS supplies at most two nested hypotheses.  A
frozen MLLM evaluates real keep/remove/modality-drop interventions on identical
16-cell carriers.  It never generates a boundary or replaces the dense curve.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_melt import MLLM, sanitized_cohort
from scripts.idea_discovery.run_tide_u32 import POLICY, score_batches
from scripts.idea_discovery.run_visual_temporal_canvas import (
    frames_at_times, make_canvas, timed_text, valid_chunks,
)


BASE_METHODS = {
    "tight": "fact_less_t3al_dualgeo_shorter_v5",
    "broad": "fact_less_t3al_dualgeo_union_v5",
    "midpoint": "fact_less_t3al_dualgeo_midpoint_v5",
}


def load_predictions(path: Path):
    output = {}
    for row in map(json.loads, path.open()):
        output[(row["dataset"], row["video_id"], row["method"])] = row
    return output


def load_relations(path: Path):
    output = {}
    for row in map(json.loads, path.open()):
        relation = row.get("modality_evidence", {}).get("event_relation")
        if relation:
            output[(row["dataset"], row["video_id"])] = relation
    return output


def interval(row):
    values = row.get("intervals", [])
    if not values:
        return None
    return float(values[0][0]), float(values[0][1])


def candidate_rows(base, key):
    found = []
    for name in ("tight", "midpoint", "broad"):
        row = base.get((*key, BASE_METHODS[name]))
        value = interval(row) if row else None
        if value and value not in [x[1] for x in found]:
            found.append((name, value))
    return found[:3]


def records_from_bins(speech, duration):
    return [{"bin": i, "start": round(i * duration / len(speech), 3),
             "end": round((i + 1) * duration / len(speech), 3), "text": text}
            for i, text in enumerate(speech) if text]


def select_records(records, visible):
    return [row for row in records if int(row["bin"]) in visible]


def shift_records(records, amount, nbins):
    output = []
    duration = max((float(row["end"]) for row in records), default=float(nbins))
    for row in records:
        item = dict(row); item["bin"] = (int(item["bin"]) + amount) % nbins
        item["start"] = round(item["bin"] * duration / nbins, 3)
        item["end"] = round((item["bin"] + 1) * duration / nbins, 3)
        output.append(item)
    return output


def mask_canvas(images, times, visible):
    blank = Image.new("RGB", images[0].size, (127, 127, 127))
    rendered = [image if i in visible else blank for i, image in enumerate(images)]
    return make_canvas(rendered, times, None, "", nbins=len(rendered))


def prompt(relation, records):
    return (
        f"{POLICY} Fixed candidate relation={json.dumps(relation, ensure_ascii=False)}. "
        "The image and timestamped speech contain exactly the evidence available "
        "under one intervention; gray cells mean evidence was withheld. Do not infer "
        "withheld content and do not use stereotypes about the uploader. Supplied "
        f"timestamped speech={json.dumps(records, ensure_ascii=False)}. Does the supplied "
        "evidence itself establish this complete source-to-hostile-act-to-target relation "
        "with asserted or endorsed stance? Answer Yes or No only:"
    )


def arm_set(images, times, records, relation, inside, shifts):
    all_ids = set(range(len(images))); outside = all_ids - inside; none = set()
    arms = []
    def add(name, visible, recs):
        arms.append((name, mask_canvas(images, times, visible), prompt(relation, recs)))
    add("full", all_ids, records)
    add("null", none, [])
    add("keep", inside, select_records(records, inside))
    add("remove", outside, select_records(records, outside))
    add("drop_visual", none, select_records(records, inside))
    add("drop_text", inside, [])
    for shift in shifts:
        shifted = shift_records(records, shift, len(images))
        add(f"shift{shift}_keep", inside, select_records(shifted, inside))
        add(f"shift{shift}_remove", outside, select_records(shifted, outside))
        add(f"shift{shift}_drop_visual", none, select_records(shifted, inside))
    return arms


def certificate(values, prefix=""):
    keep = values[prefix + "keep"]
    remove = values[prefix + "remove"]
    drop_v = values[prefix + "drop_visual"]
    return {
        "sufficiency": keep - values["null"],
        "necessity": values["full"] - remove,
        "visual_contribution": keep - drop_v,
        "text_contribution": keep - values["drop_text"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--relations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bins", type=int, default=16)
    ap.add_argument("--shifts", default="4,8,12")
    args = ap.parse_args()
    partial = args.out.with_name(args.out.name + ".partial")
    if args.out.exists() or partial.exists():
        raise RuntimeError(f"refusing existing output/partial: {args.out}")
    cohort = sanitized_cohort(args.cohort)
    if args.limit:
        cohort = cohort[:args.limit]
    base, relations = load_predictions(args.base), load_relations(args.relations)
    asr = transcript_rows(); model = MLLM(args.model)
    shifts = tuple(int(x) for x in args.shifts.split(",") if x)
    written = 0
    try:
        with partial.open("w") as handle:
            for row in cohort:
                key = row["dataset"], row["video_id"]
                candidates = candidate_rows(base, key); relation = relations.get(key)
                if not candidates or relation is None:
                    continue
                duration = float(row["duration"]); before = model.calls
                chunks, rejected = valid_chunks(asr.get(key, []), duration)
                speech = timed_text(chunks, duration, nbins=args.bins)
                records = records_from_bins(speech, duration)
                centers = (np.arange(args.bins) + .5) / args.bins * duration
                images, times, fallback = frames_at_times(row["video_path"], centers)
                if len(images) != args.bins:
                    raise RuntimeError(f"decoded {len(images)}/{args.bins}: {key}")
                all_images, all_prompts, specs = [], [], []
                for name, (start, end) in candidates:
                    inside = {i for i, time in enumerate(times) if start <= time < end}
                    if not inside:
                        continue
                    for arm, image, text in arm_set(images, times, records, relation, inside, shifts):
                        specs.append((name, start, end, arm)); all_images.append(image); all_prompts.append(text)
                logits = score_batches(model, all_images, all_prompts, args.batch_size)
                grouped = {}
                for spec, value in zip(specs, logits):
                    name, start, end, arm = spec
                    grouped.setdefault((name, start, end), {})[arm] = float(value)
                output_candidates = []
                for (name, start, end), values in grouped.items():
                    aligned = certificate(values)
                    shift_certificates = [certificate(values, f"shift{shift}_") for shift in shifts]
                    aligned_min = min(aligned.values())
                    null_mins = [min(item.values()) for item in shift_certificates]
                    output_candidates.append({
                        "name": name, "interval": [start, end], "arm_log_odds": values,
                        "aligned_effects": aligned, "aligned_min_effect": aligned_min,
                        "shift_effects": shift_certificates,
                        "shift_min_effects": null_mins,
                        "null_corrected_certificate": aligned_min - float(np.median(null_mins)),
                    })
                result = {
                    "method": "rcwl_counterfactual_tribunal_v1",
                    "dataset": key[0], "video_id": key[1], "duration": duration,
                    "relation": relation, "candidates": output_candidates,
                    "calls": model.calls - before,
                    "modality_evidence": {"invalid_asr_spans_rejected": rejected,
                                          "ffmpeg_fallback_frames": fallback},
                    "raw": {"gt_access": False, "bins": args.bins,
                            "shifts": list(shifts), "dense_authority": "LESS"},
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n"); handle.flush()
                written += 1
                print(json.dumps({"dataset": key[0], "video_id": key[1],
                                  "candidates": len(output_candidates),
                                  "calls": model.calls - before}), flush=True)
        if not written:
            raise RuntimeError("no overlapping eligible videos")
        os.replace(partial, args.out)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
