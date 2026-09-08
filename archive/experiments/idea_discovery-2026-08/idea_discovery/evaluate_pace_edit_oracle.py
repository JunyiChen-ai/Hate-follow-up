#!/usr/bin/env python3
"""GT-isolated oracle for PACE's frozen single-boundary edit grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.label_free_adapt.evaluate import binary_intervals, interval_f1, temporal_iou

DATASETS = ("HateMM", "HateClipSeg", "MHC", "MHC_zh")
FRACTIONS = (-1 / 8, -1 / 16, 1 / 16, 1 / 8)


def candidates(interval, duration):
    start, end, score = map(float, interval[:3]); length = end - start
    output = [(start, end, score, "original")]
    for fraction in FRACTIONS:
        value = min(end - .25, max(0., start + fraction * length))
        if value < end:
            output.append((value, end, score, f"start:{fraction:+.4f}"))
        value = max(start + .25, min(duration, end + fraction * length))
        if value > start:
            output.append((start, value, score, f"end:{fraction:+.4f}"))
    unique = {}
    for row in output:
        unique[(round(row[0], 8), round(row[1], 8))] = row
    return list(unique.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a10", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cohort = {(r["dataset"], r["video_id"]): r
              for r in map(json.loads, args.manifest.open())}
    predictions = {}
    for row in map(json.loads, args.a10.open()):
        key = (row["dataset"], row["video_id"])
        if key in cohort:
            predictions[key] = row
    if set(predictions) != set(cohort):
        raise RuntimeError(f"A10/cohort mismatch missing={set(cohort)-set(predictions)}")
    detail = []; per_dataset = {}
    for dataset in DATASETS:
        gt = np.load(args.gt_dir / f"{dataset}.npz", allow_pickle=True)
        y = {str(video_id): np.asarray(gt["y4"][i], dtype=np.int8)
             for i, video_id in enumerate(gt["video_ids"])
             if str(gt["split"][i]) == "test"}
        baseline_rows, oracle_rows, y_subset = {}, {}, {}
        for (d, video_id), row in predictions.items():
            if d != dataset:
                continue
            gt_intervals = binary_intervals(y[video_id]); base = row.get("intervals", [])
            if not base:
                baseline_rows[video_id] = {"intervals": []}
                oracle_rows[video_id] = {"intervals": []}; y_subset[video_id] = y[video_id]
                detail.append({"dataset": dataset, "video_id": video_id,
                               "baseline_iou": 0., "oracle_iou": 0., "edit": "empty"})
                continue
            proposal = max(base, key=lambda x: float(x[2]) if len(x) > 2 else 1.)
            options = candidates(proposal, float(row["duration"]))
            def quality(option):
                return max((temporal_iou((option[0], option[1]), target)
                            for target in gt_intervals), default=0.)
            best = max(options, key=lambda x: (quality(x), x[3] == "original"))
            original = options[0]
            baseline_rows[video_id] = {"intervals": [list(original[:3])]}
            oracle_rows[video_id] = {"intervals": [list(best[:3])]}
            y_subset[video_id] = y[video_id]
            detail.append({"dataset": dataset, "video_id": video_id,
                           "baseline_iou": quality(original), "oracle_iou": quality(best),
                           "edit": best[3], "n_candidates": len(options)})
        baseline = interval_f1(y_subset, baseline_rows)
        oracle = interval_f1(y_subset, oracle_rows)
        per_dataset[dataset] = {"n": len(baseline_rows), "baseline": baseline,
                                "oracle": oracle,
                                "miss_to_hit@0.5": sum(x["baseline_iou"] < .5 <= x["oracle_iou"]
                                                       for x in detail if x["dataset"] == dataset)}
    macro = {}
    for metric in ("interval_F1@0.3", "interval_F1@0.5", "interval_F1@0.7"):
        baseline = float(np.mean([x["baseline"][metric] for x in per_dataset.values()]))
        oracle = float(np.mean([x["oracle"][metric] for x in per_dataset.values()]))
        macro[metric] = {"baseline": baseline, "oracle": oracle, "delta": oracle - baseline}
    output = {"per_dataset": per_dataset, "macro": macro, "detail": detail,
              "total_miss_to_hit@0.5": sum(x["baseline_iou"] < .5 <= x["oracle_iou"] for x in detail),
              "grid_fractions": FRACTIONS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
