#!/usr/bin/env python3
"""Audit frame metrics under complete-case, common-cohort, and floor policies.

The primary evaluator deliberately reports frame metrics on available score
curves.  That is useful for debugging, but it is not by itself a fair basis
for comparing a partial-coverage method with a full-coverage method.  This
script makes the sensitivity explicit without changing the primary metric:

* complete_case: each method's available predictions;
* common_cohort: the intersection of GT and every compared method;
* missing_as_floor: every missing video receives a constant score strictly
  below that method's finite predictions on the same dataset.

No prediction is selected or modified using labels.  Ground truth is read only
after prediction rows and comparison cohorts have been fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.label_free_adapt.evaluate import pooled, within_video_macro


DATASETS = ("HateMM", "HateClipSeg", "MHC", "MHC_zh")


def parse_spec(value: str) -> tuple[str, Path, str]:
    """Parse NAME=PATH:METHOD while allowing colons inside neither field."""
    if "=" not in value or ":" not in value.rsplit("=", 1)[1]:
        raise argparse.ArgumentTypeError("expected NAME=PATH:METHOD")
    name, remainder = value.split("=", 1)
    path, method = remainder.rsplit(":", 1)
    if not name or not path or not method:
        raise argparse.ArgumentTypeError("expected nonempty NAME=PATH:METHOD")
    return name, Path(path), method


def load_predictions(path: Path, method: str) -> dict[tuple[str, str], np.ndarray]:
    latest: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("method") == method:
                latest[(str(row["dataset"]), str(row["video_id"]))] = row
    output = {}
    for key, row in latest.items():
        if row.get("error") or not row.get("score_curve"):
            continue
        score = np.asarray(row["score_curve"], dtype=np.float64)
        if score.size and np.isfinite(score).any():
            output[key] = score
    return output


def load_gt(path: Path, split: str) -> dict[str, np.ndarray]:
    gt = np.load(path, allow_pickle=True)
    return {
        str(video_id): np.asarray(gt["y4"][index], dtype=np.int8)
        for index, video_id in enumerate(gt["video_ids"])
        if str(gt["split"][index]) == split
    }


def metrics(y: dict[str, np.ndarray], scores: dict[str, np.ndarray]) -> dict:
    return {**pooled(y, scores), **within_video_macro(y, scores)}


def finite_floor(scores: dict[str, np.ndarray]) -> float:
    values = np.concatenate([x[np.isfinite(x)] for x in scores.values()])
    spread = float(np.std(values))
    return float(np.min(values) - max(1e-6, spread * 1e-6))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", type=parse_spec,
                        required=True, help="repeat NAME=PATH:METHOD")
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len({name for name, _, _ in args.prediction}) != len(args.prediction):
        raise RuntimeError("comparison names must be unique")

    predictions = {
        name: load_predictions(path, method)
        for name, path, method in args.prediction
    }
    per_dataset = []
    global_buffers = {
        policy: {name: ({}, {}) for name in predictions}
        for policy in ("complete_case", "common_cohort", "missing_as_floor")
    }
    for dataset in DATASETS:
        y = load_gt(args.gt_dir / f"{dataset}.npz", args.split)
        available = {
            name: {video_id: score for (source_dataset, video_id), score in rows.items()
                   if source_dataset == dataset and video_id in y}
            for name, rows in predictions.items()
        }
        common = set(y)
        for scores in available.values():
            common &= set(scores)
        for name, scores in available.items():
            floor = finite_floor(scores)
            policy_scores = {
                "complete_case": scores,
                "common_cohort": {video_id: scores[video_id] for video_id in common},
                "missing_as_floor": {
                    video_id: scores.get(video_id, np.full(len(target), floor))
                    for video_id, target in y.items()
                },
            }
            for policy, selected in policy_scores.items():
                selected_y = {video_id: y[video_id] for video_id in selected}
                result = {
                    "dataset": dataset,
                    "method": name,
                    "coverage_policy": policy,
                    "n_videos_gt": len(y),
                    "n_videos_scored": len(selected),
                    "n_videos_common": len(common),
                    **metrics(selected_y, selected),
                }
                if policy == "missing_as_floor":
                    result["imputation_score"] = floor
                per_dataset.append(result)
                global_y, global_s = global_buffers[policy][name]
                for video_id in selected:
                    key = f"{dataset}:{video_id}"
                    global_y[key] = y[video_id]
                    global_s[key] = selected[video_id]

    global_results = []
    for policy, by_method in global_buffers.items():
        for name, (y, scores) in by_method.items():
            global_results.append({
                "dataset": "GLOBAL_CONCATENATED",
                "method": name,
                "coverage_policy": policy,
                "n_videos_scored": len(scores),
                **metrics(y, scores),
            })
    output = {
        "schema_version": "1.0",
        "audit": "frame_metric_coverage_sensitivity",
        "ground_truth_access": "evaluation_only",
        "policies": {
            "complete_case": "available score curves per method",
            "common_cohort": "GT intersection shared by every compared method",
            "missing_as_floor": "missing video gets method/dataset finite minimum minus epsilon",
        },
        "inputs": [{"name": name, "path": str(path.resolve()), "method": method}
                   for name, path, method in args.prediction],
        "per_dataset": per_dataset,
        "global": global_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(global_results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
