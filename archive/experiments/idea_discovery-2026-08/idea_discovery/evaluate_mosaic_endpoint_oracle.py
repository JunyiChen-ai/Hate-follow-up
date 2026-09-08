#!/usr/bin/env python3
"""GT-isolated ceiling for MOSAIC's frozen bilateral endpoint lattice.

This evaluator never produces label-free predictions.  It asks whether the
existing tight/midpoint/broad LESS bank contains enough independent start/end
headroom to justify building an endpoint-membership selector.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.label_free_adapt.evaluate import binary_intervals, interval_f1, temporal_iou


METHODS = {
    "tight": "fact_less_t3al_dualgeo_shorter_v5",
    "midpoint": "fact_less_t3al_dualgeo_midpoint_v5",
    "broad": "fact_less_t3al_dualgeo_union_v5",
}
METRICS = ("interval_F1@0.3", "interval_F1@0.5", "interval_F1@0.7")


def load_bank(path: Path) -> dict[str, dict[tuple[str, str], dict]]:
    bank = {name: {} for name in METHODS}
    reverse = {method: name for name, method in METHODS.items()}
    for row in map(json.loads, path.open()):
        name = reverse.get(row.get("method"))
        if name is not None:
            bank[name][(row["dataset"], row["video_id"])] = row
    return bank


def first_interval(row: dict) -> tuple[float, float] | None:
    intervals = row.get("intervals", [])
    if not intervals:
        return None
    return float(intervals[0][0]), float(intervals[0][1])


def endpoint_levels(intervals: dict[str, tuple[float, float] | None], side: int,
                    levels: int) -> list[tuple[str, float]]:
    """Return a fixed 3- or 5-level endpoint lattice without consulting GT."""
    base = [(name, value[side]) for name, value in intervals.items() if value is not None]
    if levels == 3:
        return base
    if levels != 5:
        raise ValueError(f"unsupported level count: {levels}")
    values = dict(base)
    return [
        ("tight", values["tight"]),
        ("tight_mid_half", 0.5 * (values["tight"] + values["midpoint"])),
        ("midpoint", values["midpoint"]),
        ("mid_broad_half", 0.5 * (values["midpoint"] + values["broad"])),
        ("broad", values["broad"]),
    ]


def load_gt(gt_dir: Path, keys: set[tuple[str, str]]) -> dict[tuple[str, str], np.ndarray]:
    output = {}
    for dataset in sorted({key[0] for key in keys}):
        archive = np.load(gt_dir / f"{dataset}.npz", allow_pickle=True)
        for index, video_id in enumerate(archive["video_ids"]):
            key = dataset, str(video_id)
            if key in keys:
                output[key] = np.asarray(archive["y4"][index], dtype=np.int8)
    return output


def quality(interval: tuple[float, float], targets: list[tuple[float, float]]) -> float:
    return max((temporal_iou(interval, target) for target in targets), default=0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--levels", type=int, choices=(3, 5), default=3)
    args = parser.parse_args()

    bank = load_bank(args.bank)
    keys = set.intersection(*(set(rows) for rows in bank.values()))
    gt = load_gt(args.gt_dir, keys)
    keys &= set(gt)
    baseline, oracle, detail = {}, {}, []

    for key in sorted(keys):
        intervals = {name: first_interval(rows[key]) for name, rows in bank.items()}
        midpoint = intervals["midpoint"]
        if midpoint is None:
            baseline[key] = {"intervals": []}
            oracle[key] = {"intervals": []}
            detail.append({"dataset": key[0], "video_id": key[1], "empty": True,
                           "strict_gain": False, "start_source": "midpoint",
                           "end_source": "midpoint", "base_iou": 0.0, "oracle_iou": 0.0})
            continue
        targets = binary_intervals(gt[key])
        candidates = []
        for start_name, start_value in endpoint_levels(intervals, 0, args.levels):
            for end_name, end_value in endpoint_levels(intervals, 1, args.levels):
                candidate = start_value, end_value
                if candidate[0] < candidate[1]:
                    candidates.append((start_name, end_name, candidate))
        base_iou = quality(midpoint, targets)
        scored = [(quality(candidate, targets), start_name == "midpoint" and end_name == "midpoint",
                   start_name, end_name, candidate)
                  for start_name, end_name, candidate in candidates]
        best_iou, _, start_name, end_name, best = max(scored)
        baseline[key] = {"intervals": [[midpoint[0], midpoint[1], 1.0]]}
        oracle[key] = {"intervals": [[best[0], best[1], 1.0]]}
        detail.append({"dataset": key[0], "video_id": key[1], "empty": False,
                       "strict_gain": best_iou > base_iou + 1e-12,
                       "start_source": start_name, "end_source": end_name,
                       "base_iou": base_iou, "oracle_iou": best_iou})

    per_dataset = {}
    for dataset in sorted({key[0] for key in keys}):
        subset = [key for key in keys if key[0] == dataset]
        rows = [item for item in detail if item["dataset"] == dataset]
        per_dataset[dataset] = {
            "n": len(subset),
            "base": interval_f1({key: gt[key] for key in subset},
                                {key: baseline[key] for key in subset}),
            "oracle": interval_f1({key: gt[key] for key in subset},
                                  {key: oracle[key] for key in subset}),
            "strict_gain_videos": sum(item["strict_gain"] for item in rows),
            "start_only": sum(item["strict_gain"] and item["start_source"] != "midpoint"
                              and item["end_source"] == "midpoint" for item in rows),
            "end_only": sum(item["strict_gain"] and item["start_source"] == "midpoint"
                            and item["end_source"] != "midpoint" for item in rows),
            "bilateral": sum(item["strict_gain"] and item["start_source"] != "midpoint"
                             and item["end_source"] != "midpoint" for item in rows),
            "oracle_actions": dict(Counter(
                f"{item['start_source']}->{item['end_source']}" for item in rows
                if item["strict_gain"])),
        }

    macro = {}
    for metric in METRICS:
        base_value = float(np.mean([row["base"][metric] for row in per_dataset.values()]))
        oracle_value = float(np.mean([row["oracle"][metric] for row in per_dataset.values()]))
        macro[metric] = {"base": base_value, "oracle": oracle_value,
                         "delta": oracle_value - base_value}
    result = {
        "schema_version": 1,
        "purpose": "GT-isolated MOSAIC bilateral endpoint-lattice ceiling",
        "label_free": False,
        "bank_methods": METHODS,
        "endpoint_levels": args.levels,
        "n": len(keys),
        "per_dataset": per_dataset,
        "macro": macro,
        "detail": detail,
    }
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n": result["n"], "macro": macro,
                      "datasets": {name: {metric: row["oracle"][metric] - row["base"][metric]
                                           for metric in METRICS}
                                   | {"strict_gain_videos": row["strict_gain_videos"]}
                                   for name, row in per_dataset.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
