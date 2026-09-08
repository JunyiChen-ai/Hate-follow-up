#!/usr/bin/env python3
"""Parameter-free multi-event readout from centered score excursions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    count = videos = 0
    fractions = []
    with args.out.open("w") as handle:
        for row in map(json.loads, args.prediction.open()):
            if row.get("method") != args.method:
                continue
            score = np.asarray(row["score_curve"], float)
            active = score > float(score.mean())
            changes = np.diff(np.r_[False, active, False].astype(np.int8))
            starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
            rate = float(row.get("native_rate", 4.0) or 4.0)
            intervals = [[int(left) / rate,
                          min(float(row["duration"]), int(right) / rate),
                          float(score[left:right].mean())]
                         for left, right in zip(starts, ends)]
            output = dict(row)
            output["method"] = "centered_excursion_components_v1"
            output["intervals"] = intervals
            output["raw"] = {**output.get("raw", {}), "gt_access": False,
                             "module": "centered_positive_excursion_components",
                             "centering": "current_video_arithmetic_mean",
                             "component_rule": "maximal_connected_strictly_positive_excursions",
                             "dataset_parameters": 0, "label_selected_parameters": 0,
                             "score_threshold": "sample_mean_not_numeric_hyperparameter",
                             "duration_prior": None, "code_sha256": code_hash}
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
            videos += 1; count += len(intervals)
            fractions.append(float(active.mean()))
    summary = {"n": videos, "intervals": count,
               "active_fraction_mean": float(np.mean(fractions)),
               "active_fraction_median": float(np.median(fractions))}
    args.out.with_suffix(".audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
