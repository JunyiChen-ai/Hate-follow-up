#!/usr/bin/env python3
"""Decode one interval from a single score field by conserved excursions.

The decoder has no second proposal model and no threshold hyperparameter.  For
each video it centers the final frame field by that video's own arithmetic
mean, then returns the smallest half-open temporal interval containing every
strictly positive excursion.  The mean is the unique zero-total-mass baseline
of the observed field, so the rule is equivariant to positive affine score
changes and adapts to each sample without labels or corpus statistics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_role_orthogonal_transport import load


METHOD = "conserved_positive_excursion_hull_v1"


def excursion_hull(score: np.ndarray, rate: float, duration: float) -> list[list[float]]:
    finite = np.isfinite(score)
    if not finite.any():
        return []
    # Invalid positions cannot create evidence.  The baseline is computed only
    # from the observed finite field and is therefore independent of GT length.
    baseline = float(np.mean(score[finite]))
    positive = finite & (score > baseline)
    indices = np.flatnonzero(positive)
    if indices.size == 0:
        return []
    lo, hi = int(indices[0]), int(indices[-1]) + 1
    confidence = float(np.mean(score[lo:hi]) - baseline)
    return [[float(lo / rate), min(float(duration), float(hi / rate)), confidence]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--prediction-method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    rows = load(args.prediction, args.prediction_method)
    fractions = []
    with args.out.open("w", encoding="utf-8") as handle:
        for _, source in sorted(rows.items()):
            duration = float(source["duration"])
            rate = float(source.get("native_rate", 4.0) or 4.0)
            expected = max(1, int(np.floor(duration * rate)))
            score = np.asarray(source["score_curve"], dtype=np.float64)[:expected]
            intervals = excursion_hull(score, rate, duration)
            if intervals and duration > 0:
                fractions.append((intervals[0][1] - intervals[0][0]) / duration)
            output = dict(source)
            output["method"] = METHOD
            output["score_curve"] = score.tolist()
            output["intervals"] = intervals
            output["raw"] = {
                **output.get("raw", {}),
                "boundary_module": "conserved_positive_excursion_hull",
                "boundary_input": "same_final_frame_score_field",
                "second_detector": False,
                "sample_baseline": "arithmetic_mean_of_finite_frame_scores",
                "interval_rule": "minimal_half_open_hull_of_strict_positive_deviations",
                "numeric_parameters": 0,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
                "score_threshold": None,
                "gt_access": False,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")

    values = np.asarray(fractions, dtype=np.float64)
    audit = {
        "method": METHOD,
        "n_rows": len(rows),
        "n_nonempty": len(fractions),
        "duration_fraction_mean": float(np.mean(values)) if len(values) else None,
        "duration_fraction_median": float(np.median(values)) if len(values) else None,
        "duration_fraction_q10_q90": (np.quantile(values, [0.1, 0.9]).tolist()
                                      if len(values) else None),
        "second_detector": False,
        "numeric_parameters": 0,
        "gt_access": False,
    }
    args.out.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
