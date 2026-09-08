#!/usr/bin/env python3
"""Decode intervals from cross-resolution posterior disagreement.

The decoder reads three frozen score fields derived from the same base trace.
For each video it centers their logits, regards their mean as the temporal
signal and their population standard deviation as epistemic disagreement, and
extracts the positive connected component containing the strongest certified
frame.  It has no dataset statistic, target label, score threshold, duration
prior, top-k, or inherited interval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid


def component_around_max(values: np.ndarray) -> tuple[int, int] | None:
    if not len(values):
        return None
    peak = int(np.argmax(values))
    if not np.isfinite(values[peak]) or values[peak] <= 0:
        return None
    positive = values > 0
    start = peak
    while start > 0 and positive[start - 1]:
        start -= 1
    end = peak + 1
    while end < len(values) and positive[end]:
        end += 1
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.source) != 3 or len(args.method) != 3:
        raise ValueError("exactly three --source and three --method arguments are required")
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    fields = [load(path, method) for path, method in zip(args.source, args.method)]
    keys = sorted(set.intersection(*(set(field) for field in fields)))
    with args.out.open("w") as handle:
        for key in keys:
            rows = [field[key] for field in fields]
            probabilities = [
                np.clip(np.asarray(row["score_curve"], dtype=float), 1e-6, 1 - 1e-6)
                for row in rows
            ]
            lengths = {len(values) for values in probabilities}
            if len(lengths) != 1:
                raise ValueError(f"unaligned score lengths for {key}: {sorted(lengths)}")
            logits = np.stack([np.log(p / (1 - p)) for p in probabilities])
            residuals = logits - logits.mean(axis=1, keepdims=True)
            nominal = residuals.mean(axis=0)
            disagreement = residuals.std(axis=0, ddof=0)
            certificates = {
                "uncertainty_lcb_excursion_v1": nominal - disagreement,
                "uncertainty_ucb_excursion_v1": nominal + disagreement,
                "unanimous_min_excursion_control_v1": residuals.min(axis=0),
                "nominal_mean_excursion_control_v1": nominal,
            }
            propensity = float(np.mean(logits))
            duration = float(rows[0]["duration"])
            rate = float(rows[0].get("native_rate", 4.0) or 4.0)
            for method, certificate in certificates.items():
                component = component_around_max(certificate)
                intervals = []
                if component is not None:
                    start, end = component
                    intervals = [[start / rate, min(duration, end / rate),
                                  float(sigmoid(np.max(certificate[start:end])))] ]
                out = dict(rows[0])
                out["method"] = method
                out["score_curve"] = sigmoid(propensity + certificate).tolist()
                out["intervals"] = intervals
                out["raw"] = {
                    **out.get("raw", {}),
                    "gt_access": False,
                    "module": "cross_resolution_uncertainty_excursion",
                    "sources": args.method,
                    "dataset_parameters": 0,
                    "label_selected_parameters": 0,
                    "inherited_intervals": False,
                    "rule": method,
                    "positive_reference": "per-video zero-mean temporal evidence",
                    "disagreement": "population standard deviation across three frozen resolutions",
                }
                handle.write(json.dumps(out, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
