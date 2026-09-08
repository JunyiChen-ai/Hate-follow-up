#!/usr/bin/env python3
"""Decode final role-transport scores with a cross-resolution uncertainty envelope."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid
from scripts.idea_discovery.project_uncertainty_excursion_decoder import component_around_max


def logit(row: dict) -> np.ndarray:
    p = np.clip(np.asarray(row["score_curve"], dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--final-method", required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.source) != 3 or len(args.method) != 3:
        raise ValueError("exactly three uncertainty sources are required")
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    final = load(args.final, args.final_method)
    sources = [load(path, method) for path, method in zip(args.source, args.method)]
    keys = sorted(set(final).intersection(*(set(rows) for rows in sources)))
    with args.out.open("w") as handle:
        for key in keys:
            row = final[key]
            z = logit(row)
            residual = z - z.mean()
            source_logits = np.stack([logit(rows[key]) for rows in sources])
            if source_logits.shape[1] != len(z):
                raise ValueError(f"unaligned uncertainty sources at {key}")
            source_residuals = source_logits - source_logits.mean(axis=1, keepdims=True)
            disagreement = source_residuals.std(axis=0, ddof=0)
            upper_envelope = residual + disagreement
            selected = component_around_max(upper_envelope)
            duration = float(row["duration"])
            rate = float(row.get("native_rate", 4.0) or 4.0)
            intervals = []
            if selected is not None:
                start, end = selected
                intervals = [[start / rate, min(duration, end / rate),
                              float(sigmoid(np.max(upper_envelope[start:end])))] ]
            out = dict(row)
            out["method"] = "role_transport_uncertainty_boundary_v1"
            out["intervals"] = intervals
            out["raw"] = {
                **out.get("raw", {}),
                "gt_access": False,
                "module": "role_posterior_uncertainty_upper_excursion",
                "uncertainty_sources": args.method,
                "inherited_intervals": False,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
                "boundary_rule": "positive component around maximum of final residual plus cross-resolution population standard deviation",
            }
            handle.write(json.dumps(out, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
