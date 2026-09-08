#!/usr/bin/env python3
"""Shared parameter-free score-to-interval decoder for protocol parity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid
from scripts.idea_discovery.project_uncertainty_excursion_decoder import component_around_max


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    rows = load(args.input, args.method)
    with args.out.open("w") as handle:
        for _, row in sorted(rows.items()):
            p = np.clip(np.asarray(row["score_curve"], dtype=float), 1e-6, 1 - 1e-6)
            z = np.log(p / (1 - p))
            residual = z - z.mean()
            selected = component_around_max(residual)
            intervals = []
            rate = float(row.get("native_rate", 4.0) or 4.0)
            duration = float(row["duration"])
            if selected is not None:
                start, end = selected
                intervals = [[start / rate, min(duration, end / rate),
                              float(sigmoid(np.max(residual[start:end])))] ]
            out = dict(row)
            out["method"] = args.output_method
            out["intervals"] = intervals
            out["raw"] = {
                **out.get("raw", {}),
                "gt_access": False,
                "decoder": "shared_positive_centered_logit_component_around_global_max",
                "inherited_intervals": False,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
            }
            handle.write(json.dumps(out, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
