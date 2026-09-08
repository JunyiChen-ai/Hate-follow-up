#!/usr/bin/env python3
"""Development-only coordinate ablation for the penalized scan decoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_multiscale_scan_boundary import MAIN, PAIR, scan_interval
from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid
from scripts.idea_discovery.project_shapley_boundary import midrank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--final-method", required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--fields-method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    final = load(args.final, args.final_method)
    fields = load(args.fields, args.fields_method)
    with args.out.open("w") as handle:
        for key in sorted(set(final) & set(fields)):
            row = final[key]
            p = np.clip(np.asarray(row["score_curve"], float), 1e-6, 1 - 1e-6)
            nominal = midrank(np.log(p / (1 - p)))
            evidence = fields[key]["modality_evidence"]
            main = [midrank(np.asarray(evidence["main_effects"][name], float)) for name in MAIN]
            pair = [midrank(np.asarray(evidence["synergy_fields"][name], float)) for name in PAIR]
            variants = {
                "scan_ablation_nominal": [nominal],
                "scan_ablation_main": main,
                "scan_ablation_pair": pair,
                "scan_ablation_nominal_main": [nominal, *main],
                "scan_ablation_nominal_pair": [nominal, *pair],
                "scan_ablation_all": [nominal, *main, *pair],
            }
            rate = float(row.get("native_rate", 4.0) or 4.0)
            for method, coordinates in variants.items():
                field = np.mean(np.stack(coordinates), axis=0)
                start, end, score = scan_interval(field)
                output = dict(row)
                output["method"] = method
                output["intervals"] = [[start / rate, min(float(row["duration"]), end / rate),
                                        float(sigmoid(score))]]
                output["raw"] = {**output.get("raw", {}), "gt_access": False,
                                 "development_ablation": True, "coordinate_rule": method,
                                 "scan_score": score}
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
