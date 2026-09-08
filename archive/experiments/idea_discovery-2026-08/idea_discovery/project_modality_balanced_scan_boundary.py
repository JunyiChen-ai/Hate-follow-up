#!/usr/bin/env python3
"""Parameter-free interval decoding from modality-balanced direct evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_multiscale_scan_boundary import MAIN, scan_interval
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
            evidence = fields[key]["modality_evidence"]
            coordinates = [midrank(np.asarray(evidence["main_effects"][name], float))
                           for name in MAIN]
            field = np.mean(np.stack(coordinates), axis=0)
            start, end, scan_score = scan_interval(field)
            rate = float(row.get("native_rate", 4.0) or 4.0)
            output = dict(row)
            output["method"] = "modality_balanced_penalized_scan_v1"
            output["intervals"] = [[start / rate,
                                    min(float(row["duration"]), end / rate),
                                    float(sigmoid(scan_score))]]
            output["raw"] = {
                **output.get("raw", {}),
                "gt_access": False,
                "module": "modality_balanced_penalized_interval_scan",
                "coordinates": list(MAIN),
                "coordinate_pool": "equal_mass_mean_of_tie_aware_midranks",
                "interaction_reuse": False,
                "scan": "all_contiguous_intervals_inside_outside_standardized_contrast",
                "multiplicity_penalty": "sqrt(2*log(number_of_placements_at_length))",
                "duration_prior": None,
                "score_threshold": None,
                "development_selected_design": True,
                "eligible_evidence_status": "exploratory_until_untouched_confirmation",
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
                "scan_score": scan_score,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
