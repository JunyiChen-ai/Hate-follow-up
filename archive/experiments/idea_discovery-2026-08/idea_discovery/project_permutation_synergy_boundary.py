#!/usr/bin/env python3
"""Parameter-free intervals from permutation-calibrated multimodal synergy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid
from scripts.idea_discovery.project_shapley_boundary import midrank
from scripts.idea_discovery.project_uncertainty_excursion_decoder import component_around_max


MAIN = ("visual", "language", "audio")
SYNERGY = ("visual_language", "visual_audio", "language_audio")


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
            z = np.log(p / (1 - p))
            nominal = midrank(z - z.mean())
            evidence = fields[key]["modality_evidence"]
            main_fields = np.stack([
                midrank(np.asarray(evidence["main_effects"][name], float))
                for name in MAIN
            ])
            synergy_fields = np.stack([
                midrank(np.asarray(evidence["synergy_fields"][name], float))
                for name in SYNERGY
            ])
            certificates = {
                "permutation_pair_complete_union_boundary_v3": (
                    nominal + main_fields.max(axis=0) + synergy_fields.max(axis=0)
                ),
            }
            duration = float(row["duration"])
            rate = float(row.get("native_rate", 4.0) or 4.0)
            for method, certificate in certificates.items():
                selected = component_around_max(certificate)
                intervals = []
                if selected is not None:
                    start, end = selected
                    intervals = [[start / rate, min(duration, end / rate),
                                  float(sigmoid(np.max(certificate[start:end])))]]
                output = dict(row)
                output["method"] = method
                output["intervals"] = intervals
                output["raw"] = {
                    **output.get("raw", {}),
                    "gt_access": False,
                    "module": "permutation_calibrated_synergy_boundary",
                    "field_method": args.fields_method,
                    "rule": method,
                    "inherited_intervals": False,
                    "floating_tie_rule": "merge_abs_1e-12_or_rel_1e-10_before_midrank",
                    "dataset_parameters": 0,
                    "label_selected_parameters": 0,
                }
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
