#!/usr/bin/env python3
"""Alpha-free multiscale scan boundary over multimodal evidence fields.

Every possible contiguous interval competes using its standardized inside-vs-
outside evidence contrast.  The analytic square-root log penalty compensates
for the number of placements at that interval length.  The global maximizer
directly supplies both endpoints; no score threshold or duration prior is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid
from scripts.idea_discovery.project_shapley_boundary import midrank


MAIN = ("visual", "language", "audio")
PAIR = ("visual_language", "visual_audio", "language_audio")


def scan_interval(field: np.ndarray) -> tuple[int, int, float]:
    n = len(field)
    if n <= 1:
        return 0, n, float(field.max(initial=0))
    centered = field - field.mean()
    prefix = np.r_[0.0, np.cumsum(centered)]
    best = (-np.inf, 0, 1)
    # Exclude the full video: its inside/outside contrast is undefined.
    for length in range(1, n):
        sums = prefix[length:] - prefix[:-length]
        contrast = sums * np.sqrt(n / (length * (n - length)))
        multiplicity_penalty = np.sqrt(2 * np.log(max(1, n - length + 1)))
        scores = contrast - multiplicity_penalty
        start = int(np.argmax(scores))
        value = float(scores[start])
        if value > best[0]:
            best = (value, start, start + length)
    return best[1], best[2], best[0]


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
            evidence = fields[key]["modality_evidence"]
            coordinates = [midrank(z - z.mean())]
            coordinates.extend(midrank(np.asarray(evidence["main_effects"][name], float))
                               for name in MAIN)
            coordinates.extend(midrank(np.asarray(evidence["synergy_fields"][name], float))
                               for name in PAIR)
            field = np.mean(np.stack(coordinates), axis=0)
            start, end, scan_score = scan_interval(field)
            rate = float(row.get("native_rate", 4.0) or 4.0)
            duration = float(row["duration"])
            output = dict(row)
            output["method"] = "multimodal_penalized_scan_boundary_v1"
            output["intervals"] = [[start / rate, min(duration, end / rate),
                                    float(sigmoid(scan_score))]]
            output["raw"] = {
                **output.get("raw", {}),
                "gt_access": False,
                "module": "alpha_free_multiscale_evidence_scan",
                "scan": "all_contiguous_intervals_inside_outside_standardized_contrast",
                "field_scale": "mean_of_seven_midrank_coordinates_each_bounded_in_minus_half_plus_half",
                "multiplicity_penalty": "sqrt(2*log(number_of_placements_at_length))",
                "duration_prior": None,
                "score_threshold": None,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
                "scan_score": scan_score,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
