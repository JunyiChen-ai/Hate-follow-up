#!/usr/bin/env python3
"""Cyclic-group maxT fields jointly calibrated over time and scale.

Under the explicit null that relative modality phase is invariant to the cyclic
shift group, each pair uses the complete group orbit.  For every shift we take
one maximum over all valid temporal positions and all dyadic scales.  Observed
local statistics are then ranked against this same maxT reference, yielding a
single multiplicity-adjusted field per pair without an alpha threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.run_less import collect
from scripts.idea_discovery.run_permutation_synergy_fields import NAMES, support
from scripts.idea_discovery.run_local_maxstat_synergy_fields import moving_mean, expand_centers
from scripts.label_free_adapt.schema import Prediction, append_jsonl


PAIRS = ((0, 1, "visual_language"), (0, 2, "visual_audio"),
         (1, 2, "language_audio"))


def dyadic_scales(length: int) -> list[int]:
    upper = max(1, length // 4)
    scales = []
    width = 1
    while width <= upper:
        scales.append(width)
        width *= 2
    return scales or [1]


def tolerant_midcdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Tie-aware empirical position after merging numerical equivalents."""
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    magnitude = max(float(np.max(np.abs(reference), initial=0.0)),
                    float(np.max(np.abs(values), initial=0.0)))
    tolerance = max(1e-12, 1e-10 * magnitude)
    ordered = np.sort(reference)
    lower = np.searchsorted(ordered, values - tolerance, side="left")
    upper = np.searchsorted(ordered, values + tolerance, side="right")
    return (0.5 * (lower + upper) + 0.5) / (len(ordered) + 1)


def pair_field(first: np.ndarray, second: np.ndarray,
               scales: list[int]) -> np.ndarray:
    n = len(first)
    # Null-player axiom: a temporally constant modality contains no localization
    # information, so it cannot contribute a positive or negative interaction.
    # Use the same numerical-equivalence rule as the empirical CDF, rather than
    # a dataset-tuned variance threshold.
    magnitude = max(float(np.max(np.abs(first), initial=0.0)),
                    float(np.max(np.abs(second), initial=0.0)))
    tolerance = max(1e-12, 1e-10 * magnitude)
    if float(np.ptp(first)) <= tolerance or float(np.ptp(second)) <= tolerance:
        return np.zeros(n, dtype=float)
    scale_maxima = np.empty((n, len(scales)), dtype=float)
    for shift in range(n):
        shifted = np.roll(second, shift)
        product = first * shifted
        for scale_index, width in enumerate(scales):
            scale_maxima[shift, scale_index] = float(
                np.max(moving_mean(product, width))
            )
    # Studentize every scale by its own complete-orbit empirical distribution
    # before the joint max, preventing the finest scale from dominating merely
    # because nonnegative moving averages shrink with width.
    standardized_maxima = np.column_stack([
        tolerant_midcdf(scale_maxima[:, index], scale_maxima[:, index])
        for index in range(len(scales))
    ])
    joint_reference = np.max(standardized_maxima, axis=1)
    adjusted_scale_fields = []
    aligned_product = first * second
    for scale_index, width in enumerate(scales):
        observed = moving_mean(aligned_product, width)
        within_scale = tolerant_midcdf(scale_maxima[:, scale_index], observed)
        adjusted = tolerant_midcdf(joint_reference, within_scale) - 0.5
        adjusted_scale_fields.append(expand_centers(adjusted, width, n))
    return np.max(np.stack(adjusted_scale_fields), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a10", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--audio-mode", choices=("absolute", "within_video"),
                        default="within_video")
    parser.add_argument("--visual-mode", choices=("majority", "a10", "unanimous"),
                        default="majority")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    args.text_shift = "aligned"
    data = collect(args)
    scale_histogram: dict[int, int] = {}
    for key, row in sorted(data.items()):
        streams = [support(row["evidence"][:, index]) for index in range(3)]
        scales = dyadic_scales(len(streams[0]))
        scale_histogram[len(scales)] = scale_histogram.get(len(scales), 0) + 1
        fields = {
            name: pair_field(streams[first], streams[second], scales).tolist()
            for first, second, name in PAIRS
        }
        append_jsonl(args.out, Prediction(
            "cyclic_studentized_maxT_synergy_fields_v2", key[0], key[1], row["duration"],
            score_curve=np.mean(np.stack(streams), axis=0).tolist(),
            intervals=[], calls=0,
            modality_evidence={
                "main_effects": {name: (stream - stream.mean()).tolist()
                                 for name, stream in zip(NAMES, streams)},
                "synergy_fields": fields,
                "dyadic_scales": scales,
            },
            raw={
                "gt_access": False,
                "null_assumption": "relative_phase_invariant_under_complete_cyclic_group",
                "reference": "complete_n_shift_orbit_including_zero",
                "multiplicity": "scale_studentized_joint_maxT_over_positions_and_dyadic_scales",
                "field": "tie_aware_empirical_position_against_maxT_reference",
                "floating_tie_rule": "merge_abs_1e-12_or_rel_1e-10",
                "null_player_axiom": "constant_within_tolerance_implies_zero_interaction",
                "visual_mode": args.visual_mode,
                "alpha_threshold": None,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
            },
        ))
    audit = {"n": len(data), "scale_count_histogram": scale_histogram}
    args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
