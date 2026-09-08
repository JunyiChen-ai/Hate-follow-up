#!/usr/bin/env python3
"""Exact cyclic temporal-maxT cross-modal witness fields.

For each modality pair, the complete cyclic phase orbit supplies a per-video
null that preserves both marginal values and within-modality autocorrelation.
Each orbit member contributes its maximum pointwise conjunction over time.
Aligned local conjunctions are ranked against that single temporal-maxT
reference.  Scale search is intentionally left to the interval decoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.run_less import collect
from scripts.idea_discovery.run_permutation_synergy_fields import NAMES, support
from scripts.label_free_adapt.schema import Prediction, append_jsonl


PAIRS = ((0, 1, "visual_language"), (0, 2, "visual_audio"),
         (1, 2, "language_audio"))


def numerical_tolerance(*arrays: np.ndarray) -> float:
    magnitude = max(float(np.max(np.abs(array), initial=0.0)) for array in arrays)
    return max(1e-12, 1e-10 * magnitude)


def tolerant_midcdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Finite-orbit mid-CDF after merging numerical equivalents."""
    reference = np.asarray(reference, dtype=float)
    values = np.asarray(values, dtype=float)
    tolerance = numerical_tolerance(reference, values)
    ordered = np.sort(reference)
    lower = np.searchsorted(ordered, values - tolerance, side="left")
    upper = np.searchsorted(ordered, values + tolerance, side="right")
    return (0.5 * (lower + upper) + 0.5) / (len(ordered) + 1)


def temporal_maxt_field(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Multiplicity-adjusted aligned coincidence under a complete phase orbit."""
    n = len(first)
    tolerance = numerical_tolerance(first, second)
    # Null-player axiom: temporally invariant evidence cannot localize an
    # interaction, irrespective of its absolute support level.
    if float(np.ptp(first)) <= tolerance or float(np.ptp(second)) <= tolerance:
        return np.zeros(n, dtype=float)
    orbit_maxima = np.asarray([
        np.max(first * np.roll(second, shift)) for shift in range(n)
    ], dtype=float)
    aligned = first * second
    return tolerant_midcdf(orbit_maxima, aligned) - 0.5


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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    args.visual_mode = "majority"
    args.text_shift = "aligned"
    data = collect(args)
    null_player_pairs = 0
    for key, row in sorted(data.items()):
        streams = [support(row["evidence"][:, index]) for index in range(3)]
        fields = {}
        for first, second, name in PAIRS:
            if (float(np.ptp(streams[first])) <= numerical_tolerance(streams[first]) or
                    float(np.ptp(streams[second])) <= numerical_tolerance(streams[second])):
                null_player_pairs += 1
            fields[name] = temporal_maxt_field(
                streams[first], streams[second]).tolist()
        append_jsonl(args.out, Prediction(
            "cyclic_temporal_maxT_witness_fields_v1", key[0], key[1],
            row["duration"],
            score_curve=np.mean(np.stack(streams), axis=0).tolist(),
            intervals=[], calls=0,
            modality_evidence={
                "main_effects": {name: (stream - stream.mean()).tolist()
                                 for name, stream in zip(NAMES, streams)},
                "synergy_fields": fields,
            },
            raw={
                "gt_access": False,
                "null_assumption": "relative_phase_invariant_under_complete_cyclic_group",
                "reference": "complete_n_shift_orbit_including_zero",
                "multiplicity": "single_max_over_all_temporal_positions_per_pair",
                "field": "aligned_pointwise_conjunction_midcdf_against_temporal_maxT",
                "floating_tie_rule": "merge_abs_1e-12_or_rel_1e-10",
                "null_player_axiom": "constant_within_tolerance_implies_zero_interaction",
                "scale_search": "delegated_to_interval_decoder",
                "alpha_threshold": None,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
            },
        ))
    audit = {"n": len(data), "null_player_pairs": null_player_pairs}
    args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
