#!/usr/bin/env python3
"""Joint tri-modal C_n x C_n randomization witness fields.

Visual phase is fixed while language and audio are shifted independently over
the complete product group.  Every orbit state contributes one maximum across
all three modality pairs and all temporal positions.  Inclusive upper-tail
counts therefore reference an aligned local pair conjunction against one joint
cross-pair/time family under the explicit independent-relative-phase null.
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


def tolerance(*arrays: np.ndarray) -> float:
    magnitude = max(float(np.max(np.abs(array), initial=0.0)) for array in arrays)
    return max(1e-12, 1e-10 * magnitude)


def shift_maxima(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.asarray([np.max(first * np.roll(second, shift))
                       for shift in range(len(first))], dtype=float)


def joint_reference(streams: list[np.ndarray]) -> np.ndarray:
    """Construct the C_n x C_n joint maximum without an O(n^3) loop."""
    n = len(streams[0])
    visual_language = shift_maxima(streams[0], streams[1])
    visual_audio = shift_maxima(streams[0], streams[2])
    language_audio = shift_maxima(streams[1], streams[2])
    language_shift = np.arange(n)[:, None]
    audio_shift = np.arange(n)[None, :]
    relative = (audio_shift - language_shift) % n
    reference = np.maximum(visual_language[:, None], visual_audio[None, :])
    reference = np.maximum(reference, language_audio[relative])
    return np.sort(reference.reshape(-1))


def inclusive_upper_tail(sorted_reference: np.ndarray,
                         values: np.ndarray) -> np.ndarray:
    numerical_tolerance = tolerance(sorted_reference, values)
    # Values within numerical tolerance count as ties in the conservative
    # upper tail. Identity is included because the complete product group is.
    lower = np.searchsorted(sorted_reference, values - numerical_tolerance,
                            side="left")
    return (len(sorted_reference) - lower) / len(sorted_reference)


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
    audit = {"n": len(data), "null_player_pairs": 0,
             "orbit_states": 0, "max_orbit_states": 0}
    for key, row in sorted(data.items()):
        streams = [support(row["evidence"][:, index]) for index in range(3)]
        reference = joint_reference(streams)
        audit["orbit_states"] += len(reference)
        audit["max_orbit_states"] = max(audit["max_orbit_states"], len(reference))
        fields = {}
        for first, second, name in PAIRS:
            numerical_tolerance = tolerance(streams[first], streams[second])
            if (float(np.ptp(streams[first])) <= numerical_tolerance or
                    float(np.ptp(streams[second])) <= numerical_tolerance):
                audit["null_player_pairs"] += 1
                fields[name] = np.zeros(len(streams[first]), dtype=float).tolist()
            else:
                pvalue = inclusive_upper_tail(
                    reference, streams[first] * streams[second])
                fields[name] = (0.5 - pvalue).tolist()
        append_jsonl(args.out, Prediction(
            "joint_trimodal_product_orbit_witness_v1", key[0], key[1],
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
                "null_assumption": "independent_language_audio_phase_invariance_given_visual_phase",
                "group": "complete_Cn_product_Cn_including_identity",
                "joint_family": "three_modality_pairs_by_all_temporal_positions",
                "reference": "single_joint_maximum_per_product_group_state",
                "tail_count": "inclusive_conservative_upper_tail",
                "orbit_states": len(reference),
                "null_player_axiom": "constant_within_tolerance_implies_zero_interaction",
                "visual_mode": args.visual_mode,
                "scale_search": "delegated_to_interval_decoder",
                "alpha_threshold": None,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
            },
        ))
    args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
