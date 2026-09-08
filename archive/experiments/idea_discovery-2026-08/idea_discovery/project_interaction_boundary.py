#!/usr/bin/env python3
"""Decode intervals from a non-additive temporal coalition certificate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid
from scripts.idea_discovery.project_shapley_boundary import midrank
from scripts.idea_discovery.project_uncertainty_excursion_decoder import component_around_max


ROLES = ("visual", "language", "audio")
INTERACTIONS = (
    "visual_language", "visual_audio", "language_audio",
    "visual_language_audio",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--final-method", required=True)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--game-method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    final = load(args.final, args.final_method)
    game = load(args.game, args.game_method)
    with args.out.open("w") as handle:
        for key in sorted(set(final) & set(game)):
            row = final[key]
            p = np.clip(np.asarray(row["score_curve"], dtype=float), 1e-6, 1 - 1e-6)
            z = np.log(p / (1 - p))
            nominal = midrank(z - z.mean())
            evidence = game[key]["modality_evidence"]
            roles = np.stack([
                midrank(np.asarray(evidence["shapley_main_effects"][name], dtype=float))
                for name in ROLES
            ])
            interactions = np.stack([
                midrank(np.asarray(evidence["mobius_interactions"][name], dtype=float))
                for name in INTERACTIONS
            ])
            # All aggregations are symmetric and parameter-free.  The variants
            # isolate which cooperative evidence family is causally useful.
            certificates = {
                "interaction_role_union_boundary_v1": nominal + roles.max(axis=0),
                "interaction_mobius_union_boundary_v1": nominal + interactions.max(axis=0),
                "interaction_complete_union_boundary_v1": (
                    nominal + roles.max(axis=0) + interactions.max(axis=0)
                ),
                "interaction_complete_mean_boundary_v1": np.mean(
                    np.vstack([nominal, roles, interactions]), axis=0
                ),
                "interaction_nominal_control_boundary_v1": nominal,
            }
            duration = float(row["duration"])
            rate = float(row.get("native_rate", 4.0) or 4.0)
            for method, certificate in certificates.items():
                selected = component_around_max(certificate)
                intervals = []
                if selected is not None:
                    start, end = selected
                    intervals = [[
                        start / rate,
                        min(duration, end / rate),
                        float(sigmoid(np.max(certificate[start:end]))),
                    ]]
                output = dict(row)
                output["method"] = method
                output["intervals"] = intervals
                output["raw"] = {
                    **output.get("raw", {}),
                    "gt_access": False,
                    "module": "interaction_identifiable_coalition_boundary",
                    "shared_game_method": args.game_method,
                    "rule": method,
                    "inherited_intervals": False,
                    "floating_tie_rule": "merge_abs_1e-12_or_rel_1e-10_before_midrank",
                    "dataset_parameters": 0,
                    "label_selected_parameters": 0,
                }
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
