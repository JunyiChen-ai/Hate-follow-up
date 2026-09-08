#!/usr/bin/env python3
"""Continuous quantile transport from an interaction-identifiable game.

This is deliberately described as interpolation, not as an exact permutation
or exact empirical-marginal preservation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_coalition_rank_transport import centered, midrank, transport
from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid


ROLES = ("visual", "language", "audio")
INTERACTIONS = (
    "visual_language", "visual_audio", "language_audio",
    "visual_language_audio",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--game-method", required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--nominal-method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    game = load(args.game, args.game_method)
    nominal = load(args.nominal, args.nominal_method)
    with args.out.open("w") as handle:
        for key in sorted(set(game) & set(nominal)):
            row = nominal[key]
            p = np.clip(np.asarray(row["score_curve"], dtype=float), 1e-6, 1 - 1e-6)
            z = np.log(p / (1 - p))
            reference = centered(z)
            evidence = game[key]["modality_evidence"]
            roles = [midrank(np.asarray(evidence["shapley_main_effects"][name], float))
                     for name in ROLES]
            interactions = [midrank(np.asarray(evidence["mobius_interactions"][name], float))
                            for name in INTERACTIONS]
            pair_interactions = interactions[:3]
            triple_interaction = interactions[3:]
            variants = {
                "interaction_shapley_quantile_transport_v1": roles,
                "interaction_mobius_quantile_transport_v1": interactions,
                "interaction_complete_quantile_transport_v1": roles + interactions,
                "interaction_pair_quantile_transport_v1": pair_interactions,
                "interaction_shapley_pair_quantile_transport_v1": roles + pair_interactions,
                "interaction_shapley_triple_quantile_transport_v1": roles + triple_interaction,
            }
            for method, fields in variants.items():
                consensus = np.mean(np.stack([midrank(reference), *fields]), axis=0)
                residual = transport(reference, consensus)
                output = dict(row)
                output["method"] = method
                output["score_curve"] = sigmoid(float(z.mean()) + residual).tolist()
                output["raw"] = {
                    **output.get("raw", {}),
                    "gt_access": False,
                    "module": "interaction_identifiable_continuous_quantile_transport",
                    "shared_game_method": args.game_method,
                    "roles": ["nominal"] + [f"field_{index}" for index in range(len(fields))],
                    "transport": "midquantile_linear_interpolation",
                    "empirical_marginal_preserved_exactly": False,
                    "video_propensity_preserved": True,
                    "dataset_parameters": 0,
                    "label_selected_parameters": 0,
                    "inherited_intervals_for_frame_mechanism_test": True,
                }
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
