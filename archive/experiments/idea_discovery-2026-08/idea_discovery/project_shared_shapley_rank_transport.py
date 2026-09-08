#!/usr/bin/env python3
"""Exact rank transport from a shared null-complete temporal coalition game."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid


ROLES = ("visual", "language", "audio")


def centered(values: np.ndarray) -> np.ndarray:
    return values - values.mean()


def midrank(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=float)
    return (rankdata(values, method="average") - 1) / (len(values) - 1) - 0.5


def exact_transport(reference: np.ndarray, coordinate: np.ndarray, key: tuple[str, str]) -> np.ndarray:
    """Return an exact permutation; ties use nominal evidence then hash, never time."""
    digest = hashlib.sha256(("shared-shapley/" + "/".join(key)).encode()).digest()
    seed = int.from_bytes(digest[:8], "little")
    random_tie = np.random.default_rng(seed).random(len(reference))
    order = np.lexsort((random_tie, reference, coordinate))
    output = np.empty_like(reference)
    output[order] = np.sort(reference)
    return output


def orbit_authority(reference_rank: np.ndarray, role_rank: np.ndarray) -> tuple[float, float]:
    """Exact within-video circular-null authority in [0,1]."""
    n = len(reference_rank)
    if n < 2 or np.all(role_rank == role_rank[0]):
        return 0.0, 0.5
    correlations = np.asarray([
        float(np.mean(reference_rank * np.roll(role_rank, shift)))
        for shift in range(n)
    ])
    # Average rank handles equal correlations without a temporal tie break.
    rank = float(rankdata(correlations, method="average")[0] / n)
    return max(0.0, 2.0 * rank - 1.0), rank


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
    keys = sorted(set(game) & set(nominal))
    with args.out.open("w") as handle:
        for key in keys:
            row = nominal[key]
            p = np.clip(np.asarray(row["score_curve"], dtype=float), 1e-6, 1 - 1e-6)
            z = np.log(p / (1 - p))
            reference = centered(z)
            stored = game[key].get("modality_evidence", {}).get("shapley_main_effects", {})
            effects = {name: np.asarray(stored[name], dtype=float) for name in ROLES}
            stored_interactions = game[key].get("modality_evidence", {}).get("mobius_interactions", {})
            interactions = {name: np.asarray(values, dtype=float)
                            for name, values in stored_interactions.items()}
            if any(len(values) != len(reference) for values in effects.values()):
                raise ValueError(f"unaligned Shapley fields at {key}")
            variants = [
                ("shared_shapley_rank_transport_v1", ROLES, 0),
                ("shared_shapley_no_visual_v1", ("language", "audio"), 0),
                ("shared_shapley_no_language_v1", ("visual", "audio"), 0),
                ("shared_shapley_no_audio_v1", ("visual", "language"), 0),
                ("shared_shapley_halfshift_control_v1", ROLES, len(reference) // 2),
            ]
            for method, included, shift in variants:
                coordinates = [midrank(reference)]
                coordinates.extend(midrank(np.roll(effects[name], shift)) for name in included)
                consensus = np.mean(np.stack(coordinates), axis=0)
                residual = exact_transport(reference, consensus, key)
                output = dict(row)
                output["method"] = method
                output["score_curve"] = sigmoid(float(z.mean()) + residual).tolist()
                output["raw"] = {
                    **output.get("raw", {}),
                    "gt_access": False,
                    "module": "shared_null_complete_shapley_rank_transport",
                    "shared_game_method": args.game_method,
                    "coalitions": 8,
                    "roles": ["nominal", *included],
                    "role_weights": [1 / len(coordinates)] * len(coordinates),
                    "tie_rule": "average_midrank_then_nominal_then_hash",
                    "transport": "exact_order_statistic_permutation",
                    "empirical_marginal_preserved_exactly": True,
                    "video_propensity_preserved": True,
                    "temporal_role_shift": shift,
                    "dataset_parameters": 0,
                    "label_selected_parameters": 0,
                    "inherited_intervals_for_frame_mechanism_test": True,
                }
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")

            interaction_variants = [
                ("shared_mobius_interaction_transport_v1", tuple(interactions), 0, False),
                ("shared_mobius_triad_transport_v1", ("visual_language_audio",), 0, False),
                ("shared_mobius_orbit_authorized_transport_v1", tuple(interactions), 0, True),
                ("shared_mobius_orbit_authorized_halfshift_control_v1",
                 tuple(interactions), len(reference) // 2, True),
            ]
            for method, included, shift, authorize in interaction_variants:
                nominal_rank = midrank(reference)
                interaction_ranks = {name: midrank(np.roll(interactions[name], shift))
                                     for name in included}
                if authorize:
                    weights = {name: orbit_authority(nominal_rank, values)[0]
                               for name, values in interaction_ranks.items()}
                else:
                    weights = {name: 1.0 for name in included}
                denominator = 1.0 + sum(weights.values())
                consensus = nominal_rank.copy()
                for name, values in interaction_ranks.items():
                    consensus += weights[name] * values
                consensus /= denominator
                residual = exact_transport(reference, consensus, key)
                output = dict(row)
                output["method"] = method
                output["score_curve"] = sigmoid(float(z.mean()) + residual).tolist()
                output["raw"] = {
                    **output.get("raw", {}),
                    "gt_access": False,
                    "module": "shared_null_complete_mobius_interaction_transport",
                    "interaction_terms": list(included),
                    "interaction_weights": weights,
                    "orbit_authorized": authorize,
                    "temporal_interaction_shift": shift,
                    "transport": "exact_order_statistic_permutation",
                    "empirical_marginal_preserved_exactly": True,
                    "video_propensity_preserved": True,
                    "dataset_parameters": 0,
                    "label_selected_parameters": 0,
                    "inherited_intervals_for_frame_mechanism_test": True,
                }
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")

            for method, shift in (
                ("shared_shapley_orbit_authorized_transport_v2", 0),
                ("shared_shapley_orbit_authorized_halfshift_control_v2", len(reference) // 2),
            ):
                nominal_rank = midrank(reference)
                role_ranks = {name: midrank(np.roll(values, shift))
                              for name, values in effects.items()}
                authority = {name: orbit_authority(nominal_rank, values)
                             for name, values in role_ranks.items()}
                numerator = nominal_rank.copy()
                denominator = 1.0
                for name in ROLES:
                    weight = authority[name][0]
                    numerator += weight * role_ranks[name]
                    denominator += weight
                consensus = numerator / denominator
                residual = exact_transport(reference, consensus, key)
                output = dict(row)
                output["method"] = method
                output["score_curve"] = sigmoid(float(z.mean()) + residual).tolist()
                output["raw"] = {
                    **output.get("raw", {}),
                    "gt_access": False,
                    "module": "shared_shapley_exact_orbit_authorized_transport",
                    "shared_game_method": args.game_method,
                    "coalitions": 8,
                    "nominal_weight": 1.0,
                    "role_authority": {name: {"weight": values[0], "orbit_rank": values[1]}
                                       for name, values in authority.items()},
                    "authority_rule": "positive excess over exact circular-null median rank",
                    "tie_rule": "average_midrank_then_nominal_then_hash",
                    "transport": "exact_order_statistic_permutation",
                    "empirical_marginal_preserved_exactly": True,
                    "video_propensity_preserved": True,
                    "temporal_role_shift": shift,
                    "dataset_parameters": 0,
                    "label_selected_parameters": 0,
                    "inherited_intervals_for_frame_mechanism_test": True,
                }
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
