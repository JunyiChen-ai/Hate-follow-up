#!/usr/bin/env python3
"""Cross-fitted dependency-aware temporal coalition game.

One opposite-fold, video-balanced empirical-Bayes model estimates the complete
class-conditional joint distribution of the three ternary evidence views.  All
eight masked coalitions are exact marginals of that *same* joint model.  Unlike
the conditionally independent LESS game, its coalition worth can contain real
pair and triple dependence terms.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from scripts.idea_discovery.run_less import collect, fit_label_model, infer_static
from scripts.idea_discovery.run_shared_coalition_game import NAMES, ALL, centered, mobius, shapley
from scripts.label_free_adapt.schema import Prediction, append_jsonl


PATTERNS = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int8)
SUBSETS = [
    frozenset(indices)
    for size in range(4)
    for indices in itertools.combinations(range(3), size)
]


def fit_joint(sequences: list[np.ndarray]) -> dict:
    """Estimate one joint model from soft LESS states without target labels."""
    teacher = fit_label_model(sequences, (0, 1, 2))
    # Jeffreys mass is fixed by the categorical model, not selected on data.
    counts = np.full((2, len(PATTERNS)), 0.5, dtype=float)
    class_mass = np.ones(2, dtype=float)
    for sequence in sequences:
        posterior = infer_static(sequence, teacher["prior"], teacher["theta"])
        weights = np.stack((1 - posterior, posterior), axis=1) / len(sequence)
        class_mass += weights.sum(axis=0)
        pattern_index = ((sequence[:, 0] + 1) * 9 + (sequence[:, 1] + 1) * 3
                         + sequence[:, 2] + 1).astype(int)
        for state in (0, 1):
            np.add.at(counts[state], pattern_index, weights[:, state])
    probabilities = counts / counts.sum(axis=1, keepdims=True)
    prior = float(class_mass[1] / class_mass.sum())
    # The teacher fixes the otherwise arbitrary latent-state orientation.
    return {"prior": prior, "probabilities": probabilities, "teacher": teacher}


def coalition_logit(evidence: np.ndarray, model: dict,
                     subset: frozenset[int]) -> np.ndarray:
    prior = float(model["prior"])
    if not subset:
        value = math.log(max(prior, 1e-12) / max(1 - prior, 1e-12))
        return np.full(len(evidence), value, dtype=float)
    indices = tuple(sorted(subset))
    likelihood = np.zeros((len(evidence), 2), dtype=float)
    for row_index, row in enumerate(evidence):
        mask = np.ones(len(PATTERNS), dtype=bool)
        for view in indices:
            mask &= PATTERNS[:, view] == row[view]
        likelihood[row_index] = model["probabilities"][:, mask].sum(axis=1)
    return (
        math.log(max(prior, 1e-12) / max(1 - prior, 1e-12))
        + np.log(np.maximum(likelihood[:, 1], 1e-12))
        - np.log(np.maximum(likelihood[:, 0], 1e-12))
    )


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
    models = {
        target_fold: fit_joint([
            row["evidence"] for row in data.values() if row["fold"] != target_fold
        ])
        for target_fold in (0, 1)
    }
    max_efficiency_error = 0.0
    max_interaction = {"pair": 0.0, "triple": 0.0}
    for key, row in sorted(data.items()):
        model = models[row["fold"]]
        values = {
            subset: coalition_logit(row["evidence"], model, subset)
            for subset in SUBSETS
        }
        effects = {NAMES[player]: shapley(values, player) for player in range(3)}
        efficiency = sum(effects.values()) - (values[ALL] - values[frozenset()])
        max_efficiency_error = max(
            max_efficiency_error, float(np.max(np.abs(efficiency)))
        )
        raw_interactions = {
            "visual_language": mobius(values, frozenset((0, 1))),
            "visual_audio": mobius(values, frozenset((0, 2))),
            "language_audio": mobius(values, frozenset((1, 2))),
            "visual_language_audio": mobius(values, ALL),
        }
        max_interaction["pair"] = max(
            max_interaction["pair"],
            *(float(np.max(np.abs(raw_interactions[name]))) for name in
              ("visual_language", "visual_audio", "language_audio")),
        )
        max_interaction["triple"] = max(
            max_interaction["triple"],
            float(np.max(np.abs(raw_interactions["visual_language_audio"]))),
        )
        full = 1 / (1 + np.exp(-np.clip(values[ALL], -40, 40)))
        append_jsonl(args.out, Prediction(
            "interaction_identifiable_coalition_game_v1", key[0], key[1],
            row["duration"], score_curve=full.tolist(), intervals=[], calls=0,
            modality_evidence={
                "fold": row["fold"],
                "shared_prior": float(model["prior"]),
                "shapley_main_effects": {
                    name: centered(value).tolist() for name, value in effects.items()
                },
                "mobius_interactions": {
                    name: centered(value).tolist()
                    for name, value in raw_interactions.items()
                },
            },
            raw={
                "gt_access": False,
                "fit": "opposite_hash_fold_video_balanced_empirical_bayes",
                "shared_model_across_all_8_coalitions": True,
                "coalition_masking": "exact_marginalization_of_one_joint_model",
                "conditional_independence": False,
                "teacher_state": "frozen_full_LESS_soft_posterior",
                "dirichlet_prior": "Jeffreys_0.5",
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
            },
        ))
    audit = {
        "n": len(data),
        "max_shapley_efficiency_error": max_efficiency_error,
        "max_abs_mobius_interaction": max_interaction,
        "fold_priors": {str(fold): float(model["prior"])
                        for fold, model in models.items()},
    }
    args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
