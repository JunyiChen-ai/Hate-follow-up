#!/usr/bin/env python3
"""Cross-fitted, null-complete temporal coalition game under one LESS model.

All eight modality coalitions share the full model's prior and per-view
likelihood parameters. Exact Shapley main effects and Möbius interactions are
therefore computed on a common logit scale rather than contrasted across
separately fitted subset models.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from scripts.idea_discovery.run_less import collect, fit_label_model, infer_static
from scripts.label_free_adapt.schema import Prediction, append_jsonl


NAMES = ("visual", "language", "audio")
ALL = frozenset(range(3))


def coalition_logit(evidence: np.ndarray, model: dict, subset: frozenset[int]) -> np.ndarray:
    if not subset:
        prior = float(model["prior"])
        value = math.log(max(prior, 1e-9) / max(1 - prior, 1e-9))
        return np.full(len(evidence), value, dtype=float)
    indices = tuple(sorted(subset))
    posterior = infer_static(
        evidence[:, indices], model["prior"], model["theta"][list(indices)]
    )
    posterior = np.clip(posterior, 1e-9, 1 - 1e-9)
    return np.log(posterior / (1 - posterior))


def shapley(values: dict[frozenset[int], np.ndarray], player: int) -> np.ndarray:
    output = np.zeros_like(next(iter(values.values())))
    others = sorted(ALL - {player})
    for size in range(3):
        for subset_tuple in itertools.combinations(others, size):
            subset = frozenset(subset_tuple)
            weight = math.factorial(size) * math.factorial(2 - size) / math.factorial(3)
            output += weight * (values[subset | {player}] - values[subset])
    return output


def mobius(values: dict[frozenset[int], np.ndarray], subset: frozenset[int]) -> np.ndarray:
    output = np.zeros_like(next(iter(values.values())))
    members = sorted(subset)
    for size in range(len(members) + 1):
        for inner_tuple in itertools.combinations(members, size):
            inner = frozenset(inner_tuple)
            output += ((-1) ** (len(subset) - len(inner))) * values[inner]
    return output


def centered(values: np.ndarray) -> np.ndarray:
    return values - values.mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a10", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--audio-mode", choices=("absolute", "within_video"), default="within_video")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    # Reuse the audited evidence construction, but fit exactly one full model
    # per opposite hash fold and reuse it for every subset.
    args.visual_mode = "majority"
    args.text_shift = "aligned"
    data = collect(args)
    models = {}
    for target_fold in (0, 1):
        training = [row["evidence"] for row in data.values() if row["fold"] != target_fold]
        models[target_fold] = fit_label_model(training, (0, 1, 2))

    subsets = [frozenset(c) for size in range(4) for c in itertools.combinations(range(3), size)]
    max_efficiency_error = 0.0
    for key, row in sorted(data.items()):
        model = models[row["fold"]]
        values = {subset: coalition_logit(row["evidence"], model, subset) for subset in subsets}
        effects = {NAMES[index]: shapley(values, index) for index in range(3)}
        efficiency = sum(effects.values()) - (values[ALL] - values[frozenset()])
        max_efficiency_error = max(max_efficiency_error, float(np.max(np.abs(efficiency))))
        interactions = {
            "visual_language": centered(mobius(values, frozenset((0, 1)))).tolist(),
            "visual_audio": centered(mobius(values, frozenset((0, 2)))).tolist(),
            "language_audio": centered(mobius(values, frozenset((1, 2)))).tolist(),
            "visual_language_audio": centered(mobius(values, ALL)).tolist(),
        }
        full = 1 / (1 + np.exp(-np.clip(values[ALL], -40, 40)))
        append_jsonl(args.out, Prediction(
            "shared_null_complete_coalition_game_v1", key[0], key[1], row["duration"],
            score_curve=full.tolist(), intervals=[], calls=0,
            modality_evidence={
                "fold": row["fold"],
                "shared_prior": float(model["prior"]),
                "shared_reliability": dict(zip(NAMES, model["reliability"])),
                "shapley_main_effects": {name: centered(value).tolist() for name, value in effects.items()},
                "mobius_interactions": interactions,
            },
            raw={
                "gt_access": False,
                "fit": "opposite_hash_fold",
                "fit_scope": "global",
                "shared_model_across_all_8_coalitions": True,
                "coalitions": 8,
                "audio_mode": args.audio_mode,
                "orientation": "predeclared_support_polarity",
            },
        ))
    audit = {
        "n": len(data),
        "max_shapley_efficiency_error": max_efficiency_error,
        "fold_models": {
            str(fold): {
                "prior": float(model["prior"]),
                "reliability": dict(zip(NAMES, model["reliability"])),
            }
            for fold, model in models.items()
        },
    }
    args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
