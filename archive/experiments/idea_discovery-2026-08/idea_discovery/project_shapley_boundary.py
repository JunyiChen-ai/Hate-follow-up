#!/usr/bin/env python3
"""Scale-free boundaries from a final posterior and shared-game Shapley roles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid
from scripts.idea_discovery.project_uncertainty_excursion_decoder import component_around_max


ROLES = ("visual", "language", "audio")


def midrank(values: np.ndarray) -> np.ndarray:
    """Average ranks after merging floating-point-equivalent values.

    Exact coalition identities can leave cancellation residue near machine
    precision. Ranking those residues invents temporal structure, so values
    within a scale-aware numerical tolerance share one representative.
    """
    if len(values) <= 1:
        return np.zeros_like(values, dtype=float)
    values = np.asarray(values, dtype=float)
    tolerance = max(1e-12, 1e-10 * float(np.max(np.abs(values))))
    order = np.argsort(values, kind="mergesort")
    canonical = values.copy()
    group_start = 0
    for index in range(1, len(order) + 1):
        if index == len(order) or values[order[index]] - values[order[index - 1]] > tolerance:
            members = order[group_start:index]
            canonical[members] = float(np.mean(values[members]))
            group_start = index
    return (rankdata(canonical, method="average") - 1) / (len(values) - 1) - 0.5


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
    for key in sorted(set(final) & set(game)):
        row = final[key]
        p = np.clip(np.asarray(row["score_curve"], dtype=float), 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p))
        nominal = midrank(z - z.mean())
        stored = game[key]["modality_evidence"]["shapley_main_effects"]
        roles = np.stack([midrank(np.asarray(stored[name], dtype=float)) for name in ROLES])
        certificates = {
            "shapley_role_union_boundary_v1": nominal + roles.max(axis=0),
            "shapley_role_mean_boundary_v1": np.mean(np.vstack([nominal, roles]), axis=0),
            "shapley_role_bottleneck_boundary_v1": np.min(np.vstack([nominal, roles]), axis=0),
            "nominal_rank_boundary_control_v1": nominal,
        }
        duration = float(row["duration"])
        rate = float(row.get("native_rate", 4.0) or 4.0)
        for method, certificate in certificates.items():
            selected = component_around_max(certificate)
            intervals = []
            if selected is not None:
                start, end = selected
                intervals = [[start / rate, min(duration, end / rate),
                              float(sigmoid(np.max(certificate[start:end])))] ]
            output = dict(row)
            output["method"] = method
            output["intervals"] = intervals
            output["raw"] = {
                **output.get("raw", {}),
                "gt_access": False,
                "module": "shared_game_shapley_role_boundary",
                "shared_game_method": args.game_method,
                "rule": method,
                "inherited_intervals": False,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
                "floating_tie_rule": "merge_abs_1e-12_or_rel_1e-10_before_midrank",
            }
            with args.out.open("a") as handle:
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
