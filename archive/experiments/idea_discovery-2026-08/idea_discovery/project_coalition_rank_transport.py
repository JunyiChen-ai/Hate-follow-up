#!/usr/bin/env python3
"""Scale-free multimodal role transport from LESS leave-one-view coalitions.

Pairwise coalition inversion recovers one centered temporal contribution for
visual, audio, and language evidence. Average midranks place all roles on the
same bounded empirical-null scale. Their equal consensus with the nominal
posterior rank is transported back through the nominal residual distribution,
preserving its propensity and marginal scale while changing temporal order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid


METHODS = {
    "va": "less_3v_no_language_majority_global_aligned_within_video_v2",
    "vl": "less_3v_no_audio_majority_global_aligned_within_video_v2",
    "la": "less_3v_no_visual_majority_global_aligned_within_video_v2",
}


def logits(row: dict) -> np.ndarray:
    p = np.clip(np.asarray(row["score_curve"], dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def centered(values: np.ndarray) -> np.ndarray:
    return values - values.mean()


def midrank(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=float)
    # Average ranks are mandatory: ordinal/stable ranks turn repeated chunks
    # into a spurious time ramp.
    return (rankdata(values, method="average") - 1) / (len(values) - 1) - 0.5


def transport(reference: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    if len(reference) <= 1:
        return reference.copy()
    quantiles = (rankdata(coordinate, method="average") - 0.5) / len(coordinate)
    ordered = np.sort(reference)
    grid = (np.arange(len(reference)) + 0.5) / len(reference)
    output = np.interp(quantiles, grid, ordered)
    return centered(output)


def exact_transport(
    reference: np.ndarray, coordinate: np.ndarray, key: tuple[str, str]
) -> np.ndarray:
    """Permute nominal order statistics exactly without using temporal order."""
    digest = hashlib.sha256(("coalition-rank/" + "/".join(key)).encode()).digest()
    random_tie = np.random.default_rng(int.from_bytes(digest[:8], "little")).random(
        len(reference)
    )
    # Consensus is primary. Nominal evidence and a content-independent hash are
    # used only to resolve consensus ties; the frame index is never a tie-break.
    order = np.lexsort((random_tie, reference, coordinate))
    output = np.empty_like(reference)
    output[order] = np.sort(reference)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coalitions", type=Path, required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--nominal-method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    coalition = {name: load(args.coalitions, method) for name, method in METHODS.items()}
    nominal = load(args.nominal, args.nominal_method)
    keys = sorted(set(nominal).intersection(*(set(rows) for rows in coalition.values())))
    with args.out.open("w") as handle:
        for key in keys:
            z0 = logits(nominal[key])
            pair = {name: centered(logits(rows[key])) for name, rows in coalition.items()}
            if any(len(values) != len(z0) for values in pair.values()):
                raise ValueError(f"unaligned coalition at {key}")
            roles = {
                "visual": centered(0.5 * (pair["va"] + pair["vl"] - pair["la"])),
                "audio": centered(0.5 * (pair["va"] + pair["la"] - pair["vl"])),
                "language": centered(0.5 * (pair["vl"] + pair["la"] - pair["va"])),
            }
            nominal_residual = centered(z0)
            variants = [
                ("coalition_rank_transport_v1", tuple(roles), 0),
                ("coalition_rank_transport_no_visual_v1", ("audio", "language"), 0),
                ("coalition_rank_transport_no_audio_v1", ("visual", "language"), 0),
                ("coalition_rank_transport_no_language_v1", ("visual", "audio"), 0),
                ("coalition_rank_transport_halfshift_control_v1", tuple(roles), len(z0) // 2),
            ]
            for method, included, shift in variants:
                role_ranks = [midrank(np.roll(roles[name], shift)) for name in included]
                consensus = np.mean(np.stack([midrank(nominal_residual), *role_ranks]), axis=0)
                residual = transport(nominal_residual, consensus)
                row = dict(nominal[key])
                row["method"] = method
                row["score_curve"] = sigmoid(float(z0.mean()) + residual).tolist()
                row["raw"] = {
                    **row.get("raw", {}),
                    "gt_access": False,
                    "module": "coalition_inverted_midrank_transport",
                    "roles": ["nominal", *included],
                    "role_weights": [1 / (1 + len(included))] * (1 + len(included)),
                    "tie_rule": "average_midrank",
                    "marginal_reference": "nominal_temporal_residual",
                    "video_propensity_preserved": True,
                    "dataset_parameters": 0,
                    "label_selected_parameters": 0,
                    "temporal_role_shift": shift,
                    "intervals_preserved_for_frame_mechanism_test": True,
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")

            exact_variants = [
                ("coalition_rank_exact_transport_v2", tuple(roles), 0),
                ("coalition_rank_exact_no_visual_v2", ("audio", "language"), 0),
                ("coalition_rank_exact_no_audio_v2", ("visual", "language"), 0),
                ("coalition_rank_exact_no_language_v2", ("visual", "audio"), 0),
                ("coalition_rank_exact_halfshift_control_v2", tuple(roles), len(z0) // 2),
            ]
            for method, included, shift in exact_variants:
                role_ranks = [midrank(np.roll(roles[name], shift)) for name in included]
                consensus = np.mean(
                    np.stack([midrank(nominal_residual), *role_ranks]), axis=0
                )
                residual = exact_transport(nominal_residual, consensus, key)
                row = dict(nominal[key])
                row["method"] = method
                row["score_curve"] = sigmoid(float(z0.mean()) + residual).tolist()
                row["raw"] = {
                    **row.get("raw", {}),
                    "gt_access": False,
                    "module": "coalition_inverted_exact_rank_transport",
                    "roles": ["nominal", *included],
                    "role_weights": [1 / (1 + len(included))] * (1 + len(included)),
                    "tie_rule": "average_midrank_then_nominal_then_hash",
                    "marginal_reference": "nominal_temporal_residual",
                    "transport": "exact_order_statistic_permutation",
                    "empirical_marginal_preserved_exactly": True,
                    "video_propensity_preserved": True,
                    "dataset_parameters": 0,
                    "label_selected_parameters": 0,
                    "temporal_role_shift": shift,
                    "intervals_preserved_for_frame_mechanism_test": True,
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
