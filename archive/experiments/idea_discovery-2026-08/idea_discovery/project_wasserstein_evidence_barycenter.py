#!/usr/bin/env python3
"""Training-free visual-language temporal evidence barycenter pilot.

Visual and timestamp-language evidence are treated as probability measures on
the normalized video timeline.  Their equal-weight 1D Wasserstein barycenter
defines a mass-transport correction to the frozen visual field.  The final
small logit update preserves video propensity exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_role_orthogonal_transport import (
    DATA_DIR,
    first_interval,
    load,
    rank01,
    resize,
    transport,
)


METHODS = (
    "webt_base_v1",
    "webt_pointwise_v1",
    "webt_barycenter_v1",
    "webt_barycenter_shift_t_v1",
    "webt_barycenter_reverse_t_v1",
)


def normalize_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)


def evidence_measure(values: np.ndarray) -> np.ndarray:
    # Rank evidence avoids a temperature parameter and is invariant to encoder scale.
    weights = rank01(np.asarray(values, dtype=float))
    return weights / np.sum(weights)


def weighted_quantiles(probability: np.ndarray, count: int) -> np.ndarray:
    cdf = np.cumsum(probability)
    levels = (np.arange(count, dtype=float) + 0.5) / count
    return np.interp(levels, cdf, np.linspace(0.0, 1.0, len(probability)))


def deposit(locations: np.ndarray, n: int) -> np.ndarray:
    positions = np.clip(locations, 0.0, 1.0) * max(1, n - 1)
    left = np.floor(positions).astype(int)
    right = np.minimum(n - 1, left + 1)
    fraction = positions - left
    mass = np.zeros(n, dtype=float)
    np.add.at(mass, left, 1.0 - fraction)
    np.add.at(mass, right, fraction)
    # Parameter-free triangular antialiasing; all measures receive the same readout.
    if n >= 3:
        mass = np.convolve(mass, np.asarray([0.25, 0.5, 0.25]), mode="same")
    mass += 1e-9
    return mass / np.sum(mass)


def barycenter(visual: np.ndarray, language: np.ndarray) -> np.ndarray:
    n = len(visual)
    count = max(256, 4 * n)
    visual_q = weighted_quantiles(visual, count)
    language_q = weighted_quantiles(language, count)
    return deposit(0.5 * (visual_q + language_q), n)


def correction(visual: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.log(np.clip(target, 1e-9, None)) - np.log(np.clip(visual, 1e-9, None))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-method", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--tight-method", default="fact_less_t3al_dualgeo_shorter_v5")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--direct-text-root", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    base_rows = load(args.base, args.base_method)
    tight_rows = load(args.bank, args.tight_method)
    coverage = {method: 0 for method in METHODS}
    max_mean_error = 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for key, row in sorted(base_rows.items()):
            base = np.asarray(row["score_curve"], dtype=float)
            n = len(base)
            variants = {method: base.copy() for method in METHODS}
            audit = {"available": False}
            core = first_interval(tight_rows.get(key))
            text_root = (args.feature_root if args.direct_text_root
                         else args.feature_root / "bert_sentence_1fps")
            path = text_root / DATA_DIR[key[0]] / f"{key[1]}.npy"
            if core is not None and path.exists():
                features = normalize_rows(np.load(path))
                lo = max(0, min(len(features) - 1, int(np.floor(core[0]))))
                hi = max(lo + 1, min(len(features), int(np.ceil(core[1]))))
                proto = np.mean(features[lo:hi], axis=0)
                proto_norm = np.linalg.norm(proto)
                if proto_norm > 1e-9:
                    proto /= proto_norm
                    language_field = resize(features @ proto, n)
                    visual_measure = evidence_measure(base)
                    language_measure = evidence_measure(language_field)
                    pointwise = 0.5 * (visual_measure + language_measure)
                    variants["webt_pointwise_v1"] = transport(
                        base, correction(visual_measure, pointwise)
                    )
                    factual = barycenter(visual_measure, language_measure)
                    variants["webt_barycenter_v1"] = transport(
                        base, correction(visual_measure, factual)
                    )
                    shifted_language = np.roll(language_measure, max(1, n // 2))
                    shifted = barycenter(visual_measure, shifted_language)
                    variants["webt_barycenter_shift_t_v1"] = transport(
                        base, correction(visual_measure, shifted)
                    )
                    reversed_target = barycenter(visual_measure, language_measure[::-1])
                    variants["webt_barycenter_reverse_t_v1"] = transport(
                        base, correction(visual_measure, reversed_target)
                    )
                    audit = {"available": True, "core": [lo, hi],
                             "quantile_count": max(256, 4 * n)}
            for method in METHODS:
                scores = variants[method]
                coverage[method] += int(not np.array_equal(scores, base))
                max_mean_error = max(max_mean_error,
                                     abs(float(np.mean(scores)) - float(np.mean(base))))
                result = dict(row)
                result["method"] = method
                result["score_curve"] = scores.tolist()
                result["raw"] = {
                    **result.get("raw", {}), "gt_access": False,
                    "webt": audit, "gain": 0.01,
                    "video_mean_preserved": True, "intervals_preserved": True,
                }
                output.write(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({"n": len(base_rows), "coverage_nonidentical": coverage,
                      "max_video_mean_error": max_mean_error}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
