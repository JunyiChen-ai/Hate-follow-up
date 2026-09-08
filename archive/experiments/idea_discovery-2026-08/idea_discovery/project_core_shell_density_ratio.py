#!/usr/bin/env python3
"""Matched core-shell conditional density-ratio transport pilot."""
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
    "csdr_base_v1",
    "csdr_positive_core_v1",
    "csdr_local_shell_v1",
    "csdr_shifted_shell_v1",
    "csdr_global_background_v1",
)


def normalize_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)


def prototype(features: np.ndarray) -> np.ndarray | None:
    if not len(features):
        return None
    value = np.mean(features, axis=0)
    norm = np.linalg.norm(value)
    return value / norm if norm > 1e-9 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-method", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--tight-method", default="fact_less_t3al_dualgeo_shorter_v5")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    base_rows = load(args.base, args.base_method)
    tight_rows = load(args.bank, args.tight_method)
    coverage = {method: 0 for method in METHODS}
    mean_error = 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for key, row in sorted(base_rows.items()):
            base = np.asarray(row["score_curve"], dtype=float)
            n = len(base)
            variants = {method: base.copy() for method in METHODS}
            audit = {"available": False}
            core = first_interval(tight_rows.get(key))
            path = (args.feature_root / "bert_sentence_1fps" /
                    DATA_DIR[key[0]] / f"{key[1]}.npy")
            if core is not None and path.exists():
                features = normalize_rows(np.load(path))
                lo = max(0, min(len(features) - 1, int(np.floor(core[0]))))
                hi = max(lo + 1, min(len(features), int(np.ceil(core[1]))))
                length = max(1, hi - lo)
                core_proto = prototype(features[lo:hi])
                left = features[max(0, lo - length):lo]
                right = features[hi:min(len(features), hi + length)]
                shell = np.concatenate([left, right], axis=0) if len(left) + len(right) else left
                shell_proto = prototype(shell)
                outside = np.concatenate([features[:lo], features[hi:]], axis=0)
                global_proto = prototype(outside)
                if core_proto is not None:
                    positive = features @ core_proto
                    variants["csdr_positive_core_v1"] = transport(base, resize(rank01(positive), n))
                    if shell_proto is not None:
                        local_ratio = positive - features @ shell_proto
                        variants["csdr_local_shell_v1"] = transport(
                            base, resize(rank01(local_ratio), n)
                        )
                        # Same local shell content, deliberately assigned to a
                        # nonlocal time location before prototype construction.
                        shifted_features = np.roll(features, max(1, len(features) // 2), axis=0)
                        shifted_left = shifted_features[max(0, lo - length):lo]
                        shifted_right = shifted_features[hi:min(len(features), hi + length)]
                        shifted_shell = np.concatenate([shifted_left, shifted_right], axis=0)
                        shifted_proto = prototype(shifted_shell)
                        if shifted_proto is not None:
                            shifted_ratio = positive - features @ shifted_proto
                            variants["csdr_shifted_shell_v1"] = transport(
                                base, resize(rank01(shifted_ratio), n)
                            )
                    if global_proto is not None:
                        global_ratio = positive - features @ global_proto
                        variants["csdr_global_background_v1"] = transport(
                            base, resize(rank01(global_ratio), n)
                        )
                    audit = {
                        "available": True,
                        "core": [lo, hi],
                        "shell_frames": int(len(shell)),
                        "outside_frames": int(len(outside)),
                    }

            for method in METHODS:
                scores = variants[method]
                coverage[method] += int(not np.array_equal(scores, base))
                mean_error = max(mean_error, abs(float(np.mean(scores)) - float(np.mean(base))))
                result = dict(row)
                result["method"] = method
                result["score_curve"] = scores.tolist()
                result["raw"] = {
                    **result.get("raw", {}),
                    "gt_access": False,
                    "csdr": audit,
                    "gain": 0.01,
                    "video_mean_preserved": True,
                    "intervals_preserved": True,
                    "transport": method,
                }
                output.write(json.dumps(result, separators=(",", ":")) + "\n")

    print(json.dumps({
        "n_videos": len(base_rows),
        "coverage_nonidentical": coverage,
        "max_video_mean_error": mean_error,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
