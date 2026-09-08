#!/usr/bin/env python3
"""TRIARC M1 kill pilot: exact per-video multimodal orbit calibration.

The module consumes frozen V/A/T trajectories and emits only synchronization
reliabilities.  It never reads hate labels, dataset-level thresholds, gains,
top-k values, or temporal windows.  Reliability is the average-tie rank of the
factual zero-lag correlation among every circular shift of the second signal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


DATA_DIR = {
    "HateMM": "hatemm",
    "HateClipSeg": "hateclipseg",
    "MHC": "mhclip_en",
    "MHC_zh": "mhclip_zh",
}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def load_feature(root: Path, folder: str, dataset: str, video_id: str) -> np.ndarray | None:
    path = root / folder / DATA_DIR[dataset] / f"{video_id}.npy"
    if not path.exists():
        return None
    value = np.asarray(np.load(path), dtype=np.float64)
    return value if value.ndim == 2 and len(value) >= 3 else None


def novelty(value: np.ndarray) -> np.ndarray:
    """Tie-aware rank of adjacent cosine change, with the first point tied to the second."""
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    unit = value / np.maximum(norm, 1e-12)
    change = np.empty(len(value), dtype=np.float64)
    change[1:] = 1.0 - np.sum(unit[1:] * unit[:-1], axis=1)
    change[0] = change[1]
    return rankdata(change, method="average") / (len(change) + 1.0)


def standardize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    centered = value - value.mean()
    scale = np.linalg.norm(centered)
    return centered / scale if scale > 1e-12 else np.zeros_like(centered)


def orbit(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(len(a), len(b))
    # Modalities use different native grids (e.g. 128 uniform visual bins and
    # 1-fps A/T).  Align by normalized video time; truncation would compare
    # different physical moments.
    grid = np.linspace(0.0, 1.0, n)
    a = np.interp(grid, np.linspace(0.0, 1.0, len(a)), a)
    b = np.interp(grid, np.linspace(0.0, 1.0, len(b)), b)
    a, b = standardize(a), standardize(b)
    if n < 3 or not np.any(a) or not np.any(b):
        return {"available": False, "n": n}
    correlations = np.asarray([float(a @ np.roll(b, shift)) for shift in range(n)])
    ranks = rankdata(correlations, method="average")
    factual_rank = float(ranks[0] / n)
    best_shift = int(np.argmax(correlations))
    reverse = b[::-1]
    reverse_correlations = np.asarray([float(a @ np.roll(reverse, shift)) for shift in range(n)])
    reverse_factual_rank = float(rankdata(reverse_correlations, method="average")[0] / n)
    return {
        "available": True,
        "n": n,
        "factual_correlation": float(correlations[0]),
        "factual_orbit_rank": factual_rank,
        "best_shift_seconds": best_shift if best_shift <= n // 2 else best_shift - n,
        "best_correlation": float(correlations[best_shift]),
        "reverse_factual_orbit_rank": reverse_factual_rank,
        "unique_a": int(len(np.unique(a))),
        "unique_b": int(len(np.unique(b))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-method")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in rows(args.cohort):
            if args.cohort_method and row.get("method") != args.cohort_method:
                continue
            dataset, video_id = row["dataset"], row["video_id"]
            visual_path = args.visual_root / dataset / f"{video_id}.npy"
            visual = np.asarray(np.load(visual_path), dtype=np.float64) if visual_path.exists() else None
            audio = load_feature(args.feature_root, "vggish_1s", dataset, video_id)
            text = load_feature(args.feature_root, "bert_sentence_1fps", dataset, video_id)
            signals = {}
            if visual is not None and visual.ndim == 2 and len(visual) >= 3:
                signals["V"] = novelty(visual)
            if audio is not None:
                signals["A"] = novelty(audio)
            if text is not None:
                signals["T"] = novelty(text)
            pairs = {}
            for left, right in (("V", "T"), ("T", "A"), ("A", "V")):
                pairs[f"{left}{right}"] = orbit(signals[left], signals[right]) if left in signals and right in signals else {"available": False, "n": 0}
            available = [x["factual_orbit_rank"] for x in pairs.values() if x.get("available")]
            output = {
                "schema_version": 1,
                "method": "triarc_reciprocal_orbit_v1",
                "dataset": dataset,
                "video_id": video_id,
                "duration": float(row["duration"]),
                "pairs": pairs,
                "triadic_bottleneck": float(min(available)) if len(available) == 3 else None,
                "available_modalities": sorted(signals),
                "gt_access": False,
                "dataset_parameters": 0,
                "tie_rule": "average_midrank",
                "null": "all_nonzero_circular_shifts",
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
