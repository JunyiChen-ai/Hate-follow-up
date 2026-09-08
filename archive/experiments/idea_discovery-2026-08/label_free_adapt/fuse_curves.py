#!/usr/bin/env python3
"""Label-free fusion of semantic-window and temporal-grounding curves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .schema import Prediction, append_jsonl, curve_to_intervals


def latest(path: str) -> dict:
    rows = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        rows[(row["dataset"], row["video_id"])] = row
    return rows


def robust_scale(values: np.ndarray) -> np.ndarray:
    """Within-video rank scaling; invariant to each model's score calibration."""
    if len(values) == 0 or np.all(values == values[0]):
        return np.zeros_like(values, dtype=float)
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return order.astype(float) / max(1, len(values) - 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", required=True)
    parser.add_argument("--temporal", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    semantic, temporal = latest(args.semantic), latest(args.temporal)
    for key in sorted(set(semantic) & set(temporal)):
        left, right = semantic[key], temporal[key]
        prediction = Prediction("Fusion_Binwise_TimeLens", key[0], key[1],
                                float(left["duration"]), seed=int(left["seed"]))
        if left.get("error") or right.get("error"):
            prediction.error = f"semantic={left.get('error')}; temporal={right.get('error')}"
            append_jsonl(Path(args.out), prediction); continue
        n = min(len(left["score_curve"]), len(right["score_curve"]))
        semantic_curve = robust_scale(np.asarray(left["score_curve"][:n], dtype=float))
        temporal_curve = robust_scale(np.asarray(right["score_curve"][:n], dtype=float))
        # Fixed symmetric fusion: neither modality/system can dominate by scale.
        prediction.score_curve = ((semantic_curve + temporal_curve) / 2).tolist()
        prediction.intervals = curve_to_intervals(
            prediction.score_curve, prediction.duration, threshold=.5)
        prediction.modality_evidence = {"semantic_weight": .5, "temporal_weight": .5,
                                        "calibration": "within_video_rank"}
        append_jsonl(Path(args.out), prediction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
