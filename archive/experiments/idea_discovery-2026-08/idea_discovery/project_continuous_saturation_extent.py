#!/usr/bin/env python3
"""Continuously shrink support expansion as rank-1 temporal coverage saturates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl, intervals_to_curve


def load(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row for row in map(json.loads, path.open())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank1", type=Path, required=True)
    parser.add_argument("--extent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    rank1, extent = load(args.rank1), load(args.extent)
    keys = sorted(set(rank1) & set(extent)); factors = []
    for key in keys:
        base, broad = rank1[key], extent[key]
        r0 = min(float(v[0]) for v in base["intervals"])
        r1 = max(float(v[1]) for v in base["intervals"])
        e0 = min(float(v[0]) for v in broad["intervals"])
        e1 = max(float(v[1]) for v in broad["intervals"])
        duration = float(base["duration"])
        fraction = min(1.0, max(0.0, (r1 - r0) / duration))
        factor = 1.0 - fraction
        start = r0 - factor * (r0 - e0)
        end = r1 + factor * (e1 - r1)
        score = float(base["intervals"][0][2]) if len(base["intervals"][0]) > 2 else 1.0
        interval = Interval(max(0.0, start), min(duration, end), score)
        append_jsonl(args.out, Prediction(
            "continuous_saturation_extent_v1", key[0], key[1], duration,
            score_curve=intervals_to_curve([interval], duration), intervals=[interval], calls=0,
            modality_evidence={"rank1_fraction": fraction, "expansion_factor": factor,
                               "rank1": [r0, r1], "envelope": [e0, e1]},
            raw={"gt_access": False, "fitted_parameters": 0,
                 "rule": "linear_remaining_temporal_capacity"}))
        factors.append(factor)
    print(json.dumps({"n": len(keys), "mean_expansion_factor": sum(factors) / len(factors)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
