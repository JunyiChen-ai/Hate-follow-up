#!/usr/bin/env python3
"""Expand proposal support only before the rank-1 interval reaches majority scale."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


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
    keys = sorted(set(rank1) & set(extent)); counts = {"rank1_saturated": 0, "expanded": 0}
    for key in keys:
        base = rank1[key]
        start = min(float(v[0]) for v in base["intervals"])
        end = max(float(v[1]) for v in base["intervals"])
        duration = float(base["duration"])
        saturated = (end - start) / duration >= 0.5
        source = base if saturated else extent[key]
        counts["rank1_saturated" if saturated else "expanded"] += 1
        intervals = [Interval(float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 1.0)
                     for v in source["intervals"]]
        append_jsonl(args.out, Prediction(
            "saturation_extent_v1", key[0], key[1], duration,
            score_curve=source["score_curve"], intervals=intervals, calls=0,
            modality_evidence={"rank1_fraction": (end - start) / duration,
                               "majority_saturated": saturated,
                               "source": "rank1" if saturated else "top8_extent"},
            raw={"gt_access": False, "threshold": 0.5,
                 "threshold_semantics": "rank1 already covers temporal majority"}))
    print(json.dumps({"n": len(keys), "counts": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
