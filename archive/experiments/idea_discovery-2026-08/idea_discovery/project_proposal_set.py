#!/usr/bin/env python3
"""Emit the top-K diverse Vid-Group proposals as a set-valued localization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl, intervals_to_curve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    count = 0
    for row in map(json.loads, args.proposals.open()):
        duration = float(row["duration"])
        values = sorted(row["proposals"], key=lambda value: int(value["rank"]))[:args.top_k]
        intervals = [Interval(float(v["start"]), float(v["end"]),
                              1.0 / (1.0 + __import__("math").exp(-float(v["logit"]))))
                     for v in values]
        append_jsonl(args.out, Prediction(
            f"proposal_set_top{args.top_k}", row["dataset"], row["video_id"], duration,
            score_curve=intervals_to_curve(intervals, duration), intervals=intervals, calls=0,
            modality_evidence={"top_k": args.top_k},
            raw={"gt_access": False, "proposal_source": "nms05_diverse_topk"}))
        count += 1
    print(json.dumps({"n": count, "top_k": args.top_k}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
