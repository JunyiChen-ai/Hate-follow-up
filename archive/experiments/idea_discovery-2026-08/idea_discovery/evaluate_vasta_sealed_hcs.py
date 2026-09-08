#!/usr/bin/env python3
"""Evaluate all frozen methods on one sealed HateClipSeg cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.label_free_adapt.evaluate import interval_f1, pooled, within_video_macro


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    latest = {}
    for row in map(json.loads, args.predictions.open()):
        latest[(row["method"], row["video_id"])] = row
    gt = np.load(args.gt, allow_pickle=True)
    y = {str(video_id): np.asarray(gt["y4"][i], dtype=np.int8)
         for i, video_id in enumerate(gt["video_ids"])}
    rows = []
    for method in sorted({key[0] for key in latest}):
        predictions = {video_id: row for (name, video_id), row in latest.items()
                       if name == method and not row.get("error")}
        if set(predictions) != set(y):
            raise RuntimeError(f"{method}: prediction/GT ID mismatch")
        scores = {video_id: np.asarray(row["score_curve"], float)
                  for video_id, row in predictions.items()}
        rows.append({"method": method, **pooled(y, scores), **within_video_macro(y, scores),
                     **interval_f1(y, predictions), "n_videos": len(predictions),
                     "n_nonempty": sum(bool(row.get("intervals")) for row in predictions.values())})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
