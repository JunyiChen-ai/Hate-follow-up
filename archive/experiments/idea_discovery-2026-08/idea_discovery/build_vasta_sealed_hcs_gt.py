#!/usr/bin/env python3
"""Build the prediction-frozen HateClipSeg p11 train+val 4-FPS evaluation GT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.label_free_adapt.build_gt_4fps import rasterize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    gold = json.loads(args.gold.read_text())
    video_ids, durations, labels, spans_out = [], [], [], []
    for row in manifest:
        video_id, duration = str(row["video_id"]), float(row["duration"])
        entry = gold[video_id]
        spans = [[float(start), float(end)] for start, end, dimensions in entry["segments"]
                 if any(dimensions[1:])]
        y4, clean_spans = rasterize(spans, duration)
        video_ids.append(video_id); durations.append(duration)
        labels.append(y4); spans_out.append(clean_spans)
    obj = lambda values: np.asarray(values + [None], dtype=object)[:-1]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, video_ids=np.asarray(video_ids),
                        split=np.asarray(["test"] * len(video_ids)),
                        duration=np.asarray(durations), y4=obj(labels), spans=obj(spans_out),
                        n_spans=np.asarray([len(value) for value in spans_out], dtype=np.int16))
    print(json.dumps({"videos": len(video_ids), "frames": sum(map(len, labels)),
                      "positive_frames": sum(int(value.sum()) for value in labels),
                      "videos_with_spans": sum(bool(len(value)) for value in spans_out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
