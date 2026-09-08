#!/usr/bin/env python3
"""Project the no-necessity ablation from cached RCD four-arm queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.idea_discovery.run_rcd_witness import (coarse_intervals,
                                                     output_prediction, split,
                                                     tree_partition)
from scripts.label_free_adapt.schema import append_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    rows = [json.loads(x) for x in args.input.open() if x.strip()]
    for row in rows:
        evidence = row["modality_evidence"]
        windows = [tuple(map(float, x)) for x in evidence["evaluated_windows"]]
        logits = [tuple(map(float, x)) for x in evidence["evaluated_four_arm_log_odds"]]
        scores = dict(zip(windows, logits))
        coarse = coarse_intervals()
        selected = [parent for parent in coarse if all(child in scores for child in split(parent))]
        if len(selected) != 4 or len(scores) != 24:
            raise RuntimeError(f"invalid cached tree for {row['dataset']}/{row['video_id']}")
        partition = tree_partition(coarse, selected, scores, necessity=False)
        projected = output_prediction(
            "rcd_no_necessity_cached",
            {"dataset": row["dataset"], "video_id": row["video_id"],
             "duration": row["duration"]},
            partition, scores, False, 0, 96,
            evidence.get("invalid_asr_spans_rejected", []),
            evidence.get("ffmpeg_fallback_frames", 0), 0, windows)
        projected.raw["source_method"] = row["method"]
        projected.raw["cached_projection"] = True
        append_jsonl(args.out, projected)
    if sum(1 for _ in args.out.open()) != len(rows):
        raise RuntimeError("output cardinality mismatch")


if __name__ == "__main__":
    main()
