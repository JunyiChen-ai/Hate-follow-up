#!/usr/bin/env python3
"""Label-free algebraic controls for matched TIDE U32 modality fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.run_tide_u32 import bins_to_curve, positive_intervals
from scripts.label_free_adapt.schema import Prediction, append_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    groups = {}
    for line in args.input.open():
        row = json.loads(line)
        groups.setdefault((row["dataset"], row["video_id"]), {})[row["method"]] = row
    for (dataset, video_id), rows in groups.items():
        required = {"u32_joint", "u32_visual", "u32_text"}
        if set(rows) < required:
            raise RuntimeError(f"missing matched arms for {(dataset, video_id)}")
        duration = float(rows["u32_joint"]["duration"])
        fields = np.stack([
            rows[name]["modality_evidence"]["corrected_log_odds"]
            for name in ("u32_joint", "u32_visual", "u32_text")
        ])
        ordered = np.sort(fields, axis=0)
        variants = {
            "tide_mean3": fields.mean(0),
            "tide_median3": np.median(fields, axis=0),
            "tide_top2mean": ordered[1:].mean(0),
            "tide_max3": fields.max(0),
            # Joint gets only a bounded bonus when it exceeds both unimodal views.
            "tide_synergy_bonus": fields[0] + np.clip(
                fields[0] - np.maximum(fields[1], fields[2]), 0, 2),
        }
        for method, z in variants.items():
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            append_jsonl(args.out, Prediction(
                method, dataset, video_id, duration,
                score_curve=bins_to_curve(p, duration).tolist(),
                intervals=positive_intervals(z, duration), calls=0,
                modality_evidence={"source_arms": ["joint", "visual", "text"],
                                   "fused_log_odds": z.tolist()},
                raw={"label_free_algebraic_control": True, "gt_access": False}))


if __name__ == "__main__":
    main()
