#!/usr/bin/env python3
"""Deterministic interval readouts for saved role-binding fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.run_melt import dense_bins
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def close_and_filter(active: np.ndarray, max_gap: int, min_len: int) -> np.ndarray:
    active = np.asarray(active, bool).copy()
    zeros = np.flatnonzero(np.diff(np.r_[False, ~active, False])).reshape(-1, 2)
    for a, b in zeros:
        if a > 0 and b < len(active) and b - a <= max_gap:
            active[a:b] = True
    ones = np.flatnonzero(np.diff(np.r_[False, active, False])).reshape(-1, 2)
    for a, b in ones:
        if b - a < min_len:
            active[a:b] = False
    return active


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-gap", type=int, required=True)
    ap.add_argument("--min-len", type=int, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    for line in args.input.open(encoding="utf-8"):
        row = json.loads(line)
        method = f"role_binding_g{args.max_gap}_l{args.min_len}"
        if row.get("error"):
            append_jsonl(args.out, Prediction(method, row["dataset"], row["video_id"],
                                              row["duration"], error=row["error"]))
            continue
        scores = np.asarray(row["modality_evidence"]["binding_scores"], float)
        margin = np.log(np.clip(scores[:, 0], 1e-8, 1)) - np.log(
            np.clip(scores[:, 1:].max(1), 1e-8, 1))
        active = close_and_filter(margin > 0, args.max_gap, args.min_len)
        bounds = np.flatnonzero(np.diff(np.r_[False, active, False])).reshape(-1, 2)
        duration = float(row["duration"])
        intervals = [Interval(a / 16 * duration, b / 16 * duration,
                              float(scores[a:b, 0].mean())) for a, b in bounds]
        dense = dense_bins(scores[:, 0], len(row["score_curve"]))
        append_jsonl(args.out, Prediction(
            method, row["dataset"], row["video_id"], duration,
            score_curve=dense.tolist(), intervals=intervals, calls=0,
            modality_evidence={"max_interior_gap_bins": args.max_gap,
                               "minimum_component_bins": args.min_len,
                               "paths": [[int(a), int(b)] for a, b in bounds]},
            raw={"source": str(args.input)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
