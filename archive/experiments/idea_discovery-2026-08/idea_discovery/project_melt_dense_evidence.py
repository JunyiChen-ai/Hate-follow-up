#!/usr/bin/env python3
"""Project coarse relation-conditioned support onto a dense label-free curve.

The MLLM defines which coarse regions support one fixed event relation.  This
label-blind readout selects a dense evidence basin inside each region and maps
the result to the canonical 4 FPS evaluation grid.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.idea_discovery.run_counterfactual_evidence import ecdf
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def canonical_resample(values: np.ndarray, duration: float, fps: float = 4.0) -> np.ndarray:
    values = np.asarray(values, float)
    n = max(1, int(math.floor(duration * fps)))
    source_t = (np.arange(len(values)) + .5) / len(values) * duration
    target_t = (np.arange(n) + .5) / fps
    return np.interp(target_t, source_t, values, left=values[0], right=values[-1])


def peak_basin(curve: np.ndarray, lo: int, hi: int, threshold: float) -> tuple[int, int]:
    """Return the threshold basin containing the strongest point in [lo, hi)."""
    if hi <= lo:
        return lo, hi
    peak = lo + int(np.argmax(curve[lo:hi]))
    start, end = peak, peak + 1
    while start > lo and curve[start - 1] >= threshold:
        start -= 1
    while end < hi and curve[end] >= threshold:
        end += 1
    return start, end


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--t3al-curves", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quantile", type=float, required=True)
    ap.add_argument("--source-method", default="melt_adaptive")
    args = ap.parse_args()
    if not 0 <= args.quantile <= 1:
        raise ValueError("quantile must lie in [0,1]")
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    for line in args.input.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("method") != args.source_method:
            continue
        method = f"melt_dense_projection_q{args.quantile:g}"
        if row.get("error"):
            append_jsonl(args.out, Prediction(method, row["dataset"], row["video_id"],
                                              row["duration"], error=row["error"]))
            continue
        duration = float(row["duration"])
        raw = np.load(args.t3al_curves / row["dataset"] / f"{row['video_id']}.npy")
        dense = canonical_resample(ecdf(raw), duration)
        threshold = float(np.quantile(dense, args.quantile))
        refined = []
        for value in row.get("intervals", []):
            start, end = map(float, value[:2])
            lo = max(0, int(math.floor(start * 4)))
            hi = min(len(dense), int(math.ceil(end * 4)))
            a, b = peak_basin(dense, lo, hi, threshold)
            if b > a:
                refined.append(Interval(a / 4, min(duration, b / 4),
                                        float(dense[a:b].mean())))
        mask_curve = np.zeros_like(dense)
        for interval in refined:
            a, b = int(round(interval.start * 4)), int(round(interval.end * 4))
            mask_curve[a:b] = dense[a:b]
        append_jsonl(args.out, Prediction(
            method, row["dataset"], row["video_id"], duration,
            score_curve=mask_curve.tolist(), intervals=refined, calls=0,
            modality_evidence={"coarse_source": args.source_method,
                               "dense_source": "canonical_4fps_T3AL_ECDF",
                               "global_quantile": args.quantile},
            raw={"source": str(args.input), "t3al": str(args.t3al_curves)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
