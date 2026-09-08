#!/usr/bin/env python3
"""Build a deterministic multiscale proposal bank from frozen T3AL curves."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def ecdf(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    order = np.argsort(x, kind="stable")
    rank = np.empty(len(x), float); rank[order] = np.arange(len(x), dtype=float)
    return (rank + 0.5) / len(x)


def proposals(curve, duration, limit):
    x = ecdf(curve); n = len(x); pool = set()
    peaks = np.argsort(x)[::-1][:max(16, limit * 2)]
    for center in peaks:
        for frac in (.05, .10, .20, .35, .50):
            width = max(2, int(round(n * frac)))
            lo = max(0, center - width // 2); hi = min(n, lo + width); lo = max(0, hi - width)
            pool.add((lo, hi))
    scored = []
    for lo, hi in pool:
        score = float(x[lo:hi].mean() - .08 * (hi - lo) / n)
        scored.append((score, lo, hi))
    return [{"start": lo / n * duration, "end": hi / n * duration,
             "logit": score, "rank": rank + 1}
            for rank, (score, lo, hi) in enumerate(sorted(scored, reverse=True)[:limit])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--curves", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-m", type=int, default=32)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    rows = [json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    with args.out.open("x", encoding="utf-8") as out:
        for row in rows:
            path = args.curves / row["dataset"] / f"{row['video_id']}.npy"
            try:
                bank = proposals(np.load(path), float(row["duration"]), args.top_m)
                error = None
            except Exception as exc:
                bank = []; error = f"{type(exc).__name__}: {exc}"
            out.write(json.dumps({"dataset": row["dataset"], "video_id": row["video_id"],
                                  "duration": float(row["duration"]), "proposals": bank,
                                  "error": error}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
