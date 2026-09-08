#!/usr/bin/env python3
"""Build a label-blind, geometry-enriched HYPER mechanism cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--proposals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-dataset", type=int, default=2)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    rows = {(r["dataset"], r["video_id"]): r for r in
            (json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip())}
    props = [json.loads(x) for x in args.proposals.read_text().splitlines() if x.strip()]
    ranked = {}
    for row in props:
        candidates = [(float(x["start"]), float(x["end"])) for x in row["proposals"][:4]]
        if len(candidates) < 4 or (row["dataset"], row["video_id"]) not in rows:
            continue
        divergence = 1.0 - min(tiou(a, b) for i, a in enumerate(candidates)
                               for b in candidates[i + 1:])
        digest = hashlib.sha256(
            f"hyper-pilot-v1\0{row['dataset']}\0{row['video_id']}".encode()).hexdigest()
        ranked.setdefault(row["dataset"], []).append((divergence, digest, rows[(row["dataset"], row["video_id"])]))
    selected = []
    for dataset in sorted(ranked):
        values = sorted(ranked[dataset], key=lambda x: (-x[0], x[1]))
        # Geometry is label-free; choose from the most-disagreeing quartile by hash.
        pool = sorted(values[:max(args.per_dataset, len(values) // 4)], key=lambda x: x[1])
        selected.extend(row for _, _, row in pool[:args.per_dataset])
    with args.out.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"n": len(selected), "by_dataset": {d: sum(r["dataset"] == d for r in selected)
                                                          for d in sorted(ranked)}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
