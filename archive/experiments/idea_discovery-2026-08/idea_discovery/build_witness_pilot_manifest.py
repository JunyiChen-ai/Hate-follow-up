#!/usr/bin/env python3
"""Build a label-blind, hash-frozen WITNESS pilot cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-dataset", type=int, default=16)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    excluded = {(r["dataset"], r["video_id"]) for r in read(args.exclude)}
    grouped = defaultdict(list)
    for row in read(args.source):
        key = (row["dataset"], row["video_id"])
        if key in excluded:
            continue
        digest = hashlib.sha256(f"WITNESS-PILOT-V1\0{key[0]}\0{key[1]}".encode()).hexdigest()
        grouped[key[0]].append((digest, row))
    selected = []
    for dataset in sorted(grouped):
        selected.extend(row for _, row in sorted(grouped[dataset])[:args.per_dataset])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    counts = {dataset: sum(r["dataset"] == dataset for r in selected)
              for dataset in sorted(grouped)}
    print(json.dumps({"rows": len(selected), "counts": counts,
                      "excluded_overlap": 0, "label_fields_used": []}, sort_keys=True))


if __name__ == "__main__":
    main()
