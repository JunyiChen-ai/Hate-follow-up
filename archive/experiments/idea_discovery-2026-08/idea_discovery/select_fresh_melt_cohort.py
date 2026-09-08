#!/usr/bin/env python3
"""Select a deterministic label-free cohort disjoint from the tuning pilot."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, required=True)
    ap.add_argument("--curves", type=Path, required=True)
    ap.add_argument("--per-dataset", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    excluded = {(r["dataset"], r["video_id"]) for r in read(args.exclude)}
    groups = {}
    allowed = {"dataset", "duration", "transcript", "video_id", "video_path"}
    for row in read(args.manifest):
        if set(row) - allowed:
            raise RuntimeError(f"unsafe manifest fields: {set(row)-allowed}")
        key = row["dataset"], row["video_id"]
        if key in excluded or not (args.curves / key[0] / f"{key[1]}.npy").exists():
            continue
        digest = hashlib.sha256(f"melt-fresh-v1::{key[0]}::{key[1]}".encode()).hexdigest()
        groups.setdefault(key[0], []).append((digest, row))
    selected = []
    for dataset in sorted(groups):
        rows = [row for _, row in sorted(groups[dataset])[:args.per_dataset]]
        if len(rows) != args.per_dataset:
            raise RuntimeError(f"{dataset}: only {len(rows)} eligible rows")
        selected.extend(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    print(json.dumps({"out": str(args.out), "counts": {
        d: sum(r["dataset"] == d for r in selected) for d in sorted(groups)}}))


if __name__ == "__main__":
    main()
