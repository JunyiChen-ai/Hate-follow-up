#!/usr/bin/env python3
"""Merge append-only prediction JSONL files by latest method/dataset/video row."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    latest = {}
    for path in args.inputs:
        with path.open() as handle:
            for row in map(json.loads, handle):
                key = (row.get("method"), row.get("dataset"), str(row["video_id"]))
                latest[key] = row
    valid = [row for row in latest.values() if not row.get("error") and row.get("score_curve")]
    with args.out.open("w") as handle:
        for row in sorted(valid, key=lambda item: (item.get("method", ""),
                                                   item.get("dataset", ""),
                                                   str(item["video_id"]))):
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({"inputs": [str(path) for path in args.inputs],
                      "latest_rows": len(latest), "valid_rows": len(valid)}))


if __name__ == "__main__":
    main()
