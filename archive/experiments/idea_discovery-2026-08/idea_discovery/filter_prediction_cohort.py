#!/usr/bin/env python3
"""Mechanically filter prediction JSONL to a sanitized cohort manifest."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method")
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    allowed = {(r["dataset"], r["video_id"])
               for r in (json.loads(x) for x in args.cohort.open(encoding="utf-8"))}
    count = 0
    with args.out.open("x", encoding="utf-8") as out:
        for line in args.input.open(encoding="utf-8"):
            row = json.loads(line)
            if ((row.get("dataset"), row.get("video_id")) in allowed and
                    (args.method is None or row.get("method") == args.method)):
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    print(json.dumps({"allowed": len(allowed), "written": count}))


if __name__ == "__main__":
    main()
