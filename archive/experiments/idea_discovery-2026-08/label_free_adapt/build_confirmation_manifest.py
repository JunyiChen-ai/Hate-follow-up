#!/usr/bin/env python3
"""Remove mechanism-selection pilot IDs from the untouched confirmation set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", required=True)
    parser.add_argument("--pilot", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    pilot = {(row["dataset"], row["video_id"])
             for row in map(json.loads, Path(args.pilot).read_text().splitlines())}
    rows = [row for row in map(json.loads, Path(args.all).read_text().splitlines())
            if (row["dataset"], row["video_id"]) not in pilot]
    destination = Path(args.out); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(json.dumps({"confirmation_videos": len(rows), "excluded_pilot": len(pilot)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
