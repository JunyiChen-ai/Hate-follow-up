#!/usr/bin/env python3
"""Summarize append-only adapter outputs without opening ground truth."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    latest = {}
    for filename in args.predictions:
        with open(filename, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                latest[(row.get("method"), row.get("dataset"), row["video_id"])] = row
    groups = defaultdict(list)
    for (method, dataset, _), row in latest.items():
        groups[(method, dataset)].append(row)
    summary = []
    for (method, dataset), rows in sorted(groups.items()):
        valid = [row for row in rows if not row.get("error")]
        nonempty = [row for row in valid if row.get("intervals")]
        curves = [row.get("score_curve", []) for row in valid]
        summary.append({
            "method": method,
            "dataset": dataset,
            "n_records": len(rows),
            "n_valid": len(valid),
            "coverage": len(valid) / len(rows) if rows else 0.0,
            "n_nonempty": len(nonempty),
            "nonempty_rate": len(nonempty) / len(valid) if valid else 0.0,
            "n_nonconstant_curves": sum(
                bool(curve) and min(curve) < max(curve) for curve in curves),
            "errors": sorted({row.get("error") for row in rows if row.get("error")}),
        })
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
