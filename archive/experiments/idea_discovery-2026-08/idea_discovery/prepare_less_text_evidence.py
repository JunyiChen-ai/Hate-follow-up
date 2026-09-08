#!/usr/bin/env python3
"""Create the label-blind text-evidence schema consumed by LESS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.idea_discovery.project_pos_less import ASR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    count = 0
    with args.out.open("w") as handle:
        for dataset, path in ASR.items():
            for row in map(json.loads, path.open()):
                span = row["span"]
                value = row.get("z_masked", row.get("z_isolated"))
                if value is None:
                    continue
                output = {
                    "dataset": dataset,
                    "video_id": str(row["video_id"]),
                    "start": float(span[0]),
                    "end": float(span[1]),
                    "log_odds": float(value),
                    "source": "masked_or_isolated_chunk_logit",
                }
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")
                count += 1
    print(json.dumps({"rows": count, "datasets": sorted(ASR)}))


if __name__ == "__main__":
    main()
