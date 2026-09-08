#!/usr/bin/env python3
"""Apply the declared half-open 4-fps timeline contract without reading GT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-method", default="final_candidate_fixed4fps_v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    count = 0
    cropped = 0
    with args.out.open("w") as handle:
        for row in map(json.loads, args.input.open()):
            if row.get("method") != args.method:
                continue
            expected = max(1, int(float(row["duration"]) * 4))
            curve = row.get("score_curve", [])
            if len(curve) not in (expected, expected + 1):
                raise ValueError(
                    f"unexpected 4-fps length for {(row['dataset'], row['video_id'])}: "
                    f"duration={row['duration']} expected={expected} actual={len(curve)}"
                )
            cropped += len(curve) - expected
            out = dict(row)
            out["method"] = args.output_method
            out["score_curve"] = curve[:expected]
            out["native_rate"] = 4.0
            contract_end = expected / 4.0
            clipped_intervals = []
            for interval in out.get("intervals", []):
                start, end, *rest = interval
                start = max(0.0, min(float(start), contract_end))
                end = max(0.0, min(float(end), contract_end))
                if end > start:
                    clipped_intervals.append([start, end, *rest])
            out["intervals"] = clipped_intervals
            out["raw"] = {
                **out.get("raw", {}),
                "grid_contract": "half-open [0,duration) sampled at 4 fps; n=floor(4*duration)",
                "grid_contract_gt_access": False,
                "removed_terminal_samples": len(curve) - expected,
                "intervals_clipped_to_grid_contract": True,
            }
            handle.write(json.dumps(out, separators=(",", ":")) + "\n")
            count += 1
    print(json.dumps({"rows": count, "removed_terminal_samples": cropped}))


if __name__ == "__main__":
    main()
