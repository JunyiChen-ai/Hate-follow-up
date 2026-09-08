#!/usr/bin/env python3
"""Convert the reconstructed T3AL main curves to the shared prediction schema."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--curve-dir", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True,
                        help="Prediction JSONL defining the exact target cohort")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    manifest = {(row["dataset"], row["video_id"]): row
                for row in map(json.loads, args.manifest.open())}
    cohort = {(row["dataset"], row["video_id"])
              for row in map(json.loads, args.cohort.open())}
    intervals = {}
    for dataset in sorted({key[0] for key in cohort}):
        path = args.curve_dir / f"{dataset}_intervals_main.json"
        intervals[dataset] = json.loads(path.read_text())
    missing = []
    for key in sorted(cohort):
        path = args.curve_dir / key[0] / f"{key[1]}.npz"
        if not path.exists():
            missing.append(list(key))
            continue
        archive = np.load(path)
        curve = np.asarray(archive["main"], dtype=float)
        duration = float(manifest[key]["duration"])
        event_intervals = [Interval(float(value[0]), min(duration, float(value[1])),
                                    float(value[2]) if len(value) > 2 else 1.0)
                           for value in intervals[key[0]].get(key[1], [])
                           if float(value[1]) > float(value[0])]
        append_jsonl(args.out, Prediction(
            "t3al_main_s20250819_reconstructed", key[0], key[1], duration,
            score_curve=curve.tolist(), intervals=event_intervals, calls=0,
            raw={"gt_access": False, "preset": "D_anet", "seed": 20250819,
                 "variant": "main", "source": str(args.curve_dir.resolve())}))
    print(json.dumps({"requested": len(cohort), "written": len(cohort) - len(missing),
                      "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
