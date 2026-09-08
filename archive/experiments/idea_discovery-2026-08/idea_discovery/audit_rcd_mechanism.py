#!/usr/bin/env python3
"""Label-blind RCD-Witness deletion-mechanism audit."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", default="rcd_witness")
    args = parser.parse_args()
    rows = [json.loads(x) for x in args.predictions.open() if x.strip()]
    rows = [x for x in rows if x.get("method") == args.method and not x.get("error")]
    if not rows:
        raise RuntimeError(f"no valid {args.method} rows")
    by_dataset = defaultdict(list)
    for row in rows:
        values = np.asarray(row["modality_evidence"]["evaluated_four_arm_log_odds"], float)
        if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
            raise RuntimeError(f"invalid four-arm values for {row['dataset']}/{row['video_id']}")
        by_dataset[row["dataset"]].append(values)

    def summarize(arrays):
        z = np.concatenate(arrays); factual = z[:, 0] > 0; deleted = z[:, 3] > 0
        factual_count = int(factual.sum()); witnesses = factual & ~deleted
        retained = factual & deleted
        visual_delta = z[:, 0] - z[:, 1]; text_delta = z[:, 0] - z[:, 2]
        types = Counter()
        for dv, dt in zip(visual_delta[witnesses], text_delta[witnesses]):
            types[("both" if dv > 0 and dt > 0 else "visual" if dv > 0 else
                   "text" if dt > 0 else "neither")] += 1
        return {
            "evaluated_windows": int(len(z)),
            "factual_positive": factual_count,
            "joint_deletion_sign_flips": int(witnesses.sum()),
            "flip_rate_among_factual_positive": (float(witnesses.sum() / factual_count)
                                                  if factual_count else None),
            "retain_rate_among_factual_positive": (float(retained.sum() / factual_count)
                                                    if factual_count else None),
            "median_visual_delta_on_witness": (float(np.median(visual_delta[witnesses]))
                                               if witnesses.any() else None),
            "median_text_delta_on_witness": (float(np.median(text_delta[witnesses]))
                                             if witnesses.any() else None),
            "witness_provenance_signs": dict(types),
        }
    all_arrays = [array for values in by_dataset.values() for array in values]
    output = {"method": args.method,
              "overall": summarize(all_arrays),
              "per_dataset": {dataset: summarize(values)
                              for dataset, values in sorted(by_dataset.items())}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
