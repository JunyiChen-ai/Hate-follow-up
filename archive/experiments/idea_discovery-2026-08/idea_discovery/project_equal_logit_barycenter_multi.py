#!/usr/bin/env python3
"""Unweighted logit barycenter of two or more frozen score fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.source) != len(args.method) or len(args.source) < 2:
        raise ValueError("provide the same number (at least two) of --source and --method")
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    fields = [load(path, method) for path, method in zip(args.source, args.method)]
    keys = sorted(set.intersection(*(set(field) for field in fields)))
    weight = 1.0 / len(fields)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for key in keys:
            rows = [field[key] for field in fields]
            probabilities = [
                np.clip(np.asarray(row["score_curve"], dtype=float), 1e-6, 1 - 1e-6)
                for row in rows
            ]
            lengths = {len(values) for values in probabilities}
            if len(lengths) != 1:
                raise ValueError(f"unaligned score lengths for {key}: {sorted(lengths)}")
            logits = [np.log(values / (1 - values)) for values in probabilities]
            barycenter = np.mean(np.stack(logits), axis=0)
            target_mean = float(np.mean([values.mean() for values in probabilities]))
            lo, hi = -30.0, 30.0
            for _ in range(80):
                mid = (lo + hi) / 2
                if sigmoid(barycenter + mid).mean() < target_mean:
                    lo = mid
                else:
                    hi = mid

            out = dict(rows[0])
            out["method"] = f"equal_logit_barycenter_{len(fields)}way_v1"
            out["score_curve"] = sigmoid(barycenter + (lo + hi) / 2).tolist()
            out["raw"] = {
                **out.get("raw", {}),
                "gt_access": False,
                "module": "symmetric_equal_logit_barycenter",
                "sources": args.method,
                "source_weights": [weight] * len(fields),
                "learned_weights": 0,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
                "intervals_preserved_from": args.method[0],
            }
            handle.write(json.dumps(out, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
