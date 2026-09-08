#!/usr/bin/env python3
"""Select a frozen geometry state using the PoS-LESS evidence contrast."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path):
    out = {}
    for row in map(json.loads, path.open()):
        out[(row["method"], row["dataset"], row["video_id"])] = row
    return out


def contrast(curve, interval, duration):
    if not interval:
        return -float("inf")
    n = len(curve)
    lo = max(0, min(n - 1, int(float(interval[0][0]) / duration * n)))
    hi = min(n, max(lo + 1, int(np.ceil(float(interval[0][1]) / duration * n))))
    inside = float(np.mean(curve[lo:hi]))
    outside_parts = [curve[:lo], curve[hi:]]
    nonempty = [x for x in outside_parts if len(x)]
    outside = np.concatenate(nonempty) if nonempty else np.asarray([], dtype=float)
    return inside - (float(np.mean(outside)) if len(outside) else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--posterior", type=Path, required=True)
    p.add_argument("--geometry-bank", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError(f"refusing existing output: {a.out}")
    post = load(a.posterior)
    bank = load(a.geometry_bank)
    rules = ("cca", "t3al", "shorter", "midpoint", "intersection", "union")
    counts = {}
    with a.out.open("w") as sink:
        for (method, dataset, video_id), row in sorted(post.items()):
            curve = np.asarray(row["score_curve"], float)
            candidates = []
            for rule in rules:
                key = (f"fact_less_t3al_dualgeo_{rule}_v5", dataset, video_id)
                candidate = bank.get(key)
                if candidate is not None:
                    candidates.append((contrast(curve, candidate["intervals"], float(row["duration"])), rule, candidate))
            # Deterministic tie order prefers the existing midpoint abstention.
            tie = {r: i for i, r in enumerate(("midpoint", "cca", "t3al", "shorter", "intersection", "union"))}
            _, rule, selected = max(candidates, key=lambda x: (x[0], -tie[x[1]]))
            counts[(method, rule)] = counts.get((method, rule), 0) + 1
            output = dict(row)
            output["method"] = method.replace("_v1", "_extent_v1")
            output["intervals"] = selected["intervals"]
            output["raw"] = {**row.get("raw", {}), "extent_decoder": "max_inside_outside_contrast",
                             "extent_candidate_rules": list(rules), "selected_rule": rule}
            sink.write(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps({f"{m}:{r}": n for (m, r), n in sorted(counts.items())}, indent=2))


if __name__ == "__main__":
    main()
