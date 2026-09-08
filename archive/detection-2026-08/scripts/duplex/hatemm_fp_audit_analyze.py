#!/usr/bin/env python3
"""Summarize the preregistered HateMM false-positive taxonomy audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


CATEGORIES = ("G", "Q", "O", "P", "D", "I", "X")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - radius, center + radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coding", type=Path, default=Path("results/hatemm_fp_audit/coding.tsv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.coding.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    aliases = [row["alias"] for row in rows]
    if len(rows) != 70 or len(set(aliases)) != 70:
        raise ValueError(f"Expected 70 unique rows, found {len(rows)} rows/{len(set(aliases))} aliases")
    invalid = sorted({row["primary"] for row in rows} - set(CATEGORIES))
    if invalid:
        raise ValueError(f"Invalid categories: {invalid}")

    counts = Counter(row["primary"] for row in rows)
    intervals = {c: wilson(counts[c], len(rows)) for c in CATEGORIES}
    g_rate = counts["G"] / len(rows)
    if g_rate < 0.30:
        verdict = "REFUTED"
    elif intervals["G"][0] < 0.30:
        verdict = "WEAK"
    else:
        verdict = "SUPPORTED"

    result = {
        "n": len(rows),
        "counts": dict(counts),
        "rates": {c: counts[c] / len(rows) for c in CATEGORIES},
        "wilson_95": {c: list(intervals[c]) for c in CATEGORIES},
        "secondary": {
            "G_plus_O_count": counts["G"] + counts["O"],
            "G_plus_O_rate": (counts["G"] + counts["O"]) / len(rows),
            "D_plus_I_count": counts["D"] + counts["I"],
            "D_plus_I_rate": (counts["D"] + counts["I"]) / len(rows),
            "visual_reviewed_count": sum(row["visual_reviewed"] == "true" for row in rows),
        },
        "preregistered_rule": "SUPPORTED iff Wilson lower bound(G)>=0.30; WEAK iff G>=0.30; else REFUTED",
        "verdict": verdict,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
