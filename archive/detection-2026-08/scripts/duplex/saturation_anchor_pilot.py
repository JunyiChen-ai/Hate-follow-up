#!/usr/bin/env python3
"""Held-out test of the saturation-anchor hypothesis (CPU only)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


NEG_EDGE = -13.0
POS_EDGE = 13.0
NEG_ANCHOR = -18.1
POS_ANCHOR = 15.0
TOL = 2.5
MIN_N = 5
N_BOOT = 2000
SEED = 20260808
PROBES = ("stance_v1", "stance_para", "effort_ctrl")


def bootstrap_mean_ci(x: np.ndarray, seed: int) -> list[float] | None:
    if len(x) < MIN_N:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        means[i] = np.mean(x[rng.integers(0, len(x), len(x))])
    return [float(v) for v in np.quantile(means, [0.025, 0.975])]


def band_summary(z: np.ndarray, mask: np.ndarray, seed: int) -> dict:
    x = z[mask]
    return {
        "n": int(len(x)),
        "occupancy": float(len(x) / len(z)),
        "identifiable": bool(len(x) >= MIN_N),
        "mean": float(np.mean(x)) if len(x) else None,
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else None,
        "mean_bootstrap_95": bootstrap_mean_ci(x, seed),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("results/stance_gate/probe_scores.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("results/saturation_anchor_pilot/results.json"))
    args = ap.parse_args()

    by_cell: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    with args.scores.open() as handle:
        for line in handle:
            r = json.loads(line)
            if r.get("probe") not in PROBES or not isinstance(r.get("z"), (int, float)):
                continue
            by_cell[(r["arm"], r["probe"])][r["video_id"]] = float(r["z"])

    arms = sorted({arm for arm, _ in by_cell})
    cells = {}
    neg_checks, pos_checks = [], []
    for cell_i, (arm, probe) in enumerate(sorted(by_cell)):
        vals = by_cell[(arm, probe)]
        z = np.asarray(list(vals.values()), dtype=float)
        neg = band_summary(z, z <= NEG_EDGE, SEED + 2 * cell_i)
        pos = band_summary(z, z >= POS_EDGE, SEED + 2 * cell_i + 1)
        if neg["identifiable"]:
            neg_checks.append(abs(neg["mean"] - NEG_ANCHOR) <= TOL)
        if pos["identifiable"]:
            pos_checks.append(abs(pos["mean"] - POS_ANCHOR) <= TOL)
        cells[f"{arm}|{probe}"] = {
            "n": len(z),
            "min": float(np.min(z)),
            "max": float(np.max(z)),
            "negative": neg,
            "positive": pos,
            "interior_occupancy": float(np.mean((z > NEG_EDGE) & (z < POS_EDGE))),
        }

    paired = {}
    occupancy_dissociation = False
    perturbation_checks = []
    for arm in arms:
        maps = [by_cell.get((arm, p), {}) for p in PROBES]
        common = sorted(set.intersection(*(set(m) for m in maps)))
        mat = np.asarray([[m[v] for m in maps] for v in common], dtype=float)
        ranges = np.ptp(mat, axis=1)
        stable_neg = np.all(mat <= NEG_EDGE, axis=1)
        stable_pos = np.all(mat >= POS_EDGE, axis=1)
        stable_extreme = stable_neg | stable_pos
        stable_interior = np.all((mat > NEG_EDGE) & (mat < POS_EDGE), axis=1)
        ext = ranges[stable_extreme]
        interior = ranges[stable_interior]
        evaluable = len(ext) >= MIN_N and len(interior) >= MIN_N
        ratio = float(np.median(ext) / np.median(interior)) if evaluable and np.median(interior) > 0 else None
        passed = bool(evaluable and ratio is not None and ratio <= 0.75)
        perturbation_checks.append(passed)

        extreme_occ = []
        locations_ok = True
        for p in PROBES:
            c = cells[f"{arm}|{p}"]
            extreme_occ.append(c["negative"]["occupancy"] + c["positive"]["occupancy"])
            for side, anchor in (("negative", NEG_ANCHOR), ("positive", POS_ANCHOR)):
                b = c[side]
                if b["identifiable"] and abs(b["mean"] - anchor) > TOL:
                    locations_ok = False
        positive_occ = [x for x in extreme_occ if x > 0]
        occ_ratio = max(positive_occ) / min(positive_occ) if positive_occ else None
        this_dissociation = bool(occ_ratio is not None and occ_ratio >= 1.25 and locations_ok)
        occupancy_dissociation |= this_dissociation
        paired[arm] = {
            "n_common": len(common),
            "stable_extreme_n": int(np.sum(stable_extreme)),
            "stable_interior_n": int(np.sum(stable_interior)),
            "median_range_stable_extreme": float(np.median(ext)) if len(ext) else None,
            "median_range_stable_interior": float(np.median(interior)) if len(interior) else None,
            "extreme_to_interior_range_ratio": ratio,
            "perturbation_clause_evaluable": evaluable,
            "perturbation_clause_pass": passed,
            "extreme_occupancy_by_probe": dict(zip(PROBES, extreme_occ)),
            "extreme_occupancy_max_over_min": occ_ratio,
            "location_occupancy_dissociation": this_dissociation,
        }

    clauses = {
        "negative_anchor": {"n_identifiable": len(neg_checks), "pass": len(neg_checks) >= 4 and all(neg_checks)},
        "positive_anchor": {"n_identifiable": len(pos_checks), "pass": len(pos_checks) >= 4 and all(pos_checks)},
        "location_occupancy_dissociation": {"pass": occupancy_dissociation},
        "extremes_more_stable_than_interior": {"pass": bool(perturbation_checks) and all(perturbation_checks)},
    }
    verdict = "PASS" if all(x["pass"] for x in clauses.values()) else "FAIL"
    out = {
        "pilot": "saturation_anchor_mechanism",
        "held_out_from_e7": True,
        "frozen": {"anchors": [NEG_ANCHOR, POS_ANCHOR], "edges": [NEG_EDGE, POS_EDGE], "tolerance": TOL,
                   "min_band_n": MIN_N, "n_boot": N_BOOT, "seed": SEED, "probes": list(PROBES)},
        "cells": cells,
        "paired_stability": paired,
        "clauses": clauses,
        "verdict": verdict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
