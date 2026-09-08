#!/usr/bin/env python3
"""Orbit-calibrated selective endpoint edits, with no labels or fixed top-K.

For each endpoint, compare the observed directed local transition against the
same statistic after every non-trivial circular rotation of the dense temporal
field.  The empirical tail probability decides whether the edit is warranted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path, method: str | None = None):
    out = {}
    for line in path.open():
        row = json.loads(line)
        if method is None or row.get("method") == method:
            out[(row["dataset"], row["video_id"])] = row
    return out


def interval(row):
    return tuple(map(float, row["intervals"][0][:2])) if row and row.get("intervals") else None


def logit(values):
    x = np.clip(np.asarray(values, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def sample(field, times, duration):
    ix = np.clip(np.round(np.asarray(times) / max(duration, 1e-9) * (len(field) - 1)).astype(int),
                 0, len(field) - 1)
    values = field[ix]
    scale = np.median(np.abs(field - np.median(field))) + 1e-6
    return np.tanh(values / (2 * scale))


def directed_stat(values, side):
    delta = np.diff(values)
    return float(np.max(delta) if side == 0 else np.max(-delta))


def orbit_pvalue(field, times, duration, side):
    observed = directed_stat(sample(field, times, duration), side)
    # Unique rotations at approximately one-second resolution avoid treating
    # repeated 4-fps samples as independent null draws.
    stride = max(1, int(round(len(field) / max(duration, 1.0))))
    shifts = range(stride, len(field), stride)
    null = np.asarray([directed_stat(sample(np.roll(field, shift), times, duration), side)
                       for shift in shifts], float)
    p = (1.0 + float(np.sum(null >= observed - 1e-12))) / (1.0 + len(null))
    return p, observed, int(len(null))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--base-method", required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--tight-method", default="fact_less_t3al_dualgeo_shorter_v5")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    zoom = load(args.zoom, "ambi_zoom_v1")
    fused = load(args.zoom, "ambi_fused_transition_control_v1")
    base = load(args.base, args.base_method)
    tight = load(args.bank, args.tight_method)
    alphas = (0.05, 0.10, 0.20)
    counts = {a: 0 for a in alphas}
    with args.out.open("w") as handle:
        for key in sorted(zoom):
            bi, fi, ti = interval(base.get(key)), interval(fused.get(key)), interval(tight.get(key))
            stats = [None, None]
            if bi is not None and fi is not None:
                duration = float(zoom[key]["duration"])
                field = logit(zoom[key]["score_curve"]); field -= field.mean()
                audit = zoom[key].get("raw", {}).get("boundary_audit", {})
                for side, name in enumerate(("left", "right")):
                    times = audit.get(name, {}).get("times", [])
                    preserves = ti is None or (fi[side] <= ti[0] if side == 0 else fi[side] >= ti[1])
                    if len(times) == 8 and abs(fi[side] - bi[side]) > 1e-8 and preserves:
                        p, strength, nnull = orbit_pvalue(field, times, duration, side)
                        stats[side] = {"p": p, "strength": strength, "n_null": nnull}
            for alpha in alphas:
                source = base.get(key) or zoom[key]
                output = dict(source)
                output["method"] = f"orbit_endpoint_alpha_{str(alpha).replace('.', 'p')}_v1"
                accepted = [False, False]
                result = bi
                if bi is not None and fi is not None:
                    values = list(bi)
                    for side in (0, 1):
                        if stats[side] is not None and stats[side]["p"] <= alpha:
                            values[side] = fi[side]; accepted[side] = True
                    if values[0] < values[1]:
                        result = tuple(values); counts[alpha] += sum(accepted)
                    else:
                        accepted = [False, False]
                output["intervals"] = [] if result is None else [[result[0], result[1], 1.0]]
                output["raw"] = {**output.get("raw", {}), "gt_access": False,
                                 "selector": "circular_orbit_empirical_tail",
                                 "alpha": alpha, "endpoint_statistics": stats,
                                 "accepted_sides": accepted}
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps({"n_videos": len(zoom), "accepted_endpoint_actions": counts}, indent=2))


if __name__ == "__main__":
    main()
