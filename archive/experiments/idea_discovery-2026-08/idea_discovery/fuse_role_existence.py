#!/usr/bin/env python3
"""Label-blind dual-axis decoding of hostility existence and role identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.run_melt import dense_bins
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def components(active):
    return np.flatnonzero(np.diff(np.r_[False, np.asarray(active, bool), False])).reshape(-1, 2)


def dual_axis_active(role_scores, generic_scores, mode):
    role = role_scores[:, 0] > role_scores[:, 1:].max(1)
    generic = generic_scores[:, 0] > generic_scores[:, 1:].max(1)
    if mode == "intersection":
        return role & generic
    if mode == "union":
        return role | generic
    if mode == "generic_anchored":
        out = np.zeros_like(role)
        anchors = set(np.flatnonzero(role).tolist())
        for a, b in components(generic):
            if any(a <= x < b for x in anchors):
                out[a:b] = True
        return out
    if mode == "role_gapfill":
        out = role.copy()
        for a, b in components(generic):
            hits = np.flatnonzero(role[a:b]) + a
            if len(hits) >= 2:
                out[hits[0]:hits[-1] + 1] = True
        return out
    raise ValueError(mode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", type=Path, required=True)
    ap.add_argument("--generic", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("intersection", "union", "generic_anchored",
                                       "role_gapfill"), required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    def load(path):
        return {(r["dataset"], r["video_id"]): r
                for r in (json.loads(x) for x in path.open(encoding="utf-8"))}
    role, generic = load(args.role), load(args.generic)
    if set(role) != set(generic):
        raise RuntimeError("role/generic cohorts differ")
    for key in sorted(role):
        rr, gg = role[key], generic[key]
        method = f"dual_axis_{args.mode}"
        error = rr.get("error") or gg.get("error")
        if error:
            append_jsonl(args.out, Prediction(method, key[0], key[1], rr["duration"],
                                              error=error))
            continue
        rs = np.asarray(rr["modality_evidence"]["binding_scores"], float)
        gs = np.asarray(gg["modality_evidence"]["binding_scores"], float)
        active = dual_axis_active(rs, gs, args.mode)
        bounds = components(active)
        duration = float(rr["duration"])
        intervals = [Interval(a / 16 * duration, b / 16 * duration,
                              float(np.sqrt(rs[a:b, 0].mean() * gs[a:b, 0].mean())))
                     for a, b in bounds]
        # Dense ranking remains the exact-relation axis.  Generic hostility is
        # used only as a continuity witness and cannot dominate frame scores.
        curve = dense_bins(rs[:, 0], len(rr["score_curve"]))
        append_jsonl(args.out, Prediction(
            method, key[0], key[1], duration, score_curve=curve.tolist(),
            intervals=intervals, calls=0,
            modality_evidence={"mode": args.mode,
                               "paths": [[int(a), int(b)] for a, b in bounds]},
            raw={"role": str(args.role), "generic": str(args.generic)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
