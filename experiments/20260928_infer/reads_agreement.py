#!/usr/bin/env python3
"""Label-free agreement between read sets (README §11): per video, the Spearman correlation of the window score z
(max over branches) between a base read set and each other set, plus the verdict correlation and the share of
windows whose sign differs. No GT is read."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def load(d):
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, open(Path(d) / "predictions.jsonl")) if not r.get("error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--others", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = load(a.base); lines = []
    for o in a.others:
        other = load(o); rho, zv, flips, n = [], ([], []), 0, 0
        for k in sorted(set(base) & set(other)):
            wb, wo = base[k]["extra"]["windows"], other[k]["extra"]["windows"]
            if len(wb) != len(wo):
                continue
            zb, zo = np.array([w["z"] for w in wb]), np.array([w["z"] for w in wo])
            if len(zb) >= 3 and zb.std() > 0 and zo.std() > 0:
                rho.append(spearmanr(zb, zo).correlation)
            flips += int(((zb > 0) != (zo > 0)).sum()); n += len(zb)
            zv[0].append(base[k]["extra"]["z_video"]); zv[1].append(other[k]["extra"]["z_video"])
        lines.append(f"{Path(o).name} vs {Path(a.base).name}: videos {len(zv[0])}; within-video Spearman of window z: "
                     f"median {np.median(rho):.3f} (q25 {np.percentile(rho, 25):.3f}, q75 {np.percentile(rho, 75):.3f}); "
                     f"verdict Spearman {spearmanr(zv[0], zv[1]).correlation:.3f}; windows whose sign differs {flips / max(n, 1):.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
