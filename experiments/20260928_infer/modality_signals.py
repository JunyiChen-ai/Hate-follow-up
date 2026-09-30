#!/usr/bin/env python3
"""Label-free per-corpus signals about the two window branches (README §14): could a rule pick the branch weighting
per corpus without labels? For each corpus, from the isolated dual reads: (1) Spearman across videos of the verdict
z_video with the per-video mean visual read and with the mean speech read; (2) share of windows (with both
branches) where only the picture branch is positive, only the speech branch, both; (3) mean lag-1 within-video
autocorrelation of each branch; (4) median within-video Spearman between the branches. No GT is read."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="<run dir>:<dataset>[,<dataset>...]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = []
    for pair in a.pairs:
        d, dss = pair.split(":")
        recs = [r for r in map(json.loads, open(Path(d) / "predictions.jsonl")) if not r.get("error")]
        for ds in dss.split(","):
            zv, mv, ms, rho, ac = [], [], [], [], {"z_visual": [], "z_speech": []}
            n_po = n_so = n_both = n_all = 0
            for r in recs:
                if r["dataset"] != ds:
                    continue
                W = [w for w in r["extra"]["windows"] if "z_visual" in w and "z_speech" in w]
                if not W:
                    continue
                v = np.array([w["z_visual"] for w in W]); s = np.array([w["z_speech"] for w in W])
                zv.append(r["extra"]["z_video"]); mv.append(v.mean()); ms.append(s.mean())
                n_po += int(((v > 0) & (s <= 0)).sum()); n_so += int(((s > 0) & (v <= 0)).sum())
                n_both += int(((v > 0) & (s > 0)).sum()); n_all += len(W)
                if len(W) >= 4:
                    if v.std() > 0 and s.std() > 0:
                        rho.append(spearmanr(v, s).correlation)
                    for k, x in (("z_visual", v), ("z_speech", s)):
                        if x[:-1].std() > 0 and x[1:].std() > 0:
                            ac[k].append(np.corrcoef(x[:-1], x[1:])[0, 1])
            lines.append(f"== {ds} ({d}): {len(zv)} videos, {n_all} windows with both branches")
            lines.append(f"  Spearman(verdict, mean visual read) {spearmanr(zv, mv).correlation:+.3f}   "
                         f"Spearman(verdict, mean speech read) {spearmanr(zv, ms).correlation:+.3f}")
            lines.append(f"  windows: picture only > 0 {n_po / n_all:.3f}   speech only > 0 {n_so / n_all:.3f}   both > 0 {n_both / n_all:.3f}")
            lines.append(f"  lag-1 autocorrelation within video: visual {np.mean(ac['z_visual']):.3f}   speech {np.mean(ac['z_speech']):.3f}")
            lines.append(f"  median within-video Spearman(visual, speech) {np.median(rho):.3f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
