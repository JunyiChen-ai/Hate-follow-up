#!/usr/bin/env python3
"""TAD E0 checks (README section 5): is the topic read a separate measurement, and does removing it help?

Reads runs/<exp_id>/<run>/predictions.jsonl (act and topic log-odds per window) and the 4 fps GT.
The GT is a test read used for error analysis and for the declared gate (rule 10); it never enters the
scores. Prints the degeneracy check, the validity check, and within-video AUC of every declared variant.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sklearn.metrics import roc_auc_score  # noqa: E402


def window_labels(y4, wins):
    out = []
    for t1, t2 in wins:
        a, b = int(round(t1 * 4)), min(int(round(t2 * 4)), len(y4))
        seg = y4[a:b]
        out.append(1 if seg.size and float(seg.mean()) > 0.5 else 0)
    return out


def ols_beta(a, t):
    t = np.asarray(t, float)
    if len(t) < 2 or float(np.var(t)) < 1e-6:
        return 0.0
    return float(np.clip(np.cov(np.asarray(a, float), t, ddof=0)[0, 1] / np.var(t), 0.0, 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(args.run_dir) / "predictions.jsonl") if l.strip()]

    for ds in args.datasets:
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        gt = {str(v): np.asarray(y, float) for v, y in zip(g["video_ids"], g["y4"])}
        R = [r for r in rows if r["dataset"] == ds and not r.get("error") and r.get("extra")]
        sp, betas = [], []
        auc = {k: [] for k in ["act", "topic", "corr_ols", "corr_1.0", "corr_0.5"]}
        cells = {(l, m): [] for l in (0, 1) for m in (0, 1)}
        for r in R:
            y = gt.get(r["video_id"])
            if y is None:
                continue
            W = r["extra"]["windows"]
            a = np.array([w["a"] for w in W], float)
            t = np.array([w.get("t", np.nan) for w in W], float)
            if np.isnan(t).any() or len(a) < 2:
                continue
            lab = np.array(window_labels(y, [(w["start"], w["end"]) for w in W]))
            if lab.min() == lab.max():
                continue
            if a.std() > 0 and t.std() > 0:
                sp.append(float(spearmanr(a, t).correlation))
            b = ols_beta(a, t)
            betas.append(b)
            curves = {"act": a, "topic": t, "corr_ols": a - b * t, "corr_1.0": a - 1.0 * t,
                      "corr_0.5": a - 0.5 * t}
            for k, c in curves.items():
                auc[k].append(roc_auc_score(lab, c))
            # validity: mean topic log-odds in the four GT x (topic read positive) cells
            for k in range(len(lab)):
                cells[(int(lab[k]), int(t[k] > 0))].append(t[k])
        print(f"\n=== {ds}: {len(auc['act'])} videos with both classes")
        print(f"  degeneracy check  median Spearman(act, topic) = {np.median(sp):+.3f}   "
              f"(stop if >= .9)   mean {np.mean(sp):+.3f}")
        print(f"  beta (per-video OLS, clipped [0,2]): mean {np.mean(betas):.3f}  median {np.median(betas):.3f}  "
              f"frac zero {np.mean(np.array(betas) == 0):.2f}")
        print(f"  topic read positive rate: GT-neg {np.mean([1 for v in cells[(0,1)]]) if cells[(0,1)] else 0:.0f}"
              f" windows, GT-pos {len(cells[(1,1)])} windows")
        base = np.mean(auc["act"])
        for k in ["act", "topic", "corr_ols", "corr_1.0", "corr_0.5"]:
            d = np.mean(auc[k]) - base
            print(f"  window-level within  {k:9s} = {np.mean(auc[k]):.4f}" + (f"   ({d:+.4f} vs act)" if k != "act" else "   (baseline)"))


if __name__ == "__main__":
    main()
