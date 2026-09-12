#!/usr/bin/env python3
"""BND kill test: does asking about onset recover an ordering the "is this window violating" read cannot?

Curves built from the two boundary reads, all label-free and parameter-free:
  begin        b_begin
  cumulative   cumsum(sigmoid(b_begin) - sigmoid(b_end))   -- a presence state integrated from the reads
  act+cum      centred rank of the act read + centred rank of the cumulative curve
Compared against the act read on the same windows. GT is read only to score (rule 10).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scipy.stats import rankdata, spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def crank(v):
    v = np.asarray(v, float)
    if len(v) <= 1 or np.ptp(v) <= 1e-12:
        return np.zeros_like(v)
    r = (rankdata(v, method="average") - 0.5) / len(v) - 0.5
    return r - r.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(a.run_dir) / "predictions.jsonl") if l.strip()]
    for ds in a.datasets:
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        gt = {str(v): np.asarray(y, float) for v, y in zip(g["video_ids"], g["y4"])}
        auc = {k: [] for k in ("act", "begin", "cumulative", "act+cum")}
        sp, onset_hit, onset_n = [], 0, 0
        for r in [x for x in rows if x["dataset"] == ds and not x.get("error") and x.get("extra")]:
            y = gt.get(r["video_id"])
            if y is None:
                continue
            W = r["extra"]["windows"]
            if any("b_begin" not in w for w in W) or len(W) < 2:
                continue
            lab = []
            for w in W:
                i0, i1 = int(round(w["start"] * 4)), min(int(round(w["end"] * 4)), len(y))
                seg = y[i0:i1]
                lab.append(1 if seg.size and seg.mean() > 0.5 else 0)
            lab = np.array(lab)
            if lab.min() == lab.max():
                continue
            act = np.array([w["a"] for w in W], float)
            bb = np.array([w["b_begin"] for w in W], float)
            be = np.array([w["b_end"] for w in W], float)
            cum = np.cumsum(1 / (1 + np.exp(-bb)) - 1 / (1 + np.exp(-be)))
            curves = {"act": act, "begin": bb, "cumulative": cum, "act+cum": crank(act) + crank(cum)}
            for k, c in curves.items():
                auc[k].append(roc_auc_score(lab, c))
            if bb.std() > 0 and act.std() > 0:
                sp.append(float(spearmanr(bb, act).correlation))
            # does the argmax of the begin read land on the first GT-positive window?
            first = int(np.argmax(lab))
            onset_hit += int(int(np.argmax(bb)) == first)
            onset_n += 1
        print(f"\n=== {ds}: {len(auc['act'])} videos")
        for k in ("act", "begin", "cumulative", "act+cum"):
            d = np.mean(auc[k]) - np.mean(auc["act"])
            print(f"  window-level within  {k:11s} = {np.mean(auc[k]):.4f}" + ("" if k == "act" else f"  ({d:+.4f})"))
        print(f"  Spearman(begin read, act read) median {np.median(sp):+.3f}")
        print(f"  argmax of the begin read is the first GT-positive window: {onset_hit}/{onset_n} "
              f"= {onset_hit / max(onset_n, 1):.1%}")


if __name__ == "__main__":
    main()
