#!/usr/bin/env python3
"""Oracle over the pre-decision prefix representation (diagnostic only, rule 10).

Fits a classifier on the TEST window labels with grouped cross-validation over the vectors written by
prefix_probe.py, and compares the out-of-fold within-video AUC against the frozen read on the same windows.
Windows without a frame have no vector and are excluded from both sides of the comparison, so the numbers
are not directly comparable to the full-window ceilings; the comparison inside this script is.
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
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True, help="runs/<exp>/<run> containing prefix_hidden/")
    ap.add_argument("--frozen-run", default=str(ROOT / "runs/20260910_spvl/mllm/q3vl-8b/full/predictions.jsonl"))
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--dims", type=int, default=64)
    ap.add_argument("--model", choices=["logreg", "gbt"], default="logreg")
    args = ap.parse_args()

    frozen = {}
    for line in open(args.frozen_run):
        r = json.loads(line)
        if r.get("extra") and r["extra"].get("windows"):
            frozen[(r["dataset"], r["video_id"])] = {w["i"]: float(w["z"]) for w in r["extra"]["windows"]}

    hid = Path(args.probe_dir) / "prefix_hidden"
    for ds in args.datasets:
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        gt = {str(v): np.asarray(y, float) for v, y in zip(g["video_ids"], g["y4"])}
        X, Y, G, Z = [], [], [], []
        for p in sorted(hid.glob(f"{ds}__*.npz")):
            vid = p.name[len(ds) + 2:-4]
            y = gt.get(vid)
            fz = frozen.get((ds, vid))
            if y is None or fz is None:
                continue
            d = np.load(p)
            for k, vec in zip(d["keys"].tolist(), d["H"]):
                a, b = int(round(k * 8 * 4)), min(int(round((k + 1) * 8 * 4)), len(y))
                seg = y[a:b]
                if not seg.size or k not in fz:
                    continue
                X.append(vec.astype(np.float32)); Y.append(1 if seg.mean() > 0.5 else 0)
                G.append(vid); Z.append(fz[k])
        Y = np.asarray(Y); G = np.asarray(G); Z = np.asarray(Z); X = np.asarray(X)

        def within(scores):
            out = []
            for v in np.unique(G):
                m = G == v
                if m.sum() > 1 and Y[m].min() != Y[m].max():
                    out.append(roc_auc_score(Y[m], scores[m]))
            return float(np.mean(out)), len(out)

        base, nv = within(Z)
        print(f"\n=== {ds}: {len(Y)} windows with a frame, {len(set(G))} videos, {nv} with both classes")
        print(f"  frozen read on these windows (no fitting) = {base:.4f}")
        A = X - X.mean(0)
        U, S, _ = np.linalg.svd(A, full_matrices=False)
        F = StandardScaler().fit_transform(U[:, :args.dims] * S[:args.dims])
        oof = np.zeros(len(Y))
        for tr, te in GroupKFold(n_splits=min(5, len(set(G)))).split(F, Y, groups=G):
            clf = (HistGradientBoostingClassifier(max_iter=200, random_state=0) if args.model == "gbt"
                   else LogisticRegression(max_iter=2000))
            clf.fit(F[tr], Y[tr])
            oof[te] = clf.decision_function(F[te])
        v, _ = within(oof)
        print(f"  ORACLE on the pre-decision prefix representation ({args.dims} dims, {args.model}) = {v:.4f}"
              f"   ({v - base:+.4f})")


if __name__ == "__main__":
    main()
