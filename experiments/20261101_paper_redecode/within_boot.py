#!/usr/bin/env python3
"""Per-video paired bootstrap of within-video ROC-AUC, arm minus base (evaluation only: reads the test GT).

Copy of the bootstrap in experiments/20260926_twolevel/analyze.py (that script has its run root and arm lists
hard-coded and writes into runs/20260926_twolevel, so it is copied here instead of edited). Same computation: the
per-video ROC-AUC of every video with both classes, the per-video difference arm - base, 4000 resamples of those
videos with replacement, 2.5 / 97.5 % quantiles of the mean. One change: a fresh default_rng(0) per (corpus, pair), so
an interval does not depend on which other pairs are in the list. Also checks that the per-video mean equals the
evaluator's within number in the arm's metrics.json (the evaluator is the only source of the reported metrics).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]


def per_video(path, ds, Y):
    out = {}
    for l in open(path / "predictions.jsonl"):
        r = json.loads(l)
        if r["dataset"] != ds or r["video_id"] not in Y:
            continue
        y = Y[r["video_id"]]; s = np.asarray(r["score_curve"], float)[:len(y)]
        if len(s) == len(y) and y.min() != y.max():
            out[r["video_id"]] = roc_auc_score(y, s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    lines, summary = [], {}
    for ds in a.datasets:
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        Y = {str(v): np.asarray(y, int) for v, y in zip(g["video_ids"], g["y4"])}
        pv = {x: per_video(root / x, ds, Y) for x in [a.base] + a.arms}
        lines.append(f"== {ds}")
        for x, d in pv.items():
            ev = [p for p in json.load(open(root / x / "metrics.json"))["per_dataset"] if p["dataset"] == ds][0]
            mean = float(np.mean(list(d.values())))
            ok = abs(mean - ev["within_video_macro_ROC_AUC"]) < 1e-9
            lines.append(f"  {x:16s} evaluator within {ev['within_video_macro_ROC_AUC']:.4f} (per-video mean {mean:.4f}, "
                         f"n {len(d)}, {'equal' if ok else 'MISMATCH'})")
        for x in a.arms:
            vids = sorted(set(pv[x]) & set(pv[a.base]))
            d = np.array([pv[x][v] - pv[a.base][v] for v in vids])
            rng = np.random.default_rng(0)
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
            lo, hi = np.quantile(bs, [.025, .975])
            summary[f"{ds}:{x}-{a.base}"] = [float(d.mean()), float(lo), float(hi), len(vids)]
            lines.append(f"  within {x} - {a.base}: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  ({len(vids)} videos)")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
