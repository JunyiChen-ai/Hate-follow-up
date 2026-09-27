#!/usr/bin/env python3
"""Diagnostic, not the protocol (README §9.3; reads the gold). HateClipSeg segments carry six label dimensions: normal,
hateful, and four other offensive categories. The protocol GT (`data/gt_4fps/HateClipSeg.npz`) marks a frame positive
when any non-normal dimension is set, i.e. offensive content. This script rebuilds that GT from the same source
(`gold_segments.json` of Retrieval-hate, the builder's input), checks that it equals the protocol GT frame for frame,
and then scores the DVD arms against a strict GT where only the hateful dimension counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/jehc223/Retrieval-hate/data/gt/HateClipSeg/gold_segments.json")


def raster(spans, n):
    t = np.arange(n) / 4.0
    y = np.zeros(n, int)
    for a, b in spans:
        y[(t >= a) & (t < b)] = 1
    return y


def scores(recs, Y, vids):
    ys, ss, w = [], [], []
    for v in vids:
        y = Y[v]; s = np.asarray(recs[v]["score_curve"], float)[:len(y)]
        ys.append(y); ss.append(s)
        if y.min() != y.max():
            w.append(roc_auc_score(y, s))
    y, s = np.concatenate(ys), np.concatenate(ss)
    return roc_auc_score(y, s), average_precision_score(y, s), float(np.mean(w)), len(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "runs/20260927_dvd/arms"))
    ap.add_argument("--arms", nargs="+", default=["base", "dvd", "dvd_noT", "dvd_noE", "dvd_ATE"])
    ap.add_argument("--out", default=str(ROOT / "runs/20260927_dvd/analysis_hcs_strict"))
    a = ap.parse_args()
    g = np.load(ROOT / "data/gt_4fps/HateClipSeg.npz", allow_pickle=True)
    Y = {str(v): np.asarray(y, int) for v, y in zip(g["video_ids"], g["y4"])}
    src = json.loads(SOURCE.read_text())
    off, strict = {}, {}
    for v, y in Y.items():
        segs = src[v]["segments"]
        off[v] = raster([(s, e) for s, e, d in segs if any(d[1:])], len(y))
        strict[v] = raster([(s, e) for s, e, d in segs if d[1]], len(y))
    bad = [v for v in Y if not np.array_equal(off[v], Y[v])]
    lines = [f"rebuilt offensive GT equals the protocol GT on {len(Y) - len(bad)} / {len(Y)} videos"]
    recs = {arm: {r["video_id"]: r for r in map(json.loads, open(Path(a.root) / arm / "predictions.jsonl"))
                  if r["dataset"] == "HateClipSeg"} for arm in a.arms}
    vids = sorted(v for v in Y if all(v in r for r in recs.values()))
    for name, gt in (("protocol (offensive)", Y), ("strict (hateful only)", strict)):
        lab = np.array([int(gt[v].any()) for v in vids]); rate = np.mean(np.concatenate([gt[v] for v in vids]))
        lines.append(f"== HateClipSeg GT {name}: {len(vids)} videos, {lab.sum()} positive, frame base rate {rate:.3f}")
        for arm in a.arms:
            roc, pr, wi, nw = scores(recs[arm], gt, vids)
            key = np.array([float(recs[arm][v]["extra"]["p_video_logodds"]) for v in vids])
            lines.append(f"  {arm:10s} {roc:.4f} / {pr:.4f} / {wi:.4f} (within n {nw})   video AUC {roc_auc_score(lab, key):.3f} "
                         f"AP {average_precision_score(lab, key):.3f}   P(V)>.5 negative {(key[lab == 0] > 0).mean():.3f} "
                         f"positive {(key[lab == 1] > 0).mean():.3f}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
