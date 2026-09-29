#!/usr/bin/env python3
"""DVD analysis (README §5; evaluation only, reads the gold). For each corpus and arm:
- the three main metrics, copied from the evaluator's metrics.json;
- the video level: share of non-hateful and of hateful videos with P(V = 1) > .5 (key > 0), video AUC / AP of the key;
- paired bootstrap over videos (4000, seed 0) of arm minus base, for pooled ROC / PR (sklearn with frame weights = the
  video's multiplicity) and within (per-video ROC averaged over videos with both classes);
- among hateful videos only (added after the first results, README §8): pooled ROC / PR and the Spearman correlation of
  the key with the video's hate coverage, i.e. whether the key orders hateful videos by how much of them is hate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
G = {}


def _boot(cs):
    return [metrics(G["p"], c) for c in cs]


def boot(p, draws):
    G["p"] = p
    with Pool(14) as pool:
        parts = pool.map(_boot, [draws[i::14] for i in range(14)])
    out = np.zeros((len(draws), 3))
    for i in range(14):
        out[i::14] = np.array(parts[i])
    return out


def load(pred):
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, open(pred))}


def pack(recs, Y, vids, ds):
    ys, ss, idx, wv, wauc, key = [], [], [], [], [], []
    for k, v in enumerate(vids):
        r = recs[(ds, v)]; y = Y[v]; s = np.asarray(r["score_curve"], float)
        n = min(len(y), len(s)); y, s = y[:n], s[:n]
        ys.append(y); ss.append(s); idx.append(np.full(n, k)); key.append(float(r["extra"].get("p_video_logodds", r["extra"]["z_video"])))  # SPVL-composed runs: verdict
        if y.min() != y.max():
            wv.append(k); wauc.append(roc_auc_score(y, s))
    return {"y": np.concatenate(ys), "s": np.concatenate(ss), "i": np.concatenate(idx), "wv": np.array(wv),
            "wauc": np.array(wauc), "key": np.array(key)}


def metrics(p, c):
    w = c[p["i"]].astype(float); m = w > 0; wd = c[p["wv"]].astype(float)
    return np.array([roc_auc_score(p["y"][m], p["s"][m], sample_weight=w[m]),
                     average_precision_score(p["y"][m], p["s"][m], sample_weight=w[m]),
                     float((wd * p["wauc"]).sum() / wd.sum())])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory holding <arm>/predictions.jsonl")
    ap.add_argument("--arms", nargs="+", required=True, help="first arm is the base")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root); lines, summary = [], {}
    rng = np.random.default_rng(0)
    for ds in a.datasets:
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        Y = {str(v): np.asarray(y, int) for v, y in zip(g["video_ids"], g["y4"])}
        recs = {arm: load(root / arm / "predictions.jsonl") for arm in a.arms}
        # videos with GT and a prediction in every arm (the evaluator also scores only overlapping videos)
        vids = sorted(v for v in Y if all((ds, v) in r for r in recs.values()))
        lab = np.array([int(Y[v].any()) for v in vids])
        cov = np.array([Y[v].mean() for v in vids])
        P = {arm: pack(recs[arm], Y, vids, ds) for arm in a.arms}
        draws = [np.bincount(rng.integers(0, len(vids), len(vids)), minlength=len(vids)) for _ in range(4000)]
        base = a.arms[0]; ones = np.ones(len(vids), int)
        base_draws = boot(P[base], draws)
        lines.append(f"== {ds} ({len(vids)} videos, {lab.sum()} hateful)")
        for arm in a.arms:
            p = P[arm]; m = metrics(p, ones); k = p["key"]
            row = {"metrics": m.tolist(), "fp_share": float((k[lab == 0] > 0).mean()),
                   "tp_share": float((k[lab == 1] > 0).mean()), "video_auc": float(roc_auc_score(lab, k)),
                   "video_ap": float(average_precision_score(lab, k))}
            hm = metrics(p, lab.astype(int)); rho = spearmanr(k[lab == 1], cov[lab == 1])[0]
            row.update({"hateful_only": hm[:2].tolist(), "rho_key_coverage_hateful": float(rho)})
            line = (f"  {arm:14s} {m[0]:.4f} / {m[1]:.4f} / {m[2]:.4f}   P(V)>.5: non-hateful {row['fp_share']:.3f} "
                    f"hateful {row['tp_share']:.3f}   video AUC {row['video_auc']:.3f} AP {row['video_ap']:.3f}\n"
                    f"{'':18s}hateful videos only: pooled {hm[0]:.4f} / {hm[1]:.4f}   Spearman(key, coverage) {rho:+.3f}")
            if arm != base:
                d = boot(p, draws) - base_draws
                lo, hi = np.quantile(d, [.025, .975], axis=0)
                diff = m - metrics(P[base], ones)
                row["diff_vs_base"] = [[float(diff[j]), float(lo[j]), float(hi[j])] for j in range(3)]
                line += "\n" + " " * 18 + "vs base " + "  ".join(
                    f"{nm} {diff[j]:+.4f} [{lo[j]:+.4f}, {hi[j]:+.4f}]" for j, nm in enumerate(("ROC", "PR", "within")))
            summary[f"{ds}:{arm}"] = row
            lines.append(line)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
