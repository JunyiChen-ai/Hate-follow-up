#!/usr/bin/env python3
"""Per-video paired bootstrap of within-video ROC-AUC between arms (evaluation only: reads GT). Also checks that the
per-video mean equals the evaluator's within number."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs/20260926_twolevel"
ARMS = ["current", "m2", "full", "full_pooled", "abl_nocoupling", "abl_sharedchain", "abl_noleak", "abl_noatleast",
        "abl_vverdict", "abl_vreads"]
PAIRS = [("m2", "current"), ("full", "current"), ("full_pooled", "full"), ("abl_nocoupling", "full"),
         ("abl_sharedchain", "full"), ("abl_noleak", "full"), ("abl_noatleast", "full"), ("abl_vverdict", "full"),
         ("abl_vreads", "full")]
# round 2 (README §10)
ARMS_R2 = ["current", "r2_m2", "r2_full", "r2_k1", "r2_k2", "r2_k8", "r2_d40", "r2_d160", "r2_nocoupling",
           "r2_noleak", "r2_carrier"]
PAIRS_R2 = [("r2_m2", "current"), ("r2_full", "current"), ("r2_k1", "r2_m2"), ("r2_nocoupling", "r2_m2"),
            ("r2_noleak", "r2_m2"), ("r2_carrier", "r2_m2"), ("r2_k2", "r2_m2"), ("r2_k8", "r2_m2"),
            ("r2_d40", "r2_m2"), ("r2_d160", "r2_m2")]


def per_video(arm, ds, Y):
    out = {}
    for l in open(R / arm / "predictions.jsonl"):
        r = json.loads(l)
        if r["dataset"] != ds or r["video_id"] not in Y:
            continue
        y = Y[r["video_id"]]; s = np.asarray(r["score_curve"], float)[:len(y)]
        if len(s) == len(y) and y.min() != y.max():
            out[r["video_id"]] = roc_auc_score(y, s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=[1, 2], default=1)
    a = ap.parse_args()
    arms, pairs, sub = (ARMS, PAIRS, "analysis") if a.round == 1 else (ARMS_R2, PAIRS_R2, "analysis_r2")
    rng = np.random.default_rng(0)
    lines, summary = [], {}
    for ds in ["HateMM", "HateClipSeg"]:
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        Y = {str(v): np.asarray(y, int) for v, y in zip(g["video_ids"], g["y4"])}
        pv = {x: per_video(x, ds, Y) for x in arms if (R / x / "predictions.jsonl").exists()}
        lines.append(f"== {ds}")
        for x, d in pv.items():
            m = json.load(open(R / x / "metrics.json"))
            ev = [p for p in m["per_dataset"] if p["dataset"] == ds][0]
            lines.append(f"  {x:16s} pooled {ev['frame_ROC_AUC']:.4f} / {ev['frame_PR_AUC']:.4f}  within {ev['within_video_macro_ROC_AUC']:.4f} "
                         f"(per-video mean {np.mean(list(d.values())):.4f}, n {len(d)})  F1@.3/.5/.7 "
                         f"{ev['interval_F1@0.3']:.3f} / {ev['interval_F1@0.5']:.3f} / {ev['interval_F1@0.7']:.3f}")
        for x, b in pairs:
            if x not in pv or b not in pv:
                continue
            vids = sorted(set(pv[x]) & set(pv[b]))
            d = np.array([pv[x][v] - pv[b][v] for v in vids])
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
            lo, hi = np.quantile(bs, [.025, .975])
            summary[f"{ds}:{x}-{b}"] = [float(d.mean()), float(lo), float(hi)]
            lines.append(f"  within {x} - {b}: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]")
    (R / sub).mkdir(parents=True, exist_ok=True)
    (R / sub / "table.txt").write_text("\n".join(lines) + "\n")
    (R / sub / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
