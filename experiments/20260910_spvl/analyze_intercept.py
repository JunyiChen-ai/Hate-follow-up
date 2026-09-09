#!/usr/bin/env python3
"""Rule-10 test error analysis: which video-level statistic ranks videos best (video-level ROC / AP),
and pooled frame metrics for candidate intercepts. Reads test GT; results are development evidence."""
import json, sys, os, subprocess
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = Path(__file__).resolve().parents[2]
run = ROOT / "runs/20260910_spvl/full"
rows = [json.loads(l) for l in open(run / "predictions.jsonl") if l.strip()]
legacy = {}
for ds in ("HateMM", "HateClipSeg"):
    for l in open(ROOT / f"data/omsl_v6_inputs/holistic_consistent/{ds}/scores.jsonl"):
        d = json.loads(l); legacy[(ds, d["video_id"])] = float(d["z"])
gt = {}
for ds in ("HateMM", "HateClipSeg"):
    g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
    for vid, y in zip(g["video_ids"], g["y4"]):
        y = np.asarray(y); gt[(ds, str(vid))] = (int(y.max() > 0), float(y.mean()))

def lme(v):
    v = np.asarray(v, float); m = v.max(); return float(m + np.log(np.mean(np.exp(v - m))))

def stats(r):
    w = np.array([x["z"] for x in r["extra"]["windows"]], float); zv = float(r["extra"]["z_video"])
    top3 = np.sort(w)[-3:].mean() if len(w) >= 3 else w.mean()
    return {"z_video": zv, "legacy_z": legacy[(r["dataset"], r["video_id"])], "max_win": w.max(), "mean_win": w.mean(),
            "top3_win": top3, "lme_win": lme(w), "z_video+top3": zv + top3, "z_video+max": zv + w.max(),
            "lme(z_video,top3)": lme([zv, top3]), "z_video+mean": zv + w.mean(), "frac_pos_win": float((w > 0).mean()),
            "z_video+lme_win": zv + lme(w)}
for ds in ("HateMM", "HateClipSeg"):
    rs = [r for r in rows if r["dataset"] == ds and not r.get("error")]
    y = np.array([gt[(ds, r["video_id"])][0] for r in rs]); S = [stats(r) for r in rs]
    print(f"== {ds}: videos {len(rs)}, positive videos {y.sum()}")
    for k in S[0]:
        s = np.array([x[k] for x in S])
        print(f"  {k:20s} video ROC {roc_auc_score(y, s):.4f}  AP {average_precision_score(y, s):.4f}")
