"""Compute ACC/M-F1/M-P/M-R for the pinned TRIAGE configuration across 4 datasets.

Headline config: stage1=2b, triplet={gemma-3-27b-it, qwen2.5-vl-32b-awq, llava-onevision-qwen2-7b-ov-hf}.
Re-uses the same rescue logic as grid_eval_all.eval_cell, but outputs per-sample preds for full metric computation.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

sys.path.insert(0, "/data/jehc223/EMNLP3/src")
from boundary_rescue.grid_eval_all import (
    baseline_pred_path, entropy_band_path, judge_path,
    load_labels, SKIP_VIDEOS, ld_jsonl
)

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)

SLUG = "2b"
TRIPLET = ("gemma-3-27b-it", "qwen2.5-vl-32b-awq", "llava-onevision-qwen2-7b-ov-hf")
DATASETS = ["HateMM", "MHClip_EN", "MHClip_ZH", "ImpliHateVid"]

# Per-dataset criterion from paper_plan headline protocol
CRITERION = {"MHClip_EN": "otsu", "MHClip_ZH": "gmm", "HateMM": "li_lee", "ImpliHateVid": "gmm"}


def triage_preds(ds):
    crit = CRITERION[ds]
    bp = baseline_pred_path(SLUG, ds, crit)
    if not bp.exists():
        # Fall back to protocol file (special-case for 2b)
        bp = baseline_pred_path(SLUG, ds, "protocol")
    base = {r["video_id"]: int(r["pred_baseline"]) for r in ld_jsonl(bp)}

    band_rows = ld_jsonl(entropy_band_path(SLUG, ds))
    band_set = {r["video_id"] for r in band_rows if r.get("in_band")}

    judges = []
    for j in TRIPLET:
        p = judge_path(j, ds)
        judges.append({r["video_id"]: r for r in ld_jsonl(p)})

    labels = load_labels(ds)
    skip = SKIP_VIDEOS.get(ds, set())
    valid = [v for v in base if v not in skip and labels.get(v) in (0, 1)]

    y, yh = [], []
    for v in valid:
        y.append(labels[v])
        pred = base[v]
        if v in band_set:
            votes = [jd.get(v, {}).get("pred") for jd in judges]
            votes = [p for p in votes if p in (0, 1)]
            if len(votes) >= 2:
                s1 = sum(1 for p in votes if p == 1)
                s0 = len(votes) - s1
                if s1 >= 2:
                    mv = 1
                elif s0 >= 2:
                    mv = 0
                else:
                    mv = pred
                if mv != pred:
                    pred = mv
        yh.append(pred)
    return np.array(y), np.array(yh)


def main():
    rows = []
    for ds in DATASETS:
        y, yh = triage_preds(ds)
        acc = accuracy_score(y, yh) * 100
        mf1 = f1_score(y, yh, average="macro") * 100
        mp = precision_score(y, yh, average="macro", zero_division=0) * 100
        mr = recall_score(y, yh, average="macro", zero_division=0) * 100
        rows.append((ds, acc, mf1, mp, mr))
        print(f"{ds:<16s}  n={len(y):>4d}  ACC={acc:5.1f}  M-F1={mf1:5.1f}  M-P={mp:5.1f}  M-R={mr:5.1f}")


if __name__ == "__main__":
    main()
