#!/usr/bin/env python3
"""Compose SPVL raw outputs into final frame scores and evaluate.

final(t) = intercept + residual(t)
  intercept: z_video from the same forward (default), the 2026-08 whole-video z (--intercept legacy),
             or 0 (--intercept none)
  residual : zero-mean centred rank of the raw per-window curve inside the video (default),
             optionally with the Vid-Group visual curve as the primary key (--visual-primary),
             or none (--residual none, i.e. video-level score only)

Reads runs/<exp_id>/<run_name>/predictions.jsonl, writes predictions_<tag>.jsonl and
metrics_<tag>.json via the shared evaluator. No labels are read here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
LEGACY_Z = {ds: ROOT / f"data/omsl_v6_inputs/holistic_consistent/{ds}/scores.jsonl" for ds in ("HateMM", "HateClipSeg")}
VISUAL = ROOT / "data/omsl_v6_inputs/visual_A10_vidgroup_zero_shot_full643_v1.jsonl"


def centered_rank(values):
    values = np.asarray(values, float)
    if len(values) <= 1 or np.ptp(values) <= 1e-12:
        return np.zeros_like(values)
    return (rankdata(values, method="average") - 0.5) / len(values) - 0.5


def lexicographic_residual(primary, secondary):
    """Rank by primary, ties by secondary, remaining ties averaged; zero mean."""
    n = len(primary)
    order = np.lexsort((secondary, primary))
    res = np.empty(n)
    pos = 0
    while pos < n:
        end = pos + 1
        a = order[pos]
        while end < n and primary[order[end]] == primary[a] and secondary[order[end]] == secondary[a]:
            end += 1
        avg = 0.5 * (pos + end - 1)
        res[order[pos:end]] = (avg + 0.5) / n - 0.5
        pos = end
    return res - res.mean()


def resize(curve, length):
    curve = np.asarray(curve, float)
    if len(curve) == length:
        return curve
    if len(curve) == 0:
        return np.zeros(length)
    idx = np.minimum((np.arange(length) * len(curve) / length).astype(int), len(curve) - 1)
    return curve[idx]


def load_legacy_z():
    out = {}
    for ds, p in LEGACY_Z.items():
        for line in open(p):
            r = json.loads(line)
            out[(ds, r["video_id"])] = float(r["z"])
    return out


def load_visual():
    out = {}
    for line in open(VISUAL):
        r = json.loads(line)
        if r.get("error"):
            continue
        out[(r["dataset"], r["video_id"])] = r["score_curve"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--intercept", choices=["spvl", "legacy", "none", "mean_win", "zv_plus_mean", "lme_zv_top3", "zv_plus_lme", "zv_plus_kmean", "zv_plus_extent", "zv_plus_median", "zv_plus_tophalf"], default="spvl")
    ap.add_argument("--k-mean", type=float, default=1.0)
    ap.add_argument("--intercept-scale", type=float, default=1.0, help="multiply the intercept (residual stays in [-0.5, 0.5])")
    ap.add_argument("--residual", choices=["rank", "none"], default="rank")
    ap.add_argument("--visual-primary", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    tag = a.tag or f"i{a.intercept}_r{a.residual}" + ("_vis" if a.visual_primary else "") + (f"_x{a.intercept_scale:g}" if a.intercept_scale != 1.0 else "") + (f"_k{a.k_mean:g}" if a.intercept == "zv_plus_kmean" else "")
    legacy = load_legacy_z() if a.intercept == "legacy" else {}
    visual = load_visual() if a.visual_primary else {}
    rows = [json.loads(l) for l in open(run_dir / "predictions.jsonl") if l.strip()]
    out_path = run_dir / f"predictions_{tag}.jsonl"
    n_ok = 0
    with open(out_path, "w") as fh:
        for r in rows:
            if r.get("error") or not r.get("score_curve"):
                fh.write(json.dumps({**r, "method": f"{r['method']}__{tag}"}) + "\n")
                continue
            raw = np.asarray(r["score_curve"], float)
            key = (r["dataset"], r["video_id"])
            if a.intercept == "spvl":
                zv = r["extra"]["z_video"]
                if zv is None:
                    raise SystemExit(f"{key}: no z_video in this run; use --intercept legacy")
                intercept = float(zv)
            elif a.intercept == "legacy":
                if key not in legacy:
                    raise SystemExit(f"{key}: no legacy z")
                intercept = legacy[key]
            elif a.intercept == "none":
                intercept = 0.0
            else:
                w = np.array([x["z"] for x in r["extra"]["windows"]], float)
                zv = float(r["extra"]["z_video"])
                def lme(v):
                    v = np.asarray(v, float); m = v.max(); return float(m + np.log(np.mean(np.exp(v - m))))
                def top3():
                    return float(np.sort(w)[-3:].mean()) if len(w) >= 3 else float(w.mean())
                def extent():
                    pw = np.clip((1 / (1 + np.exp(-w))).mean(), 1e-6, 1 - 1e-6); return float(np.log(pw / (1 - pw)))
                # evaluated lazily: only the requested statistic is computed (ASR-window runs can have zero windows)
                fns = {"mean_win": lambda: float(w.mean()), "zv_plus_mean": lambda: zv + float(w.mean()),
                       "lme_zv_top3": lambda: lme([zv, top3()]), "zv_plus_lme": lambda: zv + lme(w),
                       "zv_plus_kmean": lambda: zv + a.k_mean * float(w.mean()),
                       "zv_plus_extent": lambda: zv + extent(),  # extent as a log-odds (no free constant)
                       "zv_plus_median": lambda: zv + float(np.median(w)),
                       "zv_plus_tophalf": lambda: zv + float(np.sort(w)[len(w) // 2:].mean())}
                # no windows at all (ASR mode, video without transcript): window statistics are undefined -> 0
                intercept = fns[a.intercept]() if len(w) else (0.0 if a.intercept == "mean_win" else zv)
            intercept *= a.intercept_scale
            if a.residual == "none":
                residual = np.zeros_like(raw)
            elif a.visual_primary:
                if key not in visual:
                    raise SystemExit(f"{key}: no visual curve")
                residual = lexicographic_residual(resize(visual[key], len(raw)), raw)
            else:
                residual = centered_rank(raw)
                residual -= residual.mean()
            final = intercept + residual
            fh.write(json.dumps({**r, "method": f"{r['method']}__{tag}", "score_curve": [float(x) for x in final],
                                 "compose": {"intercept": a.intercept, "residual": a.residual,
                                             "visual_primary": a.visual_primary}}) + "\n")
            n_ok += 1
    metrics_path = run_dir / f"metrics_{tag}.json"
    cmd = [sys.executable, str(ROOT / "src/eval/evaluate_four_datasets.py"), "--predictions", str(out_path),
           "--gt-dir", str(ROOT / "data/gt_4fps"), "--out", str(metrics_path), "--datasets", *a.datasets]
    import os
    subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
    d = json.load(open(metrics_path))
    print(f"composed {n_ok}/{len(rows)} -> {metrics_path}")
    for p in d["per_dataset"]:
        print(f"  {p['dataset']:12s} ROC {p['frame_ROC_AUC']:.4f}  PR {p['frame_PR_AUC']:.4f}  "
              f"within {p['within_video_macro_ROC_AUC']:.4f} (n={p['n_videos_defined']})")


if __name__ == "__main__":
    main()
