#!/usr/bin/env python3
"""Compose SDL outputs into frame scores and evaluate through the shared evaluator.

final(t) = [z_video + mean_i(a_i)] + centred rank of a_i, i.e. the SPVL-r2 composition unchanged; the only
difference between an SDL run and the frozen baseline is the adapter that produced a_i. No labels here.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]


def centered_rank(values):
    values = np.asarray(values, float)
    if len(values) <= 1 or np.ptp(values) <= 1e-12:
        return np.zeros_like(values)
    r = (rankdata(values, method="average") - 0.5) / len(values) - 0.5
    return r - r.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", default="izv_plus_mean_rrank")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--intercept", choices=["adapted", "frozen"], default="adapted",
                    help="frozen: use the frozen run's z_video and mean window score as the intercept, so "
                         "only the adapted within-video ordering can move the numbers (control arm)")
    ap.add_argument("--frozen-run", default=str(ROOT / "runs/20260910_spvl/mllm/q3vl-8b/full/predictions.jsonl"))
    ap.add_argument("--gt-dir", default=str(ROOT / "data/gt_4fps"),
                    help="use data/gt_4fps_hate_only for the HateClipSeg secondary evaluation")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    ws = float(cfg.get("window_seconds", 8.0))
    rows = [json.loads(l) for l in open(run_dir / "predictions.jsonl") if l.strip()]
    frozen = {}
    if a.intercept == "frozen":
        for line in open(a.frozen_run):
            r = json.loads(line)
            if r.get("extra") and r["extra"].get("windows"):
                w = np.array([x["z"] for x in r["extra"]["windows"]], float)
                frozen[(r["dataset"], r["video_id"])] = float(r["extra"]["z_video"]) + float(w.mean())
    out_path = run_dir / f"predictions_{tag}.jsonl"
    n_ok = 0
    with open(out_path, "w") as fh:
        for r in rows:
            if r.get("error") or not r.get("score_curve"):
                fh.write(json.dumps({**r, "method": f"{r.get('method', 'sdl')}__{tag}"}) + "\n")
                continue
            W = r["extra"]["windows"]
            av = np.array([w["a"] for w in W], float)
            n = len(r["score_curve"])
            centers = (np.arange(n) + 0.5) / 4.0
            idx = np.minimum((centers // ws).astype(int), len(av) - 1)
            inter = (frozen[(r["dataset"], r["video_id"])] if a.intercept == "frozen"
                     else float(r["extra"]["z_video"]) + float(av.mean()))
            final = inter + centered_rank(av[idx])
            fh.write(json.dumps({**r, "method": f"{r.get('method', 'sdl')}__{tag}",
                                 "score_curve": [float(x) for x in final]}) + "\n")
            n_ok += 1
    metrics_path = run_dir / f"metrics_{tag}.json"
    cmd = [sys.executable, str(ROOT / "src/eval/evaluate_four_datasets.py"), "--predictions", str(out_path),
           "--gt-dir", a.gt_dir, "--out", str(metrics_path), "--datasets", *a.datasets]
    subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
    d = json.load(open(metrics_path))
    print(f"composed {n_ok}/{len(rows)} -> {metrics_path}  (gt {a.gt_dir})")
    for p in d["per_dataset"]:
        print(f"  {p['dataset']:12s} ROC {p['frame_ROC_AUC']:.4f}  PR {p['frame_PR_AUC']:.4f}  "
              f"within {p['within_video_macro_ROC_AUC']:.4f} (n={p['n_videos_defined']})")


if __name__ == "__main__":
    main()
