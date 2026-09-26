#!/usr/bin/env python3
"""Diagnostic (after round 2): video-level keys for the lexicographic composition, on the round-2 time level.

key + centred rank of logit P(hateful at t | V = 1), with key one of
  current   z_video + mean window z (reproduces r2_m2)
  density   log P(V = 1 | .) + log mean_t P(hateful at t | V = 1, .)   (log expected fraction of hateful frames)
The video posterior P(V = 1 | .) is the joint one with ICC-effective reads (`--vpost icc`) or the two-component
mixture over the verdict and the modality mean reads (`--vpost mix`). No labels are read here.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import log_expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twolevel import ROOT, centered_rank, intercept, load_run, prep, to_frames  # noqa: E402
import twolevel_r2 as r2  # noqa: E402


def mix1d_slope(x, it=2000):
    """Two-component 1-D Gaussian mixture with a shared variance, EM from the 10th / 90th percentiles; returns the
    log-odds slope (mu1 - mu0) / var of the fitted posterior. Label-free."""
    mu = np.percentile(x, [10, 90]).astype(float); var = float(x.var()); pi = 0.5
    for _ in range(it):
        l0 = np.log(1 - pi) - 0.5 * (x - mu[0]) ** 2 / var; l1 = np.log(pi) - 0.5 * (x - mu[1]) ** 2 / var
        r = 1.0 / (1.0 + np.exp(l0 - l1)); pi = float(np.clip(r.mean(), 1e-6, 1 - 1e-6))
        mu = np.array([((1 - r) * x).sum() / (1 - r).sum(), (r * x).sum() / r.sum()])
        var = float(((1 - r) * (x - mu[0]) ** 2 + r * (x - mu[1]) ** 2).mean())
    return float(abs(mu[1] - mu[0]) / var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/20260926_glr/base_gridA"))
    ap.add_argument("--key", choices=["current", "density", "vpost"], required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="score = scale * key + centred rank (100 = lexicographic)")
    ap.add_argument("--noleak", action="store_true")
    ap.add_argument("--calib", choices=["none", "mix1d"], default="none",
                    help="mix1d: scale = log-odds slope of a two-component Gaussian mixture (shared variance) fitted to the corpus's keys")
    ap.add_argument("--vpost", choices=["icc", "mix"], default="icc")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    a = ap.parse_args()
    out = ROOT / "runs/20260926_twolevel" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "w")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line); logf.write(line + "\n"); logf.flush()

    log(f"host {socket.gethostname()}")
    (out / "run.pid").write_text(str(os.getpid()))
    flags = {"k": a.k, "d_gap": 80.0, "d_hate": 80.0, "sharedchain": False, "carrier": False,
             "nocoupling": False, "noleak": a.noleak}
    run = load_run(a.run)
    pred = out / "predictions.jsonl"
    with open(pred, "w") as fh:
        for ds in a.datasets:
            vids = [prep(r) for k_, r in sorted(run.items()) if k_[0] == ds]
            P = r2.em(vids, flags, log=lambda m, ds=ds: log(f"[{ds}] {m}"))
            if a.vpost == "icc":
                P["icc"] = r2.residual_icc(vids, P, flags)
            else:
                vfn, _ = r2.vlevel_mixture(vids)
            scale = a.scale
            if a.calib == "mix1d":
                scale = mix1d_slope(np.array([intercept(v) for v in vids]))
                log(f"[{ds}] key calibration slope {scale:.4f}")
            for v in vids:
                t = r2.video_terms(v, P, flags)
                miss = np.ones(v["n"])
                for mods, ch, ll, g in t["per"]:
                    miss *= 1.0 - r2.p_hate(g, ch)
                p = np.clip(1.0 - miss, 1e-12, 1 - 1e-12)
                if a.key == "current":
                    key = intercept(v)
                else:
                    lo = t["lo_icc"] if a.vpost == "icc" else vfn(v)
                    key = lo if a.key == "vpost" else float(log_expit(lo)) + float(np.log(to_frames(p, v["L"]).mean()))
                score = scale * key + centered_rank(to_frames(np.log(p) - np.log1p(-p), v["L"]))
                fh.write(json.dumps({**v["rec"], "method": f"twolevel_diag__{a.tag}", "score_curve": [float(x) for x in score],
                                     "intervals": [], "extra": {"key": key}}) + "\n")
    mp = out / "metrics.json"
    subprocess.run([sys.executable, str(ROOT / "src/eval/evaluate_four_datasets.py"), "--predictions", str(pred),
                    "--gt-dir", str(ROOT / "data/gt_4fps"), "--out", str(mp), "--datasets", *a.datasets],
                   check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=subprocess.DEVNULL)
    d = json.load(open(mp))
    log(f"{a.tag:28s} " + "  ".join(f"{p_['dataset'][:6]} {p_['frame_ROC_AUC']:.4f}/{p_['frame_PR_AUC']:.4f}/"
                                    f"{p_['within_video_macro_ROC_AUC']:.4f}" for p_ in d["per_dataset"]))
    log("RUN_DONE")


if __name__ == "__main__":
    main()
