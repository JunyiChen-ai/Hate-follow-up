#!/usr/bin/env python3
"""Compose TAD raw outputs into frame scores and evaluate through the shared evaluator.

final(t) = intercept + residual, intercept = z_video + mean_i(a_i) (unchanged from SPVL-r2),
residual = centred rank of the chosen window curve:
  --curve act        a_i                      (= SPVL-r2, the baseline row)
  --curve topic      t_i                      (validity check: what the topic read alone orders)
  --curve corrected  a_i - beta * t_i         (--beta ols | 1.0 | 0.5)
  --curve permuted   a_i - beta * t'_i        (control: t' from a DIFFERENT video, cyclically shifted)

beta=ols is this video's own cov(a,t)/var(t), clipped to [0,2], 0 when var(t) < 1e-6. No labels here.
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


def ols_beta(a, t):
    t = np.asarray(t, float)
    if len(t) < 2 or float(np.var(t)) < 1e-6:
        return 0.0
    b = float(np.cov(np.asarray(a, float), t, ddof=0)[0, 1] / np.var(t))
    return float(np.clip(b, 0.0, 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--curve", choices=["act", "topic", "corrected", "permuted", "actmargin",
                                        "act_plus_margin"], default="corrected")
    ap.add_argument("--beta", default="ols", help="ols | a float")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    tag = a.tag or (f"{a.curve}" + (f"_b{a.beta}" if a.curve in ("corrected", "permuted") else ""))
    rows = [json.loads(l) for l in open(run_dir / "predictions.jsonl") if l.strip()]

    # control: each video takes the NEXT video's topic curve (same dataset, fixed order), resized by index
    order = [r for r in rows if not r.get("error") and r.get("extra")]
    perm = {}
    for ds in a.datasets:
        sub = [r for r in order if r["dataset"] == ds]
        for k, r in enumerate(sub):
            src = sub[(k + 1) % len(sub)] if len(sub) > 1 else r
            perm[(ds, r["video_id"])] = [w.get("t") for w in src["extra"]["windows"]]

    out_path = run_dir / f"predictions_{tag}.jsonl"
    betas, n_ok = [], 0
    with open(out_path, "w") as fh:
        for r in rows:
            if r.get("error") or not r.get("score_curve"):
                fh.write(json.dumps({**r, "method": f"{r.get('method', 'tad')}__{tag}"}) + "\n")
                continue
            W = r["extra"]["windows"]
            av = np.array([w["a"] for w in W], float)
            raw = np.asarray(r["score_curve"], float)
            n = len(raw)
            idx = np.minimum((np.arange(n) * len(av) / n).astype(int), len(av) - 1)
            if a.curve == "act":
                cur = av
            elif a.curve == "topic":
                cur = np.array([w.get("t", 0.0) for w in W], float)
            elif a.curve == "actmargin":  # round 2: log P(attacks) - logsumexp(other speech acts)
                cur = np.array([w["act_margin"] for w in W], float)
            elif a.curve == "act_plus_margin":  # rank-sum of the binary evidence read and the speech-act read
                cur = centered_rank(av) + centered_rank(np.array([w["act_margin"] for w in W], float))
            else:
                if a.curve == "permuted":
                    src = perm[(r["dataset"], r["video_id"])]
                    tv = np.array([src[k % len(src)] if src and src[k % len(src)] is not None else 0.0
                                   for k in range(len(W))], float)
                else:
                    tv = np.array([w.get("t", 0.0) for w in W], float)
                b = ols_beta(av, tv) if a.beta == "ols" else float(a.beta)
                betas.append(b)
                cur = av - b * tv
            intercept = float(r["extra"]["z_video"]) + float(av.mean())
            final = intercept + centered_rank(cur)[idx]
            fh.write(json.dumps({**r, "method": f"{r.get('method', 'tad')}__{tag}",
                                 "score_curve": [float(x) for x in final],
                                 "compose": {"curve": a.curve, "beta": a.beta}}) + "\n")
            n_ok += 1

    metrics_path = run_dir / f"metrics_{tag}.json"
    cmd = [sys.executable, str(ROOT / "src/eval/evaluate_four_datasets.py"), "--predictions", str(out_path),
           "--gt-dir", str(ROOT / "data/gt_4fps"), "--out", str(metrics_path), "--datasets", *a.datasets]
    subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
    d = json.load(open(metrics_path))
    print(f"composed {n_ok}/{len(rows)} -> {metrics_path}"
          + (f"  mean beta {np.mean(betas):.3f} (median {np.median(betas):.3f})" if betas else ""))
    for p in d["per_dataset"]:
        print(f"  {p['dataset']:12s} ROC {p['frame_ROC_AUC']:.4f}  PR {p['frame_PR_AUC']:.4f}  "
              f"within {p['within_video_macro_ROC_AUC']:.4f} (n={p['n_videos_defined']})")


if __name__ == "__main__":
    main()
