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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/20260926_glr/base_gridA"))
    ap.add_argument("--key", choices=["current", "density"], required=True)
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
             "nocoupling": False, "noleak": False}
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
                    key = float(log_expit(lo)) + float(np.log(to_frames(p, v["L"]).mean()))
                score = key + centered_rank(to_frames(np.log(p) - np.log1p(-p), v["L"]))
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
