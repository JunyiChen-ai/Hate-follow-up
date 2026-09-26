#!/usr/bin/env python3
"""Interval output by MAP decoding of the explicit-duration model (README §13). CPU, cached reads.

Time level and key as `c_m2` (`twolevel_r2.py --noleak --k 4 --arm m2 --key calib`). For each video with
P(V = 1 | K) >= .5, the most probable sub-state path of each modality chain (Viterbi) gives its hate cells; a cell
is hateful if any chain's path is in the hate phase; intervals are runs of hateful cells. The score curve is the
`c_m2` composition, so the frame metrics equal `c_m2` and only the intervals differ. No labels are read here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twolevel import CELL, FPS, ROOT, centered_rank, intercept, load_run, prep, to_frames  # noqa: E402
import twolevel_r2 as r2  # noqa: E402


def viterbi(E, ch):
    """Most probable augmented-state path; returns the hate bit per cell."""
    n = E.shape[0]
    with np.errstate(divide="ignore"):
        lT = np.log(ch["T"]); lpi = np.log(ch["pi0"])
    d = lpi + E[0]; bp = np.zeros((n, E.shape[1]), int)
    for c in range(1, n):
        m = d[:, None] + lT
        bp[c] = m.argmax(0); d = m.max(0) + E[c]
    z = np.zeros(n, int); z[-1] = int(d.argmax())
    for c in range(n - 1, 0, -1):
        z[c - 1] = bp[c, z[c]]
    S = ch["S"]
    return ch["hb"][z % S]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/20260926_glr/base_gridA"))
    ap.add_argument("--gate", type=float, default=0.5, help="decode only videos with P(V = 1 | K) >= gate")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-root", default=str(ROOT / "runs/20260926_twolevel"))
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    a = ap.parse_args()
    out = Path(a.out_root) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "w")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line); logf.write(line + "\n"); logf.flush()

    log(f"host {socket.gethostname()}")
    (out / "run.pid").write_text(str(os.getpid()))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    log(f"code experiments/20260926_twolevel/decode.py at commit {commit} (plus uncommitted changes if any)")
    flags = {"k": 4, "d_gap": 80.0, "d_hate": 80.0, "sharedchain": False, "carrier": False, "nocoupling": False,
             "noleak": True}
    run = load_run(a.run)
    log(f"modalities {r2.set_modalities(run)}")
    pred = out / "predictions.jsonl"
    n_int = 0
    with open(pred, "w") as fh:
        for ds in a.datasets:
            vids = [prep(r) for k_, r in sorted(run.items()) if k_[0] == ds]
            P = r2.em(vids, flags, log=lambda m, ds=ds: log(f"[{ds}] {m}"))
            ka, kb = r2.key_calibration([intercept(v) for v in vids])
            log(f"[{ds}] key calibration {ka:.4f} K {kb:+.4f}")
            for v in vids:
                t = r2.video_terms(v, P, flags)
                miss = np.ones(v["n"]); hot = np.zeros(v["n"], bool)
                for mods, ch, ll, g in t["per"]:
                    miss *= 1.0 - r2.p_hate(g, ch)
                    hot |= viterbi(r2.emission(v, mods, P, ch), ch).astype(bool)
                p = np.clip(1.0 - miss, 1e-12, 1 - 1e-12)
                key = ka * intercept(v) + kb
                score = key + centered_rank(to_frames(np.log(p) - np.log1p(-p), v["L"]))
                intervals = []
                if expit(key) >= a.gate:
                    on = np.r_[0, hot.astype(np.int8), 0]
                    idx = np.flatnonzero(np.diff(on)).reshape(-1, 2)
                    dur = float(v["rec"]["duration"])
                    intervals = [[float(s) * CELL, min(float(e) * CELL, dur)] for s, e in idx]
                n_int += len(intervals)
                fh.write(json.dumps({**v["rec"], "method": f"twolevel_decode__{a.tag}", "score_curve": [float(x) for x in score],
                                     "intervals": intervals, "extra": {"p_video": float(expit(key))}}) + "\n")
    log(f"intervals written: {n_int}")
    mp = out / "metrics.json"
    subprocess.run([sys.executable, str(ROOT / "src/eval/evaluate_four_datasets.py"), "--predictions", str(pred),
                    "--gt-dir", str(ROOT / "data/gt_4fps"), "--out", str(mp), "--datasets", *a.datasets],
                   check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=subprocess.DEVNULL)
    d = json.load(open(mp))
    log(f"{a.tag:22s} " + "  ".join(f"{p_['dataset'][:6]} {p_['frame_ROC_AUC']:.4f}/{p_['frame_PR_AUC']:.4f}/"
                                    f"{p_['within_video_macro_ROC_AUC']:.4f} F1@.3/.5/.7 {p_['interval_F1@0.3']:.3f}/"
                                    f"{p_['interval_F1@0.5']:.3f}/{p_['interval_F1@0.7']:.3f}" for p_ in d["per_dataset"]))
    log("RUN_DONE")


if __name__ == "__main__":
    main()
