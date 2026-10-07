#!/usr/bin/env python3
"""Reporting ablation (README §23): the Temporal Decoder's fused per-cell probability used directly as the frame score.

Reads a saved `twolevel_r2.py --arm m2` run (no EM is rerun). Each record's `extra.cell_prob` is the OR-fused
P(hateful cell | V = 1, reads) that the method turns into frames (twolevel_r2.py, `p` in the prediction loop). Here
score_curve = to_frames(cell_prob, L) with L = len(score_curve) of the source record: no video term, no rank transform.
No labels are read here; metrics only through src/eval/evaluate_four_datasets.py. As a plumbing check, within-video
macro ROC-AUC must equal the source run's (the method's centred rank keeps the within-video order of cell_prob).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from twolevel import ROOT, to_frames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/20260926_twolevel/final_m2", help="saved m2 run (predictions.jsonl, metrics.json)")
    ap.add_argument("--out", default="runs/20260926_twolevel/final_postdirect")
    ap.add_argument("--tag", default="final_postdirect")
    ap.add_argument("--gt-dir", default="data/gt_4fps")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    a = ap.parse_args()
    src, out = ROOT / a.src, ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.pid").write_text(f"{os.getpid()}\n")
    logf = open(out / "run.log", "w")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    commit = subprocess.run(["git", "log", "-1", "--format=%h"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    log(f"host {socket.gethostname()}")
    log(f"code experiments/20260926_twolevel/direct_posterior.py (to_frames from twolevel.py) at commit {commit} "
        f"(plus uncommitted changes if any)")
    log(f"source {a.src}/predictions.jsonl -> {a.out}/predictions.jsonl")

    pred_path = out / "predictions.jsonl"
    n = {}
    with open(src / "predictions.jsonl") as fi, open(pred_path, "w") as fo:
        for line in fi:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error"):
                continue
            L = len(r["score_curve"])
            score = to_frames(np.asarray(r["extra"]["cell_prob"], float), L)
            r["score_curve"] = [float(x) for x in score]
            r["method"] = f"twolevel_r2__{a.tag}"
            r["intervals"] = []
            fo.write(json.dumps(r) + "\n")
            n[r["dataset"]] = n.get(r["dataset"], 0) + 1
    log(f"records written: {n}")

    metrics_path = out / "metrics.json"
    subprocess.run([sys.executable, str(ROOT / "src/eval/evaluate_four_datasets.py"), "--predictions", str(pred_path),
                    "--gt-dir", str(ROOT / a.gt_dir), "--out", str(metrics_path), "--datasets", *a.datasets],
                   check=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=subprocess.DEVNULL)
    (out / "config.json").write_text(json.dumps({**vars(a), "source_predictions": str(src / "predictions.jsonl"),
                                                 "score": "to_frames(extra.cell_prob, len(score_curve))",
                                                 "video_term": None, "rank_transform": None, "host": socket.gethostname(),
                                                 "commit": commit, "date": time.strftime("%Y-%m-%d")}, indent=2))

    new = {p["dataset"]: p for p in json.load(open(metrics_path))["per_dataset"]}
    old = {p["dataset"]: p for p in json.load(open(src / "metrics.json"))["per_dataset"]}
    ok = True
    for ds in a.datasets:
        p, q = new[ds], old[ds]
        d = p["within_video_macro_ROC_AUC"] - q["within_video_macro_ROC_AUC"]
        ok &= abs(d) < 1e-6
        log(f"[{ds}] {a.tag} {p['frame_ROC_AUC']:.4f}/{p['frame_PR_AUC']:.4f}/{p['within_video_macro_ROC_AUC']:.4f}  "
            f"source {q['frame_ROC_AUC']:.4f}/{q['frame_PR_AUC']:.4f}/{q['within_video_macro_ROC_AUC']:.4f}  "
            f"within difference {d:+.2e}")
    if not ok:
        log("WITHIN_CHECK_FAILED")
        sys.exit(1)
    log("within check passed (|difference| < 1e-6 on every corpus)")
    log("RUN_DONE")


if __name__ == "__main__":
    main()
