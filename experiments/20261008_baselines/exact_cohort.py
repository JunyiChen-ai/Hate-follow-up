#!/usr/bin/env python3
"""Exact-cohort finalisation for the label-free baseline runs made on uoa-lab2 (2026-10-08).

Used by `zs_clip/`, `zs_imagebind_audio/`, `lavad/`, `qwen3_text/` and `qwen25vl_winonly/` in this directory.
A method script hands over one 4 fps curve per video; this module

  1. restricts the rows to the fixed test cohort (`hate_query.md` §3: HateMM 215, HateClipSeg 118, DeHate 1151);
  2. checks that every cohort video has a finite score on every GT frame (the first len(y4) frames; only the
     array length of the GT is read here, never its values);
  3. applies fallback F2 of `run_plan.md` §1.3 to a video with no usable output (constant curve at the median of
     the method's frame scores over the successfully scored cohort videos of the same corpus; stop if more than
     1 % of the cohort needs it);
  4. writes `predictions.jsonl` (shared schema, cohort rows only), `coverage.json`, `config.json`;
  5. calls `src/eval/evaluate_four_datasets.py` unchanged and checks that the evaluator saw exactly the cohort.

No labels are read: the GT file is opened only for its video ids and array lengths. Test labels reach only the
evaluator subprocess.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP_ID = "20261008_baselines"
OUT_ROOT = REPO / "runs" / EXP_ID
COHORT_PRED = REPO / "runs/20260926_twolevel/final_rawkey/predictions.jsonl"
GT_DIR = REPO / "data/gt_4fps"
MANIFEST = {
    "HateMM": REPO / "data/omsl_v6_inputs/manifests/all_test.jsonl",
    "HateClipSeg": REPO / "data/omsl_v6_inputs/manifests/all_test.jsonl",
    "DeHate": REPO / "data/manifests/DeHate_test.jsonl",
}
COHORT_SIZE = {"HateMM": 215, "HateClipSeg": 118, "DeHate": 1151}
FPS = 4.0
F2_STOP_SHARE = 0.01


# ------------------------------------------------------------------ cohort ---
def manifest_rows(ds: str) -> dict[str, dict]:
    out = {}
    with MANIFEST[ds].open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["dataset"] == ds:
                out[r["video_id"]] = r
    return out


def gt_lengths(ds: str) -> dict[str, int]:
    """Test-split video id -> number of GT frames. Only lengths are read."""
    z = np.load(GT_DIR / f"{ds}.npz", allow_pickle=True)
    return {str(v): int(len(z["y4"][i])) for i, v in enumerate(z["video_ids"]) if str(z["split"][i]) == "test"}


def cohort(ds: str) -> list[str]:
    if ds in ("HateMM", "HateClipSeg"):
        ids = set()
        with COHORT_PRED.open() as fh:
            for line in fh:
                r = json.loads(line)
                if r["dataset"] == ds:
                    ids.add(r["video_id"])
    else:
        ids = set(gt_lengths(ds))
    ids = sorted(ids)
    if len(ids) != COHORT_SIZE[ds]:
        raise SystemExit(f"cohort {ds}: {len(ids)} ids, expected {COHORT_SIZE[ds]}")
    return ids


def durations(ds: str) -> dict[str, float]:
    return {v: float(r["duration"]) for v, r in manifest_rows(ds).items()}


def curve_length(duration: float) -> int:
    return max(1, int(math.ceil(duration * FPS - 1e-9)))


# ----------------------------------------------------------- rasterisation ---
def broadcast_to_4fps(curve, native_rate: float, T: int) -> np.ndarray:
    """Retrieval-hate `eval_frame.broadcast_to_4fps` (same arithmetic as `convert_rh_baselines.py`): 4 fps frame i
    takes native sample floor(i / 4 * rate), index clipped to the last sample; frames past the last native sample
    hold its value."""
    curve = np.asarray(curve, dtype=np.float64).reshape(-1)
    if native_rate == FPS:
        s = curve
    else:
        idx = np.floor(np.arange(T) / FPS * native_rate).astype(int)
        idx = np.clip(idx, 0, len(curve) - 1)
        s = curve[idx]
    if len(s) < T:
        s = np.concatenate([s, np.full(T - len(s), s[-1] if len(s) else np.nan)])
    return s[:T].astype(np.float64)


def tail_frames(n_native: int, native_rate: float, T: int) -> int:
    """Frames of a T-frame 4 fps curve that lie past the method's last native sample (held by broadcast)."""
    covered = n_native if native_rate == FPS else int(math.ceil(n_native * FPS / native_rate))
    return max(0, T - covered)


# --------------------------------------------------------------- logging ---
class RunLog:
    """run.log whose first line is the host name."""

    def __init__(self, path: Path, append: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not (append and path.exists())
        self.fh = open(path, "a" if append else "w")
        if fresh:
            self.fh.write(f"host {socket.gethostname()}\n")
        self.fh.flush()

    def __call__(self, msg: str) -> None:
        line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        self.fh.write(line + "\n")
        self.fh.flush()
        print(line, flush=True)

    def close(self) -> None:
        self.fh.close()


def code_version() -> str:
    """Readable code-version note (path + date + commit), as CLAUDE.md asks for run outputs."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                                text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/20261008_baselines"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
    except Exception:
        commit, dirty = "unknown", ""
    return f"commit {commit}{' + uncommitted changes in experiments/20261008_baselines' if dirty else ''}, " \
           f"{datetime.date.today().isoformat()}"


# --------------------------------------------------------------- finalise ---
def finalize(method: str, ds: str, out_dir: Path, curves: dict[str, np.ndarray], *, native_rate: float,
             code_path: str, config: dict, log: RunLog, extra: dict | None = None,
             failures: dict | None = None, notes: dict | None = None) -> dict:
    """curves: video id -> 4 fps curve of length >= the GT length (normally ceil(4 * duration)).
    failures: video id -> reason, for videos the method could not score at all (candidates for F2).
    Returns the coverage report; writes metrics.json only if the exact-cohort rule holds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = cohort(ds)
    dur = durations(ds)
    t_gt = gt_lengths(ds)
    failures = dict(failures or {})
    extra = extra or {}
    ok, bad = {}, {}
    for v in ids:
        c = curves.get(v)
        if c is None:
            bad[v] = failures.get(v, "no_output")
            continue
        c = np.asarray(c, dtype=np.float64)
        T = t_gt[v]
        if len(c) < T:
            bad[v] = f"curve shorter than GT ({len(c)} < {T})"
        elif not np.isfinite(c[:T]).all():
            bad[v] = f"{int((~np.isfinite(c[:T])).sum())} non-finite GT frames"
        else:
            ok[v] = c
    n_f2 = len(bad)
    report = {"dataset": ds, "cohort_videos": len(ids), "videos_scored": len(ok),
              "f2_videos": bad, "f2_share": n_f2 / len(ids), "notes": notes or {}}
    if n_f2:
        if n_f2 > F2_STOP_SHARE * len(ids):
            report["exact_test_set"] = False
            report["stopped"] = f"F2 needed for {n_f2}/{len(ids)} videos (> 1 %); not evaluated"
            (out_dir / "coverage.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            log(f"FAILED exact-cohort rule: {report['stopped']}: {bad}")
            return report
        med = float(np.median(np.concatenate([c for c in ok.values()])))
        report["f2_constant"] = med
        for v, why in bad.items():
            log(f"F2 {ds}/{v}: {why} -> constant {med:.6g}")
            ok[v] = np.full(curve_length(dur[v]), med)
            extra.setdefault(v, {})["fallback"] = {"code": "F2", "reason": why, "value": med}
    report["exact_test_set"] = True
    pred = out_dir / "predictions.jsonl"
    with pred.open("w") as fh:
        for v in ids:
            c = ok[v]
            row = {"schema_version": 1, "method": method, "dataset": ds, "video_id": v, "duration": dur[v],
                   "native_rate": native_rate, "score_curve": [float(x) for x in c], "intervals": [],
                   "error": None, "calls": None, "seed": None, "code_path": code_path,
                   "extra": extra.get(v, {})}
            fh.write(json.dumps(row) + "\n")
    cfg = {"exp_id": EXP_ID, "method": method, "dataset": ds, "date": datetime.date.today().isoformat(),
           "host": socket.gethostname(), "code": code_path, "code_version": code_version(),
           "cohort": (f"{COHORT_PRED.relative_to(REPO)} (dataset {ds})" if ds != "DeHate"
                      else "test video ids of data/gt_4fps/DeHate.npz"),
           "cohort_size": len(ids), "durations": str(MANIFEST[ds].relative_to(REPO)),
           "exact_test_set_rule": "every cohort video has a finite score on every GT frame; F2 fallback otherwise "
                                  "(run_plan.md §1.3), stop above 1 %",
           "evaluator": f"python3 -m src.eval.evaluate_four_datasets --predictions {pred.relative_to(REPO)} "
                        f"--gt-dir data/gt_4fps --out {(out_dir / 'metrics.json').relative_to(REPO)} --datasets {ds}",
           **config}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    metrics = out_dir / "metrics.json"
    # EVAL_PYTHON: the jobs on uoa-lab2 evaluate with the HateVideo env (the env of every other evaluated row)
    cmd = [os.environ.get("EVAL_PYTHON", sys.executable), "-m", "src.eval.evaluate_four_datasets",
           "--predictions", str(pred),
           "--gt-dir", str(GT_DIR), "--out", str(metrics), "--datasets", ds]
    log("evaluator " + " ".join(cmd[1:]))
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(REPO)})
    if res.returncode != 0:
        log(f"FAILED evaluator rc={res.returncode}: {res.stderr[-2000:]}")
        raise SystemExit(res.returncode)
    m = json.loads(metrics.read_text())
    row = [r for r in m["per_dataset"] if r["dataset"] == ds][0]
    if row["n_videos_predicted"] != len(ids) or row["n_videos_overlap"] != len(ids):
        log(f"FAILED evaluator saw {row['n_videos_predicted']} predicted / {row['n_videos_overlap']} overlapping "
            f"videos, expected {len(ids)}")
        metrics.rename(out_dir / "metrics_REJECTED.json")
        raise SystemExit(3)
    report["n_frames_evaluated"] = row["n_frames"]
    (out_dir / "coverage.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log(f"{ds}: ROC {row['frame_ROC_AUC']:.4f} PR {row['frame_PR_AUC']:.4f} within "
        f"{row['within_video_macro_ROC_AUC']:.4f} n_videos {row['n_videos_predicted']} n_frames {row['n_frames']} "
        f"n_within {row['n_videos_defined']} F2 {n_f2}")
    return report
