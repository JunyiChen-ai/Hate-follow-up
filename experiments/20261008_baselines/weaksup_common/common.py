"""Shared plumbing for the weakly supervised baselines of 20261008_baselines.

Everything a weakly supervised run reads comes from `data/weaksup_1fps/` (built by `prepare_inputs.py`, provenance
in `data/weaksup_1fps/PROVENANCE.md`):

    splits/<corpus>_{train,val,test}.txt   train / val ids; test = the exact cohort of hate_query.md
    labels/<corpus>.json                   video-level labels of the train and val ids only
    <feature>/<corpus>/<video_id>.npy      one row per second (1 fps)

Corpus keys are the port keys (hatemm, hateclipseg, dehate); dataset names (HateMM, HateClipSeg, DeHate) are the
evaluator's.  Test video labels and every frame label stay out of this module: the only GT access is the frame
count per test video, which the coverage check needs, and the evaluator itself.
"""
from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EXP_ID = "20261008_baselines"
INPUTS = REPO / "data" / "weaksup_1fps"
RUNS = REPO / "runs" / EXP_ID
GT_DIR = REPO / "data" / "gt_4fps"
SEEDS = (2025, 234, 3407)
FPS = 4

CORPORA = ("hatemm", "hateclipseg", "dehate")
DATASET = {"hatemm": "HateMM", "hateclipseg": "HateClipSeg", "dehate": "DeHate"}
CORPUS = {v: k for k, v in DATASET.items()}
COHORT_SIZE = {"hatemm": 215, "hateclipseg": 118, "dehate": 1151}
MANIFESTS = {"hatemm": REPO / "data/omsl_v6_inputs/manifests/all_test.jsonl",
             "hateclipseg": REPO / "data/omsl_v6_inputs/manifests/all_test.jsonl",
             "dehate": REPO / "data/manifests/DeHate_test.jsonl"}
TEST_LABEL_PLACEHOLDER = -1


def read_ids(path):
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def split_ids(corpus, split):
    return read_ids(INPUTS / "splits" / f"{corpus}_{split}.txt")


def train_val_labels(corpus):
    """video_id -> 0/1 for the train and val ids (the file holds nothing else)."""
    return {k: int(v) for k, v in json.loads((INPUTS / "labels" / f"{corpus}.json").read_text()).items()}


def port_labels(corpus):
    """The label map the ports need: train/val labels, and a -1 slot for each test id.

    The ports' dataset classes refuse an id without a label even at inference, where the label is never used.
    The test slot is filled with -1, so no test label is ever read.
    """
    labels = train_val_labels(corpus)
    for vid in split_ids(corpus, "test"):
        if vid in labels:
            raise RuntimeError(f"{corpus}: test id {vid} also has a train/val label")
        labels[vid] = TEST_LABEL_PLACEHOLDER
    return labels


def durations(corpus):
    ds = DATASET[corpus]
    out = {}
    with open(MANIFESTS[corpus]) as fh:
        for line in fh:
            row = json.loads(line)
            if row["dataset"] == ds:
                out[row["video_id"]] = float(row["duration"])
    return out


def gt_lengths(corpus):
    """Frame count of each cohort video's GT array (lengths only; labels are not returned)."""
    z = np.load(GT_DIR / f"{DATASET[corpus]}.npz", allow_pickle=True)
    return {str(v): len(z["y4"][i]) for i, v in enumerate(z["video_ids"]) if str(z["split"][i]) == "test"}


def to_4fps(curve_1fps, duration):
    """1 fps -> 4 fps: repeat each value 4 times, then pad with the last value or cut to ceil(4 * duration).

    The rule of `experiments/20260927_dehate_external/convert_weaksup.py` and run_plan.md §1.2 (re-implemented here,
    no import across experiment directories).
    """
    x = np.repeat(np.asarray(curve_1fps, dtype=float), FPS)
    n = max(1, math.ceil(duration * FPS))
    if len(x) >= n:
        return x[:n], 0
    return np.concatenate([x, np.full(n - len(x), x[-1])]), n - len(x)


class RunLog:
    """run.log whose first line is the host name."""

    def __init__(self, path, mode="a"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.path.exists() or mode == "w"
        self.fh = self.path.open("w" if mode == "w" else "a")
        if fresh:
            self.fh.write(socket.gethostname() + "\n")
            self.fh.flush()

    def __call__(self, msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()


def code_version():
    """Readable code version: commit id and date (CLAUDE.md allows the commit id for code sync only)."""
    try:
        head = subprocess.check_output(["git", "log", "-1", "--format=%h %cs"], cwd=REPO, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=REPO, text=True).strip()
        return f"{head}{' (tracked files modified)' if dirty else ''}"
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({exc})"


def load_scores(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                if rec["video_id"] in out:
                    raise ValueError(f"{path}: duplicate {rec['video_id']}")
                out[rec["video_id"]] = rec
    return out


def build_predictions(corpus, scores_path, branches, method_names, seed, code_path, log):
    """Convert one seed's 1 fps scores.jsonl into shared-schema rows, with the exact-cohort check.

    branches: score keys in scores.jsonl; method_names: the method name written for each branch.
    Returns (rows, coverage).  Raises if any cohort video lacks a finite score on any GT frame.
    """
    ds = DATASET[corpus]
    cohort = split_ids(corpus, "test")
    if len(cohort) != COHORT_SIZE[corpus]:
        raise RuntimeError(f"{corpus}: cohort has {len(cohort)} ids, expected {COHORT_SIZE[corpus]}")
    dur = durations(corpus)
    tgt = gt_lengths(corpus)
    scores = load_scores(scores_path)
    missing = [v for v in cohort if v not in scores]
    extra = sorted(set(scores) - set(cohort))
    if missing:
        raise RuntimeError(f"{corpus}: {len(missing)} cohort videos have no scores, e.g. {missing[:5]}")
    rows, padded, problems = [], {}, []
    for branch, method in zip(branches, method_names):
        for vid in cohort:
            curve, pad = to_4fps(scores[vid][branch], dur[vid])
            if len(curve) < tgt[vid] or not np.all(np.isfinite(curve[:tgt[vid]])):
                problems.append((branch, vid, len(curve), tgt[vid]))
            if pad:
                padded[vid] = pad
            rows.append({"schema_version": 1, "method": method, "dataset": ds, "video_id": vid,
                         "duration": dur[vid], "native_rate": 1.0,
                         "score_curve": [round(float(x), 6) for x in curve], "intervals": [], "error": None,
                         "calls": 0, "seed": seed, "code_path": code_path,
                         "extra": {"branch": branch, "rule_to_4fps": "x4 repeat, last-value pad / cut to ceil(4D)",
                                   "fallback": None}})
    if problems:
        raise RuntimeError(f"{corpus}: {len(problems)} curves do not cover their GT frames, e.g. {problems[:3]}")
    coverage = {"dataset": ds, "cohort_videos": len(cohort), "videos_scored": len(cohort),
                "extra_ids_dropped": extra, "tail_padded_videos": len(padded),
                "tail_padded_frames": int(sum(padded.values())),
                "max_tail_pad_frames": int(max(padded.values())) if padded else 0,
                "gt_frames": int(sum(tgt[v] for v in cohort))}
    log(f"{ds}: {len(cohort)}/{len(cohort)} cohort videos, finite on every GT frame; "
        f"{len(extra)} non-cohort ids dropped; tail pad {coverage['tail_padded_frames']} frames "
        f"in {coverage['tail_padded_videos']} videos (max {coverage['max_tail_pad_frames']})")
    return rows, coverage


def evaluate(corpus, rows, out_dir, log):
    """Write predictions.jsonl and run the canonical evaluator; check the frame pool is the whole cohort."""
    out_dir = Path(out_dir)
    pred = out_dir / "predictions.jsonl"
    with pred.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    metrics = out_dir / "metrics.json"
    cmd = [sys.executable, "-m", "src.eval.evaluate_four_datasets", "--predictions", str(pred),
           "--gt-dir", str(GT_DIR), "--out", str(metrics), "--datasets", DATASET[corpus]]
    log("evaluator: " + " ".join(cmd[1:]))
    subprocess.run(cmd, cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    result = json.loads(metrics.read_text())
    tgt = gt_lengths(corpus)
    want_frames = sum(tgt[v] for v in split_ids(corpus, "test"))
    for r in result["per_dataset"]:
        if r["n_videos_overlap"] != COHORT_SIZE[corpus] or r["n_videos_predicted"] != COHORT_SIZE[corpus]:
            raise RuntimeError(f"evaluator saw {r['n_videos_overlap']} videos for {r['method']}")
        if r.get("n_frames") is not None and r["n_frames"] != want_frames:
            raise RuntimeError(f"evaluator frame pool {r['n_frames']} != {want_frames} for {r['method']}")
        log(f"{r['method']} {r['dataset']}: ROC {r['frame_ROC_AUC']:.4f} PR {r['frame_PR_AUC']:.4f} "
            f"within {r['within_video_macro_ROC_AUC']:.4f} (videos {r['n_videos_overlap']}, frames {r.get('n_frames')})")
    return result
