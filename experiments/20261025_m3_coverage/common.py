"""Shared loaders and statistics for the M3 coverage analyses (CPU, analysis only).

Inputs (read-only):
  runs/20260926_twolevel/r6_bma/predictions.jsonl   r6_bma output: z_video, calibrated key (p_video_logodds), cell_prob,
                                                    score_curve = key + within-video centred rank of the M2 posterior
  runs/20260926_glr/base_gridA/predictions.jsonl    the SPVL-r2 window reads (window mean = mean of the windows' z)
  runs/20260927_dehate_external/{r6_bma,reads_gridA}  the same two inputs for DeHate (external validation only)
  data/gt_4fps/<dataset>.npz                        protocol GT, test split (evaluation and analysis only)
  Retrieval-hate gold_segments.json                 HateClipSeg source segments, for the hateful-only diagnostic GT
                                                    (rebuilt exactly as experiments/20260927_dvd/hcs_strict.py)
GT is read only for evaluation and analysis; nothing here feeds a GT-derived number into a score, except the scripts
that are explicitly labelled as diagnostics (diag_gt_extent.py).
"""
from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
R6 = ROOT / "runs/20260926_twolevel/r6_bma"
READS = ROOT / "runs/20260926_glr/base_gridA"
GT_DIR = ROOT / "data/gt_4fps"
HCS_SOURCE = Path("/home/jehc223/Retrieval-hate/data/gt/HateClipSeg/gold_segments.json")
OUT_ROOT = ROOT / "runs/20261025_m3_coverage"
DATASETS = ("HateMM", "HateClipSeg")            # main corpora (the gates)
EXTERNAL = ("DeHate",)                          # external validation, numbers only
CORPORA = {ds: {"r6": R6, "reads": READS} for ds in DATASETS}
CORPORA["DeHate"] = {"r6": ROOT / "runs/20260927_dehate_external/r6_bma",
                     "reads": ROOT / "runs/20260927_dehate_external/reads_gridA"}


# ----------------------------------------------------------------------------------------------- data

def load_gt(ds):
    g = np.load(GT_DIR / f"{ds}.npz", allow_pickle=True)
    return {str(v): np.asarray(g["y4"][i], np.int8) for i, v in enumerate(g["video_ids"]) if str(g["split"][i]) == "test"}


def _raster(spans, n):
    t = np.arange(n) / 4.0
    y = np.zeros(n, np.int8)
    for a, b in spans:
        y[(t >= a) & (t < b)] = 1
    return y


def load_hcs_strict(Y):
    """Diagnostic GT (not the protocol): HateClipSeg frames positive only where the hateful dimension is set.
    Checks first that the 'any non-normal dimension' rebuild equals the protocol GT on every video."""
    src = json.loads(HCS_SOURCE.read_text())
    off = {v: _raster([(s, e) for s, e, d in src[v]["segments"] if any(d[1:])], len(y)) for v, y in Y.items()}
    bad = [v for v in Y if not np.array_equal(off[v], Y[v])]
    if bad:
        raise SystemExit(f"HCS strict GT: offensive rebuild differs from the protocol GT on {len(bad)} videos")
    return {v: _raster([(s, e) for s, e, d in src[v]["segments"] if d[1]], len(y)) for v, y in Y.items()}


def gt_sets(datasets=DATASETS):
    """{name: (dataset, {vid: y})}: protocol GT per corpus and, with HateClipSeg, its hateful-only diagnostic."""
    out = {}
    for ds in datasets:
        out[ds] = (ds, load_gt(ds))
    if "HateClipSeg" in datasets:
        out["HateClipSeg_strict"] = ("HateClipSeg", load_hcs_strict(out["HateClipSeg"][1]))
    return out


def _sources(kind, path, datasets):
    """(predictions file, dataset filter) pairs: one given run dir for all datasets, or each corpus's own run."""
    if path is not None:
        return [(Path(path) / "predictions.jsonl", set(datasets))]
    by = {}
    for ds in datasets:
        by.setdefault(CORPORA[ds][kind], set()).add(ds)
    return [(Path(p) / "predictions.jsonl", d) for p, d in by.items()]


def _records(kind, path, datasets):
    for f, keep in _sources(kind, path, datasets):
        for line in open(f):
            r = json.loads(line)
            if not r.get("error") and r["dataset"] in keep:
                yield r


def load_r6(path=None, datasets=DATASETS):
    """Per (dataset, video): z_video, calibrated key, the r6 frame score and the within-video rank term."""
    out = {}
    for r in _records("r6", path, datasets):
        s = np.asarray(r["score_curve"], float); lo = float(r["extra"]["p_video_logodds"])
        out[(r["dataset"], r["video_id"])] = {"zv": float(r["extra"]["z_video"]), "lo": lo, "score": s,
                                              "rank": s - lo, "cell_prob": np.asarray(r["extra"]["cell_prob"], float)}
    return out


def load_reads(path=None, datasets=DATASETS):
    """Per (dataset, video): the window mean (mean of the windows' z) and z_video of the r6 reads."""
    out = {}
    for r in _records("reads", path, datasets):
        z = [float(w["z"]) for w in r["extra"]["windows"]]
        out[(r["dataset"], r["video_id"])] = {"wm": float(np.mean(z)) if z else 0.0, "zv": float(r["extra"]["z_video"])}
    return out


def load_values(path):
    """jsonl with dataset, video_id, value; the last record per (dataset, video_id) wins."""
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        out[(r["dataset"], r["video_id"])] = float(r["value"])
    return out


def write_values(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for (ds, v), x in sorted(values.items()):
            fh.write(json.dumps({"dataset": ds, "video_id": v, "value": float(x)}) + "\n")


def coverage(y, L):
    """Share of GT-positive frames over the frames the evaluator scores (first min(len(y), L) frames)."""
    n = min(len(y), L)
    return float(np.asarray(y[:n]).mean()) if n else 0.0


# ----------------------------------------------------------------------------------------------- statistics

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def partial_spearman(x, y, controls):
    """Spearman partial correlation of x and y given the controls: Pearson correlation of the residuals of rank(x) and
    rank(y) after a least-squares fit on the ranks of the controls (with intercept). 0 when x is explained exactly."""
    x, y = rankdata(np.asarray(x, float)), rankdata(np.asarray(y, float))
    Z = np.column_stack([np.ones(len(x))] + [rankdata(np.asarray(c, float)) for c in controls])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    if rx.std() < 1e-9 * max(1.0, x.std()) or ry.std() < 1e-9 * max(1.0, y.std()):
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def video_auc(score, lab):
    from sklearn.metrics import roc_auc_score
    lab = np.asarray(lab, int)
    if lab.min() == lab.max():
        return float("nan")
    return float(roc_auc_score(lab, np.asarray(score, float)))


def ci(a, q=(0.025, 0.975)):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(x) for x in np.quantile(a, q)] if len(a) else [float("nan"), float("nan")]


# ----------------------------------------------------------------------------------------------- run records

# Readable code-version note for run records (no commit identifiers: CLAUDE.md hash rule).
IMPL_TAG = "m3cov-2026-10-25a"

def start_run(out, script, config):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out / "run.pid").write_text(str(__import__("os").getpid()))
    (out / "config.json").write_text(json.dumps({**config, "script": script, "date": time.strftime("%Y-%m-%d"),
                                                  "code": f"{script}; candidate 47 implementation {IMPL_TAG}"},
                                                 indent=1, default=str))
    logf = open(out / "run.log", "w")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()
    log(f"host {socket.gethostname()}")
    log(f"code {script}; candidate 47 implementation {IMPL_TAG}")
    return log
