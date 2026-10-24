"""Helpers for the M3 composition exploration (explore_*.py; CPU; test error analysis under rule 10).

Re-derived here (not imported from experiments/20260926_twolevel, which must not be imported across experiment
directories), each checked against the archived r6_bma output before use:
  frames(cell_vals, L)   cell -> 4 fps frame mapping of twolevel.to_frames (frame centre // 4 s cell)
  quantile_rank(v)       (average rank - .5) / L; r6's centred rank term equals this minus .5
  key_mixture(x)         the label-free two-Gaussian calibration of twolevel_r2.key_calibration (same EM, same start),
                         returning the mixture as well as (a, b)
  nscore(ys)             twolevel_r2's --transform nscore (normal scores of the corpus ranks)
Scoring goes through src/eval/evaluate_four_datasets.py (metrics.json) and src/eval/evaluate.pooled /
within_video_macro (paired video bootstrap); nothing here evaluates on its own.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import norm, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, ci  # noqa: E402

sys.path.insert(0, str(ROOT))
from src.eval.evaluate import pooled, within_video_macro  # noqa: E402

FPS, CELL, EPS = 4.0, 4.0, 1e-6
METRICS = ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC")
NOISE_POOLED, NOISE_WITHIN, ABL_DROP = 0.005, 0.01, 0.01


def frames(cell_vals, L):
    centers = (np.arange(L) + 0.5) / FPS
    idx = np.clip((centers // CELL).astype(int), 0, len(cell_vals) - 1)
    return np.asarray(cell_vals, float)[idx]


def quantile_rank(v):
    v = np.asarray(v, float)
    if len(v) <= 1 or np.ptp(v) <= 1e-12:
        return np.full(len(v), 0.5)
    return (rankdata(v, method="average") - 0.5) / len(v)


def key_mixture(keys, it=2000):
    x = np.asarray(keys, float)
    mu = np.percentile(x, [10, 90]).astype(float); var = float(x.var()); pi = 0.5
    for _ in range(it):
        l0 = math.log(1 - pi) - 0.5 * (x - mu[0]) ** 2 / var; l1 = math.log(pi) - 0.5 * (x - mu[1]) ** 2 / var
        r = expit(l1 - l0); pi = float(np.clip(r.mean(), EPS, 1 - EPS))
        mu = np.array([((1 - r) * x).sum() / (1 - r).sum(), (r * x).sum() / r.sum()])
        var = float(((1 - r) * (x - mu[0]) ** 2 + r * (x - mu[1]) ** 2).mean())
    if mu[1] < mu[0]:
        mu = mu[::-1]; pi = 1 - pi
    a = (mu[1] - mu[0]) / var
    b = math.log(pi) - math.log(1 - pi) - (mu[1] ** 2 - mu[0] ** 2) / (2 * var)
    return {"a": float(a), "b": float(b), "mu0": float(mu[0]), "mu1": float(mu[1]), "sd": math.sqrt(var), "pi": pi,
            "dprime": float((mu[1] - mu[0]) / math.sqrt(var))}


def nscore(ys):
    ys = np.asarray(ys, float)
    return norm.ppf((rankdata(ys) - 0.5) / len(ys))


def evaluate_scores(name, scores, datasets, out):
    """scores {(ds, vid): frame array} -> predictions.jsonl + metrics.json by the repository evaluator."""
    d = Path(out) / "arms" / name; d.mkdir(parents=True, exist_ok=True)
    with open(d / "predictions.jsonl", "w") as fh:
        for (ds, v), s in sorted(scores.items()):
            fh.write(json.dumps({"schema_version": 1, "method": f"explore__{name}", "dataset": ds, "video_id": v,
                                 "score_curve": [float(x) for x in s], "intervals": []}) + "\n")
    subprocess.run([sys.executable, "-m", "src.eval.evaluate_four_datasets", "--predictions", str(d / "predictions.jsonl"),
                    "--gt-dir", str(ROOT / "data/gt_4fps"), "--out", str(d / "metrics.json"), "--datasets", *datasets],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return {p["dataset"]: {m: p[m] for m in METRICS + ("n_videos_defined",)}
            for p in json.load(open(d / "metrics.json"))["per_dataset"]}


def point(Y, S):
    p = pooled(Y, S); w = within_video_macro(Y, S)
    return [p.get("frame_ROC_AUC"), p.get("frame_PR_AUC"), w["within_video_macro_ROC_AUC"]]


def paired_bootstrap(arms, base, gts, B, seed, boot_arms=None):
    """arms {name: {(ds, vid): frames}}, base the same for r6_bma. Per GT set: point metrics of every arm (pooled +
    within) and, for boot_arms, the paired video bootstrap of pooled ROC / PR differences against the base."""
    rng = np.random.default_rng(seed)
    boot_arms = list(arms) if boot_arms is None else boot_arms
    res = {}
    for gname, (ds, Y) in gts.items():
        vids = sorted(v for (d, v) in base if d == ds and v in Y)
        S = {a: {v: s[(ds, v)] for v in vids} for a, s in arms.items()}
        S0 = {v: base[(ds, v)] for v in vids}
        p0 = point(Y, S0)
        out = {"n_videos": len(vids), "r6_bma": p0, "arms": {}}
        for a in arms:
            pa = point(Y, S[a])
            out["arms"][a] = {"point": pa, "delta": [None if x is None or y is None else x - y for x, y in zip(pa, p0)]}
        diffs = {a: [] for a in boot_arms}
        for _ in range(B if boot_arms else 0):
            idx = rng.integers(0, len(vids), len(vids))
            yb = {f"{vids[i]}#{j}": Y[vids[i]] for j, i in enumerate(idx)}
            q0 = pooled(yb, {f"{vids[i]}#{j}": S0[vids[i]] for j, i in enumerate(idx)})
            for a in boot_arms:
                q = pooled(yb, {f"{vids[i]}#{j}": S[a][vids[i]] for j, i in enumerate(idx)})
                diffs[a].append((q["frame_ROC_AUC"] - q0["frame_ROC_AUC"], q["frame_PR_AUC"] - q0["frame_PR_AUC"]))
        for a in boot_arms:
            dd = np.array(diffs[a], float).reshape(-1, 2)
            out["arms"][a]["ROC_ci95"] = ci(dd[:, 0]); out["arms"][a]["PR_ci95"] = ci(dd[:, 1])
        res[gname] = out
    return res


def g1(res, main, arm):
    """No metric of the arm below r6_bma by more than the noise floor on the main corpora."""
    ok = True
    for ds in main:
        d = res[ds]["arms"][arm]["delta"]
        ok &= d[0] >= -NOISE_POOLED and d[1] >= -NOISE_POOLED and d[2] >= -NOISE_WITHIN
    return bool(ok)


def ablation_drop(res, main, arm, abl):
    """Revised G2: removing the component (arm -> abl) drops at least one main metric by >= .01 on both corpora."""
    out = {}
    for j, m in enumerate(METRICS):
        out[m] = all(res[ds]["arms"][arm]["point"][j] - res[ds]["arms"][abl]["point"][j] >= ABL_DROP for ds in main)
    return {"any": any(out.values()), **out}


def fmt(x, sign=False, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
