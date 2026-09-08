#!/usr/bin/env python3
"""Analyze the preregistered generic-harm nuisance signal."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde, rankdata


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "our_method"))
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402

N_GRID = 4001
GRID_PAD = 2.0
N_BOOT = 2000
SEED = 20260808

CFG = {
    "HateMM": {
        "scores": ROOT / "results/crossbench/hatemm/judge_8b/scores.jsonl",
        "features": ROOT / "results/generic_harm_nuisance/HateMM_train_scaled.jsonl",
        "positive": {"Hate"},
    },
    "ImpliHateVid": {
        "scores": ROOT / "results/c2_fullcorpus/judge_8b/scores.jsonl",
        "features": ROOT / "results/generic_harm_nuisance/ImpliHateVid_train_scaled.jsonl",
        "positive": {"Hateful"},
    },
}


def read_map(path: Path, key: str) -> dict[str, float]:
    out = {}
    for line in path.open():
        r = json.loads(line)
        if isinstance(r.get(key), (int, float)):
            out[r["video_id"]] = float(r[key])
    return out


def auc(y: np.ndarray, s: np.ndarray) -> float:
    n1, n0 = int(np.sum(y == 1)), int(np.sum(y == 0))
    if not n1 or not n0:
        return float("nan")
    ranks = rankdata(s, method="average")
    return float((np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    vals = []
    for c in (0, 1):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        vals.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(vals))


def kde_valley(z: np.ndarray) -> float | None:
    kde = gaussian_kde(z, bw_method="scott")
    grid = np.linspace(np.min(z) - GRID_PAD, np.max(z) + GRID_PAD, N_GRID)
    d = kde(grid)
    loc = [i for i in range(1, N_GRID - 1) if d[i] > d[i - 1] and d[i] > d[i + 1]]
    if len(loc) < 2:
        return None
    a, b = sorted(sorted(loc, key=lambda i: -d[i])[:2])
    j = a + 1 + int(np.argmin(d[a + 1:b]))
    return float(grid[j])


def residual(z: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    med = np.median(g)
    mad = np.median(np.abs(g - med))
    gs = (g - med) / max(1.4826 * mad, 1e-8)
    beta = float(np.sum((gs - np.mean(gs)) * (z - np.mean(z))) /
                 max(np.sum((gs - np.mean(gs)) ** 2), 1e-12))
    return z - max(beta, 0.0) * gs, beta, gs


def bootstrap_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray, fn, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        vals.append(fn(y[idx], b[idx]) - fn(y[idx], a[idx]))
    return [float(x) for x in np.quantile(vals, [0.025, 0.975])]


def main() -> None:
    os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")
    reports = {}
    arrays = {}
    for di, (ds, cfg) in enumerate(CFG.items()):
        zmap, gmap = read_map(cfg["scores"], "z"), read_map(cfg["features"], "g_max")
        ann = load_annotations(ds)
        clean = set(load_clean_split_ids(ds, "train"))
        ids = sorted(clean & set(zmap) & set(gmap) & set(ann))
        z = np.asarray([zmap[v] for v in ids])
        g = np.asarray([gmap[v] for v in ids])
        y = np.asarray([ann[v]["label"] in cfg["positive"] for v in ids], dtype=int)
        r, beta, gs = residual(z, g)
        vz, vr = kde_valley(z), kde_valley(r)
        if vz is None or vr is None:
            raise SystemExit(f"{ds}: missing KDE valley z={vz} r={vr}")
        pz, pr = z >= vz, r >= vr

        rng = np.random.default_rng(SEED)
        shuffled = g[rng.permutation(len(g))]
        rsh, beta_sh, _ = residual(z, shuffled)
        auc_z, auc_r, auc_sh = auc(y, z), auc(y, r), auc(y, rsh)
        f1_z, f1_r = macro_f1(y, pz), macro_f1(y, pr)
        fp, tp = (~y.astype(bool) & pz), (y.astype(bool) & pz)
        nuisance_fp_vs_tp = auc(np.r_[np.ones(np.sum(fp)), np.zeros(np.sum(tp))],
                                np.r_[g[fp], g[tp]])
        reports[ds] = {
            "n": len(ids), "prevalence": float(np.mean(y)), "beta": beta,
            "beta_shuffled": beta_sh, "valley_z": vz, "valley_r": vr,
            "auc_z": auc_z, "auc_r": auc_r, "auc_gain": auc_r - auc_z,
            "auc_shuffled": auc_sh, "auc_shuffled_gain": auc_sh - auc_z,
            "macro_f1_z": f1_z, "macro_f1_r": f1_r, "macro_f1_gain": f1_r - f1_z,
            "nuisance_auc_fp_over_tp": nuisance_fp_vs_tp,
            "original_confusion": {"fp": int(np.sum(fp)), "tp": int(np.sum(tp)),
                                   "fn": int(np.sum(y.astype(bool) & ~pz)),
                                   "tn": int(np.sum(~y.astype(bool) & ~pz))},
            "fp_flip_down_rate": float(np.mean(~pr[fp])) if np.any(fp) else None,
            "tp_flip_down_rate": float(np.mean(~pr[tp])) if np.any(tp) else None,
            "auc_gain_bootstrap_95": bootstrap_diff(y, z, r, auc, SEED + 10 * di),
            "macro_f1_gain_bootstrap_95": bootstrap_diff(
                y, pz.astype(float), pr.astype(float),
                lambda yy, pp: macro_f1(yy, pp >= 0.5), SEED + 10 * di + 1),
        }
        arrays[ds] = (y, z, r, pz, pr)

    h, i = reports["HateMM"], reports["ImpliHateVid"]
    clauses = {
        "hatemm_g_auc_fp_over_tp_ge_0.65": h["nuisance_auc_fp_over_tp"] >= 0.65,
        "hatemm_auc_gain_ge_0.03": h["auc_gain"] >= 0.03,
        "hatemm_f1_gain_ge_0.10": h["macro_f1_gain"] >= 0.10,
        "hatemm_flip_signature": h["fp_flip_down_rate"] >= 0.25 and h["tp_flip_down_rate"] <= 0.10,
        "ihv_noninferiority": i["auc_gain"] >= -0.02 and i["macro_f1_gain"] >= -0.02,
        "shuffle_less_than_half_real_auc_gain": (
            h["auc_gain"] > 0 and h["auc_shuffled_gain"] < 0.5 * h["auc_gain"]),
    }
    out = {"pilot": "generic_harm_nuisance", "reports": reports,
           "clauses": clauses, "verdict": "PASS" if all(clauses.values()) else "FAIL"}
    path = ROOT / "results/generic_harm_nuisance/results.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
