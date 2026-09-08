#!/usr/bin/env python3
"""Preregistered 2-D generic-harm conditioned mixture probe."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "our_method"))
from data_utils import load_annotations, load_clean_split_ids  # noqa: E402
from generic_harm_analyze import CFG, macro_f1, auc, read_map  # noqa: E402

SEED = 20260808
FLOOR = 1e-3


def robust(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    return (x - med) / np.maximum(1.4826 * mad, 1e-8)


def logpdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    sign, ld = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("non-positive covariance")
    d = x - mu
    return -0.5 * (x.shape[1] * np.log(2 * np.pi) + ld + np.einsum("ni,ij,nj->n", d, np.linalg.inv(cov), d))


def fit(x: np.ndarray, diagonal: bool = False) -> dict:
    n, d = x.shape
    q1, q3 = np.quantile(x[:, 0], [0.25, 0.75])
    groups = [x[:, 0] <= q1, x[:, 0] >= q3]
    mu = np.stack([np.mean(x[g], axis=0) for g in groups])
    cov = np.stack([np.atleast_2d(np.cov(x[g], rowvar=False)) + FLOOR * np.eye(d) for g in groups])
    if diagonal:
        cov = np.stack([np.diag(np.diag(c)) for c in cov])
    w = np.array([0.5, 0.5])
    prev = -np.inf
    for _ in range(500):
        lp = np.stack([np.log(w[k] + 1e-15) + logpdf(x, mu[k], cov[k]) for k in range(2)], axis=1)
        ll = float(np.sum(logsumexp(lp, axis=1)))
        resp = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))
        nk = np.sum(resp, axis=0)
        w = nk / n
        mu = (resp.T @ x) / nk[:, None]
        for k in range(2):
            dx = x - mu[k]
            c = (dx.T * resp[:, k]) @ dx / nk[k] + FLOOR * np.eye(d)
            cov[k] = np.diag(np.diag(c)) if diagonal else c
        if abs(ll - prev) < 1e-8:
            break
        prev = ll
    pos = int(np.argmax(mu[:, 0]))
    return {"posterior": resp[:, pos], "weights": w, "means": mu, "cov": cov, "ll": prev}


def main() -> None:
    os.environ.setdefault("HVD_DATA_ROOT", "/home/jehc223/data")
    reports = {}
    for ds, cfg in CFG.items():
        zmap, gmap = read_map(cfg["scores"], "z"), read_map(cfg["features"], "g_max")
        ann, clean = load_annotations(ds), set(load_clean_split_ids(ds, "train"))
        ids = sorted(clean & set(zmap) & set(gmap) & set(ann))
        z = np.asarray([zmap[v] for v in ids]); g = np.asarray([gmap[v] for v in ids])
        y = np.asarray([ann[v]["label"] in cfg["positive"] for v in ids], int)
        gl = np.log(np.clip(g, 1e-6, 1 - 1e-6) / np.clip(1 - g, 1e-6, 1))
        x = robust(np.c_[z, gl])
        full, diag = fit(x), fit(x, diagonal=True)
        one = fit(robust(z[:, None]))
        rng = np.random.default_rng(SEED)
        shuf = fit(robust(np.c_[z, gl[rng.permutation(len(gl))]]))
        arms = {}
        for name, obj in (("full", full), ("diag", diag), ("z_only", one), ("shuffled", shuf)):
            p = obj["posterior"] >= 0.5
            arms[name] = {"auc": auc(y, obj["posterior"]), "macro_f1": macro_f1(y, p),
                          "positive_rate": float(np.mean(p)), "weights": obj["weights"].tolist(),
                          "means": obj["means"].tolist(), "cov": obj["cov"].tolist()}
        baseline = 0.6428763065534366 if ds == "HateMM" else 0.8796001091985122
        raw_auc = auc(y, z)
        reports[ds] = {"n": len(ids), "raw_auc": raw_auc, "kde_macro_f1": baseline,
                       "arms": arms, "full_auc_gain": arms["full"]["auc"] - raw_auc,
                       "full_f1_gain": arms["full"]["macro_f1"] - baseline}

    h, i = reports["HateMM"], reports["ImpliHateVid"]
    hf, hi = h["arms"]["full"], i["arms"]["full"]
    clauses = {
        "hatemm_f1_gain_ge_0.10": h["full_f1_gain"] >= 0.10,
        "hatemm_auc_gain_ge_0.03": h["full_auc_gain"] >= 0.03,
        "ihv_noninferiority": i["full_f1_gain"] >= -0.02 and i["full_auc_gain"] >= -0.02,
        "interaction_load_bearing": (hf["macro_f1"] - h["arms"]["z_only"]["macro_f1"] >= 0.03 and
                                     hf["macro_f1"] - h["arms"]["diag"]["macro_f1"] >= 0.03),
        "shuffle_less_than_half_gain": (h["full_f1_gain"] > 0 and
                                         h["arms"]["shuffled"]["macro_f1"] - h["kde_macro_f1"] < 0.5 * h["full_f1_gain"]),
    }
    out = {"pilot": "generic_harm_2d", "reports": reports, "clauses": clauses,
           "verdict": "PASS" if all(clauses.values()) else "FAIL"}
    path = ROOT / "results/generic_harm_nuisance/results_2d.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
