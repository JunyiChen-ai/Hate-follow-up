#!/usr/bin/env python3
"""AnyAnomaly: the release's own grid search on the stored per-window scores (README §3.3.1, planning-session brief 5
item K4, 2026-11-15). CPU only, no new inference. Test-selected: every configuration here is chosen on test labels.

Inputs: runs/20261113_lf_baselines/anyanomaly/<DS>/u8/readout.jsonl, the `prob` read-out of the three context
images per 8 s window (key frame `key`, WinCLIP attention image `wa`, temporal grid `tc`).

Grid, as the release enumerates it (third_party/AnyAnomaly/functions/eval_func.py `evaluate_auc`, defaults of
vad_proposed_Qwen_vllm.py `--sigma_range 1,30 --weight_step 0.1`):
  for sigma in range(1, 30):                      # 1 .. 29 (the release's range excludes 30)
    for a1 in np.arange(0.0, 1.1, 0.1):
      for a2 in np.arange(0.0, 1.1, 0.1):
        a3 = 1 - a1 - a2; kept only if 0 <= a3 <= 1 (floating point as in the release)
score = a1 * key + a2 * wa + a3 * tc. sigma is in release frames; one release clip = 24 frames = one of our 8 s windows,
so sigma s is applied as s / 24 windows = s / 3 seconds = 4 s / 3 frames on the 4 fps curve, per video (as our port;
the release smooths the concatenated test set and min-max normalises it, which is monotone and changes no metric).

Rows written (evaluator metrics.json under runs/20261113_lf_baselines/anyanomaly/<DS>/u8/<name>/):
  k4_clipunit_default   alpha .6, beta .3, gamma .1, sigma 15 release frames (5 s); defined before its numbers were seen
  k4_grid_common        one grid configuration for all three corpora: most of the 9 cells >= the naive row, ties by
                        the smallest largest shortfall, then release enumeration order
  k4_grid_per_corpus    the release protocol: per corpus the configuration with the highest pooled ROC (first in the
                        release enumeration order on ties, as the release's strict `>`)
Metrics in the grid: the canonical evaluator functions (src/eval/evaluate.py) on the test split. The grid smooths each
of the three channels and then combines them (linear, the same curve up to floating-point rounding); within-video ROC of
AnyAnomaly is sensitive to rounding because many curves have plateaus of near-equal values (DeHate: .6023 in memory vs
.6019 from the stored predictions for the same configuration). Selection uses the grid values; the reported numbers
are the evaluator's metrics.json of the written rows.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm_common import L                                      # noqa: E402
from src.eval.evaluate import pooled, within_video_macro      # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = os.path.join(REPO, "runs/20261113_lf_baselines")
OUT = os.path.join(RUNS, "anyanomaly_k4")
UNIT = 8.0
DS = L.DATASETS


def weights():
    out = []
    for a1 in np.arange(0.0, 1.1, 0.1):
        for a2 in np.arange(0.0, 1.1, 0.1):
            a3 = 1 - a1 - a2
            if 0 <= a3 <= 1:
                out.append((float(a1), float(a2), float(a3)))
    return out


def gt(ds):
    z = np.load(os.path.join(REPO, f"data/gt_4fps/{ds}.npz"), allow_pickle=True)
    return {str(v): np.asarray(y, np.int8) for v, s, y in zip(z["video_ids"], z["split"], z["y4"]) if str(s) == "test"}


def load(ds):
    d = os.path.join(RUNS, "anyanomaly", ds, "u8")
    recs = {}
    for line in open(os.path.join(d, "units.jsonl")):
        r = json.loads(line)
        recs[r["video_id"]] = r
    prob = {}
    for line in open(os.path.join(d, "readout.jsonl")):
        r = json.loads(line)
        prob[r["video_id"]] = r["scores"]
    cohort = set(L.cohort(ds))
    base = {}
    for v, r in recs.items():
        if v not in cohort:
            continue
        us = r["units"]
        if r.get("error") or not us or v not in prob or len(prob[v]) != len(us):
            raise SystemExit(f"K4 input missing: {ds} {v} (error {r.get('error')}, units {len(us)}, "
                             f"scores {len(prob.get(v, []))})")
        n = L.n_frames(r["duration"])
        starts = [u["start"] for u in us]
        base[v] = (r["duration"], np.stack([L.units_to_4fps(starts, None, [s[k] for s in prob[v]], n)
                                           for k in ("key", "wa", "tc")]))
    return base


def curves(base, w, sigma):
    sf = sigma / 3.0 * L.RATE
    return {v: gaussian_filter1d(np.asarray(w, float) @ c, sigma=sf) for v, (_, c) in base.items()}


def score(y, s):
    p, wv = pooled(y, s), within_video_macro(y, s)
    return [p["frame_ROC_AUC"], p["frame_PR_AUC"], wv["within_video_macro_ROC_AUC"]]


def _sigma_job(args):
    ds, sigma = args
    part = os.path.join(OUT, "parts", f"{ds}_s{sigma:02d}.json")
    if os.path.exists(part):                  # resumable: one file per (corpus, sigma)
        return json.load(open(part))
    base, y = load(ds), gt(ds)
    sf = sigma / 3.0 * L.RATE
    sm = {v: np.stack([gaussian_filter1d(c[k], sigma=sf) for k in range(3)]) for v, (_, c) in base.items()}
    out = []
    for w in weights():
        s = {v: np.asarray(w) @ c for v, c in sm.items()}
        out.append({"dataset": ds, "sigma": sigma, "w": w, "m": score(y, s)})
    json.dump(out, open(part + ".tmp", "w"))
    os.replace(part + ".tmp", part)
    return out


def write_row(ds, name, w, sigma):
    base = load(ds)
    method = "anyanomaly_q3vl_" + name
    cur = curves(base, w, sigma)
    rows = {v: L.row(method, ds, v, base[v][0], cur[v], 1.0 / UNIT, __file__,
                     extra={"alpha": w[0], "beta": w[1], "gamma": w[2], "sigma_release_frames": sigma})
            for v in base}
    med, failed = L.apply_f2(rows, ds)
    assert not failed
    res = L.finalize(ds, list(rows.values()), os.path.join(RUNS, "anyanomaly", ds, "u8", name), method)
    r = res["per_dataset"][0]
    return [r["frame_ROC_AUC"], r["frame_PR_AUC"], r["within_video_macro_ROC_AUC"]]


def main():
    os.makedirs(os.path.join(OUT, "parts"), exist_ok=True)
    W = weights()
    print("weight triples", len(W), "sigmas 1..29, configurations per corpus", 29 * len(W), flush=True)
    # check: the stored prob configuration (alpha .6 / .3 / .1, sigma 0.6 s) reproduces its metrics.json
    for ds in DS:
        base, y = load(ds), gt(ds)
        s = {v: gaussian_filter1d(np.array([.6, .3, .1]) @ c, sigma=15 / 25.0 * L.RATE) for v, (_, c) in base.items()}
        ref = json.load(open(os.path.join(RUNS, "anyanomaly", ds, "u8", "prob", "metrics.json")))["per_dataset"][0]
        mr = [ref["frame_ROC_AUC"], ref["frame_PR_AUC"], ref["within_video_macro_ROC_AUC"]]
        print(ds, "stored prob check max |diff|", max(abs(a - b) for a, b in zip(score(y, s), mr)), flush=True)
    jobs = [(ds, s) for ds in DS for s in range(1, 30)]
    grid = []
    with ProcessPoolExecutor(int(os.environ.get("K4_JOBS", "12"))) as ex:
        for part in ex.map(_sigma_job, jobs):
            grid += part
            print("done", part[0]["dataset"], "sigma", part[0]["sigma"], flush=True)
    order = {(s, w): i for i, (s, w) in enumerate((s, w) for s in range(1, 30) for w in W)}
    grid.sort(key=lambda g: (DS.index(g["dataset"]), order[(g["sigma"], tuple(g["w"]))]))
    with open(os.path.join(OUT, "grid.jsonl"), "w") as fh:
        for g in grid:
            fh.write(json.dumps(g) + "\n")
    naive = {ds: json.load(open(os.path.join(RUNS, "naive/vlm", ds, "metrics.json")))["per_dataset"][0] for ds in DS}
    naive = {ds: [r["frame_ROC_AUC"], r["frame_PR_AUC"], r["within_video_macro_ROC_AUC"]] for ds, r in naive.items()}
    by = {(g["sigma"], tuple(g["w"])): {} for g in grid}
    for g in grid:
        by[(g["sigma"], tuple(g["w"]))][g["dataset"]] = g["m"]
    common = []
    for key, mm in by.items():
        d = [100 * (mm[ds][k] - naive[ds][k]) for ds in DS for k in range(3)]
        common.append((-sum(x >= 0 for x in d), max([-x for x in d if x < 0], default=0.0), order[key], key))
    common.sort()
    c_key = common[0][3]
    per = {}
    for ds in DS:
        best = None
        for g in grid:                         # release order, strict >
            if g["dataset"] == ds and (best is None or g["m"][0] > best["m"][0]):
                best = g
        per[ds] = (best["sigma"], tuple(best["w"]))
    sel = {"k4_clipunit_default": {ds: (15, (0.6, 0.3, 0.1)) for ds in DS},
           "k4_grid_common": {ds: c_key for ds in DS},
           "k4_grid_per_corpus": per}
    written = {}
    for name, cfg in sel.items():
        written[name] = {}
        for ds in DS:
            sigma, w = cfg[ds]
            written[name][ds] = {"sigma": sigma, "w": list(w), "m": write_row(ds, name, w, sigma)}
    summ = {"n_weight_triples": len(W), "naive": naive, "selected": written,
            "grid_common_rank": [{"cells": -c[0], "shortfall": c[1], "sigma": c[3][0], "w": list(c[3][1])}
                                 for c in common[:10]]}
    for ds in DS:
        g = [x for x in grid if x["dataset"] == ds]
        summ.setdefault("range", {})[ds] = {k: [min(x["m"][i] for x in g), max(x["m"][i] for x in g)]
                                            for i, k in enumerate(("roc", "pr", "within"))}
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print("DONE K4")


if __name__ == "__main__":
    main()
