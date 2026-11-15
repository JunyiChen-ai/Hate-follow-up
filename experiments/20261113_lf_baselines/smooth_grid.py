#!/usr/bin/env python3
"""Gaussian temporal smoothing of stored baseline scores (README §3.5, 2026-11-16). CPU only, no new inference.

The smoothing step is our addition to the ports (not an option of the MARS, EventVAD, CoReVAD or Cog-VADU releases);
every configuration is chosen on test labels and is labelled so. Grid and selection rule: README §3.5, written before
this script was run.

Input: the stored 4 fps curves (predictions.jsonl) of the rows in SOURCES. Per video
`gaussian_filter1d(curve, sigma = 4 * sigma_s)`; score space `raw` (stored score) or, for log-odds rows, `prob`
(logistic of the stored score, then the filter).

Rows written (evaluator metrics.json under runs/20261113_lf_baselines/smooth/<method>/<DS>/<name>/):
  s5_raw   sigma 5 s on the stored score (fixed before its numbers were seen)
  sel      one configuration for all three corpora; rule: add the row to gate set A (gate.py rows without any
           smoothing row); most cells >= naive for the enlarged set; ties by the smallest largest shortfall of the
           enlarged set; then by the row's own number of cells >= naive; then grid order
The naive row (`naive_vlm`) is a reference: its grid and `s5_raw` are written, it gets no `sel` row and is not a
gate row. Metrics in the grid: the canonical evaluator functions on the test split.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vllm_common import L                                      # noqa: E402
from src.eval.evaluate import pooled, within_video_macro      # noqa: E402
import gate as G                                               # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = os.path.join(REPO, "runs/20261113_lf_baselines")
OUT = os.path.join(RUNS, "smooth")
DS = L.DATASETS
SIGMAS = (2, 4, 5, 8, 12, 16, 24, 32)
# method -> (stored row directory, stored score is a log-odds, reference only)
SOURCES = {
    "mars": ("mars/{ds}/prob", True, False),
    "eventvad": ("eventvad/{ds}/prob", False, False),
    "corevad": ("corevad/{ds}/u8/prob_lrc1_tau0.05_g9-5_pos1", False, False),
    "cogvadu": ("cogvadu/{ds}/u8/prob_default", False, False),
    "naive_vlm": ("naive/vlm/{ds}", True, True),
}
SMOOTH_TAG = "Gaussian smoothing"       # gate.py labels of the rows written here contain this text


def configs(method):
    spaces = ("raw", "prob") if SOURCES[method][1] else ("raw",)
    return [(s, sp) for s in SIGMAS for sp in spaces]


def gt(ds):
    z = np.load(os.path.join(REPO, f"data/gt_4fps/{ds}.npz"), allow_pickle=True)
    return {str(v): np.asarray(y, np.int8) for v, s, y in zip(z["video_ids"], z["split"], z["y4"]) if str(s) == "test"}


def load(method, ds):
    path = os.path.join(RUNS, SOURCES[method][0].format(ds=ds), "predictions.jsonl")
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("error"):
            raise SystemExit(f"stored row with an error: {method} {ds} {r['video_id']}")
        rows[r["video_id"]] = r
    if len(rows) != L.EXPECTED[ds]:
        raise SystemExit(f"{method} {ds}: {len(rows)} videos, expected {L.EXPECTED[ds]}")
    return rows


def smooth(curve, sigma, space):
    c = np.asarray(curve, float)
    if space == "prob":
        c = expit(c)
    return gaussian_filter1d(c, sigma=sigma * L.RATE)


def score(y, s):
    p, wv = pooled(y, s), within_video_macro(y, s)
    return [p["frame_ROC_AUC"], p["frame_PR_AUC"], wv["within_video_macro_ROC_AUC"]]


def stored_metrics(method, ds):
    ref = json.load(open(os.path.join(RUNS, SOURCES[method][0].format(ds=ds), "metrics.json")))["per_dataset"][0]
    return [ref["frame_ROC_AUC"], ref["frame_PR_AUC"], ref["within_video_macro_ROC_AUC"]]


def _job(args):
    method, ds = args
    rows, y = load(method, ds), gt(ds)
    raw = score(y, {v: np.asarray(r["score_curve"], float) for v, r in rows.items()})
    diff = max(abs(a - b) for a, b in zip(raw, stored_metrics(method, ds)))
    if diff > 1e-9:
        raise SystemExit(f"{method} {ds}: unsmoothed curves do not reproduce the stored metrics (max diff {diff})")
    out = []
    for sigma, space in configs(method):
        s = {v: smooth(r["score_curve"], sigma, space) for v, r in rows.items()}
        out.append({"method": method, "dataset": ds, "sigma_s": sigma, "space": space, "m": score(y, s)})
    return out


def write_row(method, ds, name, sigma, space):
    rows = load(method, ds)
    tag = f"{method}_q3vl_smooth_{name}"
    new = {v: L.row(tag, ds, v, r["duration"], smooth(r["score_curve"], sigma, space), r.get("native_rate"), __file__,
                    extra={"source": SOURCES[method][0].format(ds=ds), "sigma_s": sigma, "space": space,
                           "note": "Gaussian smoothing added by the port; not a release option"})
           for v, r in rows.items()}
    res = L.finalize(ds, list(new.values()), os.path.join(OUT, method, ds, name), tag)
    r = res["per_dataset"][0]
    return [r["frame_ROC_AUC"], r["frame_PR_AUC"], r["within_video_macro_ROC_AUC"]]


def main():
    os.makedirs(OUT, exist_ok=True)
    L.start_run_log(OUT)
    print("code", L.git_version(), flush=True)
    jobs = [(m, ds) for m in SOURCES for ds in DS]
    grid = []
    with ProcessPoolExecutor(int(os.environ.get("SMOOTH_JOBS", "8"))) as ex:
        for part in ex.map(_job, jobs):
            grid += part
            print("done", part[0]["method"], part[0]["dataset"], "(unsmoothed check passed)", flush=True)
    with open(os.path.join(OUT, "grid.jsonl"), "w") as fh:
        for g in grid:
            fh.write(json.dumps(g) + "\n")
    naive = {ds: stored_metrics("naive_vlm", ds) for ds in DS}
    cells = [(ds, k) for ds in DS for k in range(3)]
    set_a = [r for r in G.rows() if SMOOTH_TAG not in r[0]]
    best_a = {c: max(r[1][c[0]][c[1]] for r in set_a) for c in cells}
    by = {}
    for g in grid:
        by.setdefault((g["method"], g["sigma_s"], g["space"]), {})[g["dataset"]] = g["m"]
    summ = {"naive": naive, "set_A_rows": len(set_a), "set_A_best": {f"{d} {k}": best_a[(d, k)] for d, k in cells},
            "methods": {}}
    for method, (_, _, ref_only) in SOURCES.items():
        table, rank = [], []
        for i, (sigma, space) in enumerate(configs(method)):
            mm = by[(method, sigma, space)]
            own = [100 * (mm[d][k] - naive[d][k]) for d, k in cells]
            enl = [100 * (max(best_a[(d, k)], mm[d][k]) - naive[d][k]) for d, k in cells]
            n_enl, short_enl = sum(x >= 0 for x in enl), max([-x for x in enl if x < 0], default=0.0)
            n_own = sum(x >= 0 for x in own)
            table.append({"sigma_s": sigma, "space": space, "m": mm, "own_cells": n_own,
                          "enlarged_cells": n_enl, "enlarged_shortfall": short_enl})
            rank.append((-n_enl, short_enl, -n_own, i, (sigma, space)))
        rank.sort()
        entry = {"grid": table, "written": {}}
        todo = {"s5_raw": (5, "raw")}
        if not ref_only:
            todo["sel"] = rank[0][4]
        for name, (sigma, space) in todo.items():
            entry["written"][name] = {"sigma_s": sigma, "space": space,
                                      "m": {ds: write_row(method, ds, name, sigma, space) for ds in DS}}
        summ["methods"][method] = entry
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    f = lambda v: " / ".join("%.2f" % (100 * x) for x in v)   # noqa: E731
    print("naive", " | ".join(f(naive[ds]) for ds in DS))
    for method, e in summ["methods"].items():
        print("==", method)
        for t in e["grid"]:
            print("  sigma %2d %-4s | %s | own %d, set A + row %d of 9, shortfall %.2f" % (
                t["sigma_s"], t["space"], " | ".join(f(t["m"][ds]) for ds in DS), t["own_cells"],
                t["enlarged_cells"], t["enlarged_shortfall"]))
        for name, w in e["written"].items():
            print("  written %s: sigma %s %s | %s" % (name, w["sigma_s"], w["space"], " | ".join(f(w["m"][ds]) for ds in DS)))
    print("DONE smooth_grid")


if __name__ == "__main__":
    main()
