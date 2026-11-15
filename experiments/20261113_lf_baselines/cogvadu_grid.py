#!/usr/bin/env python3
"""Cog-VADU: grid over the release's re-ranking arguments on the stored chain, read-out and ImageBind embeddings
(README §3.4.1, planning-session brief 5 item H5). CPU only, no new inference. Test-selected: the selected
configuration is chosen on test labels.

Inputs: runs/20261113_lf_baselines/cogvadu/<DS>/u8/{units.jsonl, readout.jsonl, imagebind/}, read-out `prob`.

Grid (README §3.4.1), enumerated in this order (the order breaks ties):
  eval num_neighbors   2 .. 10
  kept num_neighbors   2, 5, 10, 20, 40
  initial_k            8, 15, 30, 60, 120
  text_weight          0.125, 0.25, 0.5, 1, 2
  visual_weight        0.125, 0.25, 0.5, 1, 2
  temporal_window      1, 2, 5, 10, 20, 0
Only the ratio text_weight / visual_weight changes the ranking, so weight pairs with the same ratio share one
computation (9 distinct ratios out of 25 pairs); every pair still appears in the grid with its own position.

Per-segment arithmetic is `refine_video` of cogvadu_q3vl.py, split so that the retrieval (stage 5) runs once per
(kept, initial_k, ratio, temporal_window) and the averaging (stage 6) once per eval num_neighbors. A check at start
compares this split with `refine_video` on the default configuration.

Selection: the K1 rule. One configuration for all three corpora; most of the 9 cells >= the naive `vlm` row; ties by
the smallest largest shortfall, then grid order. Metrics in the grid: the canonical evaluator functions
(src/eval/evaluate.py) on the test split. The selected row is written with the evaluator (metrics.json under
runs/20261113_lf_baselines/cogvadu/<DS>/u8/grid_selected/); the default configuration is u8/prob_default.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cogvadu_q3vl as C                                       # noqa: E402
from vllm_common import L                                      # noqa: E402
from src.eval.evaluate import pooled, within_video_macro      # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = os.path.join(REPO, "runs/20261113_lf_baselines")
OUT = os.path.join(RUNS, "cogvadu_grid")
DS = L.DATASETS
EVAL_NN = list(range(2, 11))
KEPT = [2, 5, 10, 20, 40]
INIT_K = [8, 15, 30, 60, 120]
WTS = [0.125, 0.25, 0.5, 1.0, 2.0]
TWIN = [1, 2, 5, 10, 20, 0]


def ratio(tw, vw):
    return Fraction(tw).limit_denominator() / Fraction(vw).limit_denominator()


def grid():
    """All configurations in grid order."""
    return [(e, k, i, tw, vw, t) for e in EVAL_NN for k in KEPT for i in INIT_K for tw in WTS for vw in WTS
            for t in TWIN]


def gt(ds):
    z = np.load(os.path.join(REPO, f"data/gt_4fps/{ds}.npz"), allow_pickle=True)
    return {str(v): np.asarray(y, np.int8) for v, s, y in zip(z["video_ids"], z["split"], z["y4"]) if str(s) == "test"}


_CACHE = {}


def inputs(ds):
    """Per test video: descriptions, embeddings, raw prob chain values, segment starts, frame count."""
    if ds in _CACHE:
        return _CACHE[ds]
    recs, ro, emb = C.load_all(ds)
    cohort = set(L.cohort(ds))
    out = {}
    for v, r in recs.items():
        if v not in cohort:
            continue
        us = r["units"]
        if r.get("error") or not us or v not in emb or v not in ro:
            raise SystemExit(f"grid input missing: {ds} {v} (error {r.get('error')}, segments {len(us)})")
        out[v] = ([u["description"] for u in us], emb[v][0], emb[v][1], C.chain_values(us, ro[v]),
                  [u["start"] for u in us], L.n_frames(r["duration"]), r["duration"])
    missing = cohort - set(out)
    if missing:
        raise SystemExit(f"grid input missing: {ds} {len(missing)} cohort videos without records")
    _CACHE[ds] = out
    return out


def retrieve(desc, vemb, temb, scores, kept_n, init_k, tw, vw, twin):
    """Stage 5 of refine_video: per segment the kept neighbours' (similarity, score), in kept order."""
    first = {}
    for j, d in enumerate(desc):
        first.setdefault(d, j)
    keys = sorted(first, key=lambda d: first[d])
    rep = [first[d] for d in keys]
    members = [[j for j, d in enumerate(desc) if d == k] for k in keys]
    T = temb[rep] / np.maximum(np.linalg.norm(temb[rep], axis=1, keepdims=True), 1e-12)
    V = vemb / np.maximum(np.linalg.norm(vemb, axis=1, keepdims=True), 1e-12)
    m, e = len(desc), len(keys)
    k_init = min(init_k, e)
    k_keep = min(kept_n, k_init)
    out = []
    for i in range(m):
        D = T @ V[i]
        cand = np.argsort(-D, kind="stable")[:k_init]
        dist = np.abs(np.arange(m) - i)
        w = np.where(dist <= twin, 1.0 + 0.5 * np.exp(-dist / C.DECAY), 1.0)
        vis = (V @ V[i]) * w
        comb = [(tw * (1.0 / (1.0 + D[c])) + vw * max(vis[j] for j in members[c]), c) for c in cand]
        order = sorted(range(len(comb)), key=lambda q: comb[q][0], reverse=True)[:k_keep]
        kept = [comb[q][1] for q in order]
        out.append((np.array([D[c] for c in kept]), np.array([scores[rep[c]] for c in kept])))
    return out


def average(kept, eval_n):
    """Stage 6 of refine_video."""
    s = []
    for sim, val in kept:
        if len(sim) > eval_n:
            top = np.argsort(sim, kind="stable")[-eval_n:]
            sim, val = sim[top], val[top]
        wts = np.exp(sim) / np.sum(np.exp(sim))
        s.append(float(np.sum(val * wts)))
    return s


def metrics(y, cur):
    p, wv = pooled(y, cur), within_video_macro(y, cur)
    return [p["frame_ROC_AUC"], p["frame_PR_AUC"], wv["within_video_macro_ROC_AUC"]]


def _job(args):
    ds, kept_n, init_k, tw, vw, twin = args
    part = os.path.join(OUT, "parts", f"{ds}_k{kept_n}_i{init_k}_w{tw}-{vw}_t{twin}.json")
    if os.path.exists(part):                           # resumable
        return json.load(open(part))
    inp, y = inputs(ds), gt(ds)
    kept = {v: retrieve(d, ve, te, sc, kept_n, init_k, tw, vw, twin) for v, (d, ve, te, sc, *_ ) in inp.items()}
    res = {}
    for en in EVAL_NN:
        cur = {v: L.units_to_4fps(inp[v][4], None, average(kept[v], en), inp[v][5]) for v in inp}
        res[str(en)] = metrics(y, cur)
    out = {"dataset": ds, "kept": kept_n, "init_k": init_k, "tw": tw, "vw": vw, "twin": twin, "by_eval": res}
    json.dump(out, open(part + ".tmp", "w"))
    os.replace(part + ".tmp", part)
    return out


def write_row(ds, cfg, name):
    inp = inputs(ds)
    recs = C.load_records(os.path.join(C.out_dir(ds), "units.jsonl"))
    method = "cogvadu_q3vl_u8_" + name
    rows = {}
    for v, (d, ve, te, sc, starts, n, dur) in inp.items():
        s = C.refine_video(d, ve, te, sc, cfg)
        rows[v] = L.row(method, ds, v, dur, L.units_to_4fps(starts, None, s, n), 1.0 / C.UNIT, C.__file__,
                        extra={"config": cfg})
    for v, r in recs.items():                           # non-cohort records, as cmd_raster writes them
        if v not in rows:
            us = r["units"]
            if r.get("error") or not us:
                rows[v] = L.row(method, ds, v, r["duration"], [], None, C.__file__, error=r.get("error") or "no segments")
    med, failed = L.apply_f2(rows, ds)
    assert not failed, failed
    res = L.finalize(ds, list(rows.values()), os.path.join(C.out_dir(ds), name), method)
    r = res["per_dataset"][0]
    return [r["frame_ROC_AUC"], r["frame_PR_AUC"], r["within_video_macro_ROC_AUC"]]


def main():
    os.makedirs(os.path.join(OUT, "parts"), exist_ok=True)
    # check: split computation == refine_video on the default configuration, and == the stored prob_default metrics
    for ds in DS:
        inp, y = inputs(ds), gt(ds)
        dmax, cur = 0.0, {}
        for v, (d, ve, te, sc, starts, n, _) in inp.items():
            a = C.refine_video(d, ve, te, sc, C.DEFAULT)
            b = average(retrieve(d, ve, te, sc, 10, 30, 0.5, 0.5, 5), 10)
            dmax = max(dmax, max(abs(p - q) for p, q in zip(a, b)))
            cur[v] = L.units_to_4fps(starts, None, b, n)
        ref = json.load(open(os.path.join(C.out_dir(ds), "prob_default", "metrics.json")))["per_dataset"][0]
        mr = [ref["frame_ROC_AUC"], ref["frame_PR_AUC"], ref["within_video_macro_ROC_AUC"]]
        print(ds, "split vs refine_video max |diff|", dmax, "; vs stored prob_default metrics max |diff|",
              max(abs(p - q) for p, q in zip(metrics(y, cur), mr)), flush=True)
    # one job per (corpus, kept, initial_k, ratio, temporal_window); the first weight pair of each ratio computes it
    rep = {}
    for tw in WTS:
        for vw in WTS:
            rep.setdefault(ratio(tw, vw), (tw, vw))
    jobs = [(ds, k, i) + rep[r] + (t,) for ds in DS for k in KEPT for i in INIT_K for r in rep.keys() for t in TWIN]
    done = {}
    with ProcessPoolExecutor(int(os.environ.get("GRID_JOBS", "12"))) as ex:
        for n, r in enumerate(ex.map(_job, jobs, chunksize=1)):
            done[(r["dataset"], r["kept"], r["init_k"], ratio(r["tw"], r["vw"]), r["twin"])] = r["by_eval"]
            if n % 50 == 0:
                print("done", n + 1, "/", len(jobs), flush=True)
    naive = {}
    for ds in DS:
        r = json.load(open(os.path.join(RUNS, "naive/vlm", ds, "metrics.json")))["per_dataset"][0]
        naive[ds] = [r["frame_ROC_AUC"], r["frame_PR_AUC"], r["within_video_macro_ROC_AUC"]]
    G = grid()
    rank = []
    with open(os.path.join(OUT, "grid.jsonl"), "w") as fh:
        for pos, (e, k, i, tw, vw, t) in enumerate(G):
            mm = {ds: done[(ds, k, i, ratio(tw, vw), t)][str(e)] for ds in DS}
            d = [100 * (mm[ds][j] - naive[ds][j]) for ds in DS for j in range(3)]
            rank.append((-sum(x >= 0 for x in d), max([-x for x in d if x < 0], default=0.0), pos))
            fh.write(json.dumps({"eval_neighbors": e, "num_neighbors": k, "initial_k": i, "text_weight": tw,
                                 "visual_weight": vw, "temporal_window": t, "m": mm}) + "\n")
    rank.sort()
    e, k, i, tw, vw, t = G[rank[0][2]]
    cfg = {"initial_k": i, "num_neighbors": k, "eval_neighbors": e, "text_weight": tw, "visual_weight": vw,
           "temporal_window": t}
    sel = {ds: write_row(ds, cfg, "grid_selected") for ds in DS}
    summ = {"n_configurations": len(G), "naive": naive, "selected_config": cfg, "selected": sel,
            "top10": [{"cells": -r[0], "shortfall": r[1], "config": dict(zip(
                ("eval_neighbors", "num_neighbors", "initial_k", "text_weight", "visual_weight", "temporal_window"),
                G[r[2]]))} for r in rank[:10]],
            "range": {}}
    for ds in DS:
        allm = [v[str(en)] for key, v in done.items() if key[0] == ds for en in EVAL_NN]
        summ["range"][ds] = {n: [min(m[j] for m in allm), max(m[j] for m in allm)]
                             for j, n in enumerate(("roc", "pr", "within"))}
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print("DONE grid")


if __name__ == "__main__":
    main()
