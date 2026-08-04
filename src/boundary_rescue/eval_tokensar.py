#!/usr/bin/env python3
"""Evaluate TokenSAR-gated rescue rules vs V1 baseline across 4 datasets.

Gate logic: per sample, accept the stage-2 judge flip only if TokenSAR <= tau.
Tau is chosen label-free per sample (dataset-agnostic universal threshold in
quantile units of the band's TokenSAR distribution — 'q20', 'q50' etc).

Inputs (per dataset, under results/boundary_rescue/<ds>/):
  - baseline_preds_v2.jsonl            stage-1 predictions (pred_baseline)
  - candidates_bayes_band_<mode>.jsonl band membership + posterior_hi
  - offline_test_band_tokensar_<model>.jsonl   judge output with `tokensar`

Output: printed table of {judge × band × rule × tau} passing V1 gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data/jehc223/EMNLP2/results/boundary_rescue")
DATA = Path("/data/jehc223/EMNLP2/datasets")
DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
V1 = {"MHClip_EN": (0.7826, 0.6958),
      "MHClip_ZH": (0.8255, 0.8023),
      "HateMM":    (0.8465, 0.8362),
      "ImpliHateVid": (0.8204, 0.8199)}
N_TEST = {"MHClip_EN": 161, "MHClip_ZH": 149, "HateMM": 215, "ImpliHateVid": 401}
LABMAP = {"MHClip_EN": {"Hateful": 1, "Offensive": 1, "Normal": 0},
          "MHClip_ZH": {"Hateful": 1, "Offensive": 1, "Normal": 0},
          "HateMM":    {"Hate": 1, "Non Hate": 0},
          "ImpliHateVid": {"Hateful": 1, "Normal": 0}}


def f1m(y, yh):
    cls = sorted(set(y))
    s = 0.0
    for c in cls:
        tp = sum(1 for a, b in zip(y, yh) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, yh) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, yh) if a == c and b != c)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        s += 2 * p * r / (p + r) if p + r else 0
    return s / len(cls) if cls else 0


def ld(p: Path):
    return [json.loads(l) for l in open(p)] if p.exists() else []


def eval_config(judge_files, band_fname, tau_mode, rule,
                tokensar_field="tokensar"):
    """tau_mode ∈ {'q20','q30','q50','q70','median','lt0','lt1','lt2','all'}.
    rule ∈ {'raw_gated','out_med_gated'} — gated = only apply rule when
        TokenSAR <= dataset_tau (computed on band).
    """
    results = []
    for ds in DATASETS:
        ann = json.load(open(DATA / ds / "annotation(new).json"))
        lab = {r["Video_ID"]: LABMAP[ds].get(r["Label"], -1) for r in ann}
        base_rows = ld(ROOT / ds / "baseline_preds_v2.jsonl")
        base = {r["video_id"]: r["pred_baseline"] for r in base_rows}
        band_rows = ld(ROOT / ds / band_fname)
        band_post = {r["video_id"]: r["posterior_hi"] for r in band_rows}
        if not band_post:
            return None, None
        jf = judge_files.get(ds)
        if jf is None:
            return None, None
        jrec_rows = ld(ROOT / ds / jf)
        jrec = {r["video_id"]: r for r in jrec_rows}
        # pick per-sample TokenSAR in band
        ts_vals = []
        for vid in band_post:
            r = jrec.get(vid)
            if r is None:
                continue
            v = r.get(tokensar_field)
            if v is not None:
                ts_vals.append(float(v))
        if len(ts_vals) < 5:
            return None, None
        ts_arr = np.asarray(ts_vals, dtype=np.float64)
        if tau_mode.startswith("q"):
            q = int(tau_mode[1:])
            tau = float(np.percentile(ts_arr, q))
        elif tau_mode == "median":
            tau = float(np.median(ts_arr))
        elif tau_mode.startswith("lt"):
            tau = float(tau_mode[2:])
        elif tau_mode == "all":
            tau = float("inf")
        else:
            raise ValueError(tau_mode)
        post_med = float(np.median(list(band_post.values())))
        valid = [v for v in base if lab.get(v, -1) >= 0]
        y = [lab[v] for v in valid]
        yh = [base[v] for v in valid]
        for i, v in enumerate(valid):
            if v not in band_post or v not in jrec:
                continue
            rec = jrec[v]
            pr = rec.get("pred")
            if pr not in (0, 1):
                continue
            ts = rec.get(tokensar_field)
            if ts is None or float(ts) > tau:
                continue  # gate: refuse flip when uncertain
            if rule == "raw_gated":
                yh[i] = pr
            elif rule == "out_med_gated":
                if band_post[v] < post_med:
                    yh[i] = pr
            else:
                raise ValueError(rule)
        c = sum(1 for a, b in zip(y, yh) if a == b)
        m = f1m(y, yh)
        acc = c / N_TEST[ds]
        v1a, v1m_ = V1[ds]
        ok = (acc >= v1a - 1e-9) and (m >= v1m_ - 1e-9)
        results.append((ds, c, acc, m, ok, tau, len(ts_arr)))
    return results


def print_row(label, results):
    if results is None:
        return
    passes = sum(1 for r in results if r[4])
    cells = " ".join(
        f"{r[0][:2]}:{r[1]}/{r[3]:.3f}{'✓' if r[4] else '✗'}"
        for r in results
    )
    taus = "/".join(f"{r[5]:.3f}" for r in results)
    bannerp = "🎯" if passes == 4 else ""
    print(f"  {label:60s} | {cells} | τ={taus} | {passes}/4 {bannerp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-stem", required=True,
                    help="stem like 'qwen2.5-vl-32b-awq' → reads "
                         "offline_test_band_tokensar_<stem>.jsonl per ds")
    ap.add_argument("--bands", nargs="+",
                    default=["candidates_bayes_band_rate.jsonl",
                             "candidates_bayes_band_mass0.75.jsonl",
                             "candidates_bayes_band_mass1.00.jsonl"])
    ap.add_argument("--taus", nargs="+",
                    default=["q20", "q30", "q50", "q70", "all"])
    ap.add_argument("--rules", nargs="+",
                    default=["raw_gated", "out_med_gated"])
    ap.add_argument("--field", default="tokensar")
    args = ap.parse_args()
    stem = args.judge_stem
    judge_files = {
        ds: f"offline_test_band_lp_tsar_{stem}_scored.jsonl" for ds in DATASETS
    }
    print(f"Judge={stem}  field={args.field}")
    print(f"V1 = {V1}")
    for band in args.bands:
        for tau in args.taus:
            for rule in args.rules:
                res = eval_config(judge_files, band, tau, rule,
                                  tokensar_field=args.field)
                label = f"{band[:35]} | {rule:14s} | τ={tau}"
                print_row(label, res)


if __name__ == "__main__":
    main()
