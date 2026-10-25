#!/usr/bin/env python3
"""Candidate 50, stage 0 (GT-free): the three stop statistics of review_m1.md §5.3, per corpus.

Reads only model reads (no GT, no labels). For each video among the first N by sorted id of each corpus:
  d   = z_visual(A1) - z_visual(all_local), one value per window (every window has a visual branch);
  d_o = z_visual(A1_other) - z_visual(all_local).
Statistics (gates are on the visual change; README "Implementation reading of the protocol"):
  S1 median over videos of the within-video sd of d (population sd)          pass if >= .5 logit
  S2 median over videos of the constant share n * mean(d)^2 / sum(d^2)          pass if <= .8
     (a video whose change is exactly zero everywhere counts as share 1)
  S3 median over videos of the within-video Pearson correlation of d and d_o    pass if <= .9
S1 and S2 use videos with >= 2 windows, S3 videos with >= 3 windows and non-zero sd of both changes.
Stop if any statistic fails on any corpus. Diagnostics (not gates): the same statistics on the speech change, the
Spearman form of S3, mean changes, and |A1_other - A1|.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.video_inputs import load_manifest  # noqa: E402

MANIFESTS = {"HateMM": "data/omsl_v6_inputs/manifests/all_test.jsonl",
             "HateClipSeg": "data/omsl_v6_inputs/manifests/all_test.jsonl",
             "DeHate": "data/manifests/DeHate_test.jsonl"}
S1_MIN_SD = .5     # logit
S2_MAX_SHARE = .8
S3_MAX_CORR = .9
ARMS = ("all_local", "A1", "A1_other")


def jl(path):
    out = {}
    for r in map(json.loads, open(path)):
        key = (r["dataset"], r["video_id"])
        assert key not in out and not r.get("error"), key
        out[key] = r
    return out


def changes(r_arm, r_ref, mod):
    """Per-window change arm - all_local for modality `mod` over windows where both have it."""
    wa, wr = r_arm["extra"]["windows"], r_ref["extra"]["windows"]
    assert [(w["start"], w["end"]) for w in wa] == [(w["start"], w["end"]) for w in wr]
    idx = [i for i in range(len(wr)) if mod in wr[i] and mod in wa[i]]
    assert all((mod in wa[i]) == (mod in wr[i]) for i in range(len(wr))), "branch sets differ between arms"
    return np.array([wa[i][mod] - wr[i][mod] for i in idx], float)


def video_stats(d, d_o):
    n = len(d)
    out = {"n_windows": n, "mean_change": float(d.mean()) if n else None,
           "mean_abs_change": float(np.abs(d).mean()) if n else None,
           "mean_abs_other_minus_A1": float(np.abs(d_o - d).mean()) if n else None}
    if n >= 2:
        out["sd"] = float(d.std())
        ss = float((d ** 2).sum())
        out["constant_share"] = float(n * d.mean() ** 2 / ss) if ss > 0 else 1.0
    if n >= 3 and d.std() > 0 and d_o.std() > 0:
        out["corr_pearson"] = float(pearsonr(d, d_o).statistic)
        out["corr_spearman"] = float(spearmanr(d, d_o).statistic)
    return out


def med(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return (float(np.median(v)) if v else None), len(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reads", type=Path, help="run directory with all_local/, A1/, A1_other/ predictions")
    ap.add_argument("--arm-map", nargs="*", default=[], metavar="ARM=DIR",
                    help="dry runs only: read arm ARM from DIR/predictions.jsonl instead of <reads>/<ARM>")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg", "DeHate"])
    ap.add_argument("--first", type=int, default=20)
    ap.add_argument("--out", type=Path, help="summary json (default <reads>/stage0_summary.json)")
    a = ap.parse_args()
    src = {arm: (a.reads / arm if a.reads else None) for arm in ARMS}
    for item in a.arm_map:
        arm, path = item.split("=", 1)
        assert arm in ARMS, arm
        src[arm] = Path(path)
    assert all(src.values()), "give --reads or every arm in --arm-map"
    out_path = a.out or (a.reads / "stage0_summary.json")
    reads = {arm: jl(p / "predictions.jsonl") for arm, p in src.items()}
    result = {"host": socket.gethostname(), "date": time.strftime("%Y-%m-%d"), "GT_read": False,
              "sources": {arm: str(p) for arm, p in src.items()}, "first": a.first,
              "thresholds": {"S1_min_median_sd": S1_MIN_SD, "S2_max_median_constant_share": S2_MAX_SHARE,
                             "S3_max_median_corr": S3_MAX_CORR}, "datasets": {}}
    stop = False
    lines = []
    for ds in a.datasets:
        ids = sorted(r["video_id"] for r in load_manifest(ROOT / MANIFESTS[ds], [ds]))[:a.first]
        missing = {arm: [v for v in ids if (ds, v) not in reads[arm]] for arm in ARMS}
        assert not any(missing.values()), (ds, {k: v[:3] for k, v in missing.items() if v})
        per = {}
        for mod in ("z_visual", "z_speech"):
            rows = []
            for v in ids:
                al, a1, ao = (reads[arm][(ds, v)] for arm in ARMS)
                assert a1["extra"]["z_video"] == al["extra"]["z_video"] == ao["extra"]["z_video"], (ds, v)
                d, d_o = changes(a1, al, mod), changes(ao, al, mod)
                rows.append({"video_id": v, **video_stats(d, d_o)})
            per[mod] = rows
        summary = {}
        for mod, rows in per.items():
            s1, n1 = med(rows, "sd")
            s2, n2 = med(rows, "constant_share")
            s3, n3 = med(rows, "corr_pearson")
            s3s, _ = med(rows, "corr_spearman")
            summary[mod] = {"S1_median_sd": s1, "S2_median_constant_share": s2, "S3_median_corr_pearson": s3,
                            "S3_median_corr_spearman": s3s, "n_videos_S1_S2": n1, "n_videos_S3": n3,
                            "median_mean_change": med(rows, "mean_change")[0],
                            "median_mean_abs_change": med(rows, "mean_abs_change")[0],
                            "median_mean_abs_other_minus_A1": med(rows, "mean_abs_other_minus_A1")[0]}
        v = summary["z_visual"]
        gates = {"S1": v["S1_median_sd"] is not None and v["S1_median_sd"] >= S1_MIN_SD,
                 "S2": v["S2_median_constant_share"] is not None and v["S2_median_constant_share"] <= S2_MAX_SHARE,
                 "S3": v["S3_median_corr_pearson"] is not None and v["S3_median_corr_pearson"] <= S3_MAX_CORR}
        stop |= not all(gates.values())
        result["datasets"][ds] = {"videos": ids, "summary": summary, "gates": gates, "per_video": per}
        for mod in ("z_visual", "z_speech"):
            s = summary[mod]
            f = lambda x: "nan" if x is None else f"{x:.3f}"  # noqa: E731
            tag = "  gate" if mod == "z_visual" else "  diag"
            lines.append(f"{ds:12s} {mod:9s}{tag} S1 sd {f(s['S1_median_sd'])} (n {s['n_videos_S1_S2']})  "
                         f"S2 share {f(s['S2_median_constant_share'])}  S3 r {f(s['S3_median_corr_pearson'])} "
                         f"(rho {f(s['S3_median_corr_spearman'])}, n {s['n_videos_S3']})  mean change "
                         f"{f(s['median_mean_change'])}  |other-A1| {f(s['median_mean_abs_other_minus_A1'])}")
        lines.append(f"{ds:12s} " + "  ".join(f"{g} {'PASS' if ok else 'FAIL'}" for g, ok in gates.items()))
    result["verdict"] = "STOP" if stop else "CONTINUE"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1) + "\n")
    print("\n".join(lines))
    print(f"STAGE0_VERDICT {result['verdict']}  ({out_path})")


if __name__ == "__main__":
    main()
