#!/usr/bin/env python3
"""M3 exploration, items 1 and 2 (CPU; test error analysis under rule 10: every variant tried is listed here and in
the output; GT read for evaluation only). Needs explore_shares.py's values (pcc, pi_laplace, expit_wm, joint_lo).

Item 1, two-part (hurdle) frame score  log P(V = 1) + log q_t,  q_t = P(frame t in the behaviour | V = 1):
  P(V = 1):  K   = expit(r6_bma calibrated key)                 (z_video + window mean, two-Gaussian calibration)
             zv  = expit(the same calibration fitted on z_video alone)
             none = the term removed (ablation)
  q_t:       M2        the M2 posterior P(hate at t | V = 1, reads) of r6_bma (cell_prob on the 4 fps grid)
             shift_c   the M2 posterior with its prevalence moved to a share estimate c_v (Saerens-style prior shift):
                       q_t = expit(logit p_t + delta_v), delta_v solving mean_t q_t = c_v
             lin_c     share x rank: q_t = 2 c_v u_t, u_t = within-video rank quantile of the M2 posterior (mean 1)
             lin_1     rank only, u_t (the share removed)
  c_v:       pbar (mean M2 posterior; shift_pbar = M2), pcc (mean_w expit z_w), pi (explore_shares pi_laplace),
             ewm (expit of the window mean)
Item 2, width of the within-video term against the key (score = key + w_v r_t, r_t = r6's centred rank):
  w_dprime   w = d' = (mu1 - mu0) / sd of the key's two-Gaussian mixture (per corpus)
  w_bern     w_v = 4 P(V = 1 | K) (1 - P(V = 1 | K)): the within term widest where the key is least certain
  s_<s>      the declared scan of twolevel README section 11.1 on the calibrated key: score = s key + r_t, s in
             {.5, .25, .125}
  joint_key  the M2 joint posterior log-odds as the key (twolevel README section 10.2 pilot d, on the r6 time level)
Ablation pairs (revised G2: removing the component drops a main metric by >= .01 on both corpora):
  share of lin_c -> lin_1;  share of shift_c -> M2;  M2's level -> lin_1;  P term -> none;  item 2 rule -> r6_bma.
Checks before use: frames()/quantile_rank() reproduce r6's rank term; key_mixture() reproduces r6's (a, b) and the
CLI verdict-only key; H_K_M2 reproduces runs/20260926_twolevel/final_full (--arm full).
Output: runs/20261025_m3_coverage/explore_hurdle/{arms/<arm>/{predictions.jsonl, metrics.json}, summary.txt, summary.json}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import expit, log_expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASETS, OUT_ROOT, ROOT, gt_sets, load_r6, load_reads, load_values, start_run  # noqa: E402
from explore_lib import (METRICS, ablation_drop, evaluate_scores, fmt, frames, g1, key_mixture,  # noqa: E402
                         paired_bootstrap, quantile_rank)

OUT = OUT_ROOT / "explore_hurdle"
SHARES = OUT_ROOT / "explore_shares" / "values"
FULL = ROOT / "runs/20260926_twolevel/final_full"
VERDICT_CLI = OUT_ROOT / "analysis_keyeval_selftest/cli/verdict_only"
QS = ["M2", "shift_pcc", "shift_pi", "shift_ewm", "lin_pbar", "lin_pcc", "lin_pi", "lin_ewm", "lin_1"]
PS = ["K", "zv", "none"]


def shift(lp, c, lo=-80.0, hi=80.0):
    """delta with mean(expit(lp + delta)) = c, by bisection (monotone)."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if expit(lp + mid).mean() < c:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def boot_one(args):
    arms, base, g, B, seed, boot = args
    return paired_bootstrap(arms, base, g, B, seed, boot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    log = start_run(OUT, "experiments/20261025_m3_coverage/explore_hurdle.py", vars(a))
    r6, reads, gts = load_r6(), load_reads(), gt_sets()
    P6 = json.load(open(ROOT / "runs/20260926_twolevel/r6_bma/params.json"))
    sh = {n: load_values(SHARES / f"{n}.jsonl") for n in ("pcc", "pi_laplace", "expit_wm", "joint_lo")}
    keys = sorted(r6)
    # ---- checks
    V = {}
    for k in keys:
        L = len(r6[k]["score"]); p = frames(r6[k]["cell_prob"], L); lp = np.log(p) - np.log1p(-p)
        V[k] = {"p": p, "lp": lp, "u": quantile_rank(lp), "L": L}
    d_rank = max(float(np.abs(V[k]["u"] - 0.5 - r6[k]["rank"]).max()) for k in keys)
    mix, mixz, d_ab, d_lo = {}, {}, 0.0, 0.0
    for ds in DATASETS:
        ks = [k for k in keys if k[0] == ds]
        mix[ds] = key_mixture([r6[k]["zv"] + reads[k]["wm"] for k in ks])
        mixz[ds] = key_mixture([r6[k]["zv"] for k in ks])
        d_ab = max(d_ab, abs(mix[ds]["a"] - P6[ds]["key_ab"][0]), abs(mix[ds]["b"] - P6[ds]["key_ab"][1]))
        d_lo = max(d_lo, max(abs(mix[ds]["a"] * (r6[k]["zv"] + reads[k]["wm"]) + mix[ds]["b"] - r6[k]["lo"]) for k in ks))
    lo_zv = {k: mixz[k[0]]["a"] * r6[k]["zv"] + mixz[k[0]]["b"] for k in keys}
    cli_zv = {(r["dataset"], r["video_id"]): float(r["extra"]["p_video_logodds"])
              for r in map(json.loads, open(VERDICT_CLI / "predictions.jsonl"))}
    d_zv = max(abs(lo_zv[k] - cli_zv[k]) for k in keys)
    full = {(r["dataset"], r["video_id"]): np.asarray(r["score_curve"]) for r in map(json.loads, open(FULL / "predictions.jsonl"))}
    d_full = max(float(np.abs(float(log_expit(r6[k]["lo"])) + np.log(V[k]["p"]) - full[k]).max()) for k in keys)
    checks = {"rank term": d_rank, "key (a, b) vs params.json": d_ab, "calibrated key vs r6": d_lo,
              "verdict-only key vs CLI": d_zv, "H_K_M2 vs final_full": d_full}
    for c, x in checks.items():
        log(f"CHECK {c}: max abs diff {x:.2e} {'PASS' if x < 1e-8 else 'FAIL'}")
    if any(x >= 1e-8 for x in checks.values()):
        raise SystemExit("CHECK FAILED")
    # ---- item 1
    shares = {"pbar": {k: float(V[k]["p"].mean()) for k in keys}, "pcc": sh["pcc"], "pi": sh["pi_laplace"], "ewm": sh["expit_wm"]}
    logP = {"K": {k: float(log_expit(r6[k]["lo"])) for k in keys}, "zv": {k: float(log_expit(lo_zv[k])) for k in keys},
            "none": {k: 0.0 for k in keys}}
    logq = {}
    for q in QS:
        logq[q] = {}
        for k in keys:
            v = V[k]
            if q == "M2":
                x = np.log(v["p"])
            elif q.startswith("shift_"):
                c = shares[q[6:]][k]; x = log_expit(v["lp"] + shift(v["lp"], c))
            elif q == "lin_1":
                x = np.log(v["u"])
            else:
                x = math.log(2 * shares[q[4:]][k]) + np.log(v["u"])
            logq[q][k] = np.asarray(x, float)
    arms = {f"H_{P}_{q}": {k: logP[P][k] + logq[q][k] for k in keys} for P in PS for q in QS}
    # ---- item 2
    for s in (0.5, 0.25, 0.125):
        arms[f"s_{s:g}"] = {k: s * r6[k]["lo"] + r6[k]["rank"] for k in keys}
    arms["w_dprime"] = {k: r6[k]["lo"] + mix[k[0]]["dprime"] * r6[k]["rank"] for k in keys}
    arms["w_bern"] = {k: r6[k]["lo"] + 4 * expit(r6[k]["lo"]) * (1 - expit(r6[k]["lo"])) * r6[k]["rank"] for k in keys}
    arms["joint_key"] = {k: sh["joint_lo"][k] + r6[k]["rank"] for k in keys}
    log("d' of the key mixture: " + ", ".join(f"{ds} {mix[ds]['dprime']:.3f}" for ds in DATASETS))
    # ---- evaluator metrics (metrics.json per arm) and paired bootstrap
    ev = {n: evaluate_scores(n, s, DATASETS, OUT) for n, s in arms.items()}
    log(f"evaluated {len(ev)} arms")
    base = {k: r6[k]["score"] for k in keys}
    boot = list(arms)
    with ProcessPoolExecutor(3) as ex:
        parts = list(ex.map(boot_one, [(arms, base, {g: v}, a.B, a.seed, boot) for g, v in gts.items()]))
    res = {g: p[g] for p, g in zip(parts, gts)}
    for n in arms:                                   # bootstrap points use the same evaluator functions; cross-check
        for ds in DATASETS:
            for j, m in enumerate(METRICS):
                x, y = res[ds]["arms"][n]["point"][j], ev[n][ds][m]
                if (x is None) != (y is None) or (x is not None and abs(x - y) > 1e-12):
                    raise SystemExit(f"{n} {ds} {m}: bootstrap point {x} vs metrics.json {y}")
    # ---- ablations and G1
    pairs = {}
    for P in PS:
        for q in QS:
            n = f"H_{P}_{q}"; ab = {"P_term": f"H_none_{q}"} if P != "none" else {}
            if q.startswith("shift_"):
                ab["share"] = f"H_{P}_M2"
            elif q.startswith("lin_") and q != "lin_1":
                ab["share"] = f"H_{P}_lin_1"
            elif q == "M2":
                ab["M2_level"] = f"H_{P}_lin_1"
            pairs[n] = ab
    for n in ("s_0.5", "s_0.25", "s_0.125", "w_dprime", "w_bern", "joint_key"):
        pairs[n] = {}
    main = list(DATASETS)
    tab = {}
    for n in arms:
        tab[n] = {"G1": g1(res, main, n), "ablations": {c: {"to": t, **ablation_drop(res, main, n, t)} for c, t in pairs[n].items()}}
    # ---- write
    L = [f"explore_hurdle (items 1, 2): development-selected test error analysis; every variant tried is listed. "
         f"Columns: pooled ROC / PR / within, d = arm - r6_bma [95% paired video bootstrap, B={a.B}]; "
         f"HateClipSeg_strict = hateful-only GT (diagnostic). G1 = no main metric below r6_bma by more than .005 / .01."]
    L += [f"checks: " + "; ".join(f"{c} {x:.1e}" for c, x in checks.items())]
    L.append(f"r6_bma: " + "  ".join(f"{g} {fmt(res[g]['r6_bma'][0])} / {fmt(res[g]['r6_bma'][1])} / {fmt(res[g]['r6_bma'][2])}" for g in gts))
    for n in arms:
        head = f"\n{n:16s}"
        if n in tab:
            head += f" G1 {'PASS' if tab[n]['G1'] else 'FAIL'}"
            for c, x in tab[n]["ablations"].items():
                drops = [f"{m.split('_')[1] if m != 'within_video_macro_ROC_AUC' else 'within'}" for m in METRICS if x[m]]
                head += f"  | remove {c} (-> {x['to']}): drop >= .01 on both corpora: {', '.join(drops) if drops else 'none'}"
        L.append(head)
        for g in gts:
            r = res[g]["arms"][n]; p, d = r["point"], r["delta"]
            ci_r, ci_p = r.get("ROC_ci95", [None, None]), r.get("PR_ci95", [None, None])
            L.append(f"   {g:19s} {fmt(p[0])} / {fmt(p[1])} / {fmt(p[2])}   d {fmt(d[0], True)} [{fmt(ci_r[0], True)},"
                     f"{fmt(ci_r[1], True)}] / {fmt(d[1], True)} [{fmt(ci_p[0], True)},{fmt(ci_p[1], True)}] / {fmt(d[2], True)}")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    (OUT / "summary.json").write_text(json.dumps({"checks": checks, "mixture": mix, "mixture_zv": mixz, "results": res,
                                                  "gates": tab, "evaluator_metrics": ev}, indent=1, default=str))
    print(txt)
    log("RUN_DONE")


if __name__ == "__main__":
    main()
