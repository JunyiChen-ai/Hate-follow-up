#!/usr/bin/env python3
"""M3 exploration, items 3 and 4 (CPU; test error analysis under rule 10: GT read for the pre-checks and evaluation).

Per-video extent quantities from the existing r6 reads and the r6_bma fit, no free constants:
  item 3  pi_v: per-video mixing proportion MLE of the window reads under M2's corpus-level emissions
          (runs/20260926_twolevel/r6_bma/params.json: per modality N(mu11, s2) when that modality's chain is hateful in
          the window's cells, N(mu10 = mu00, s2) otherwise; reads normal-scored per corpus and modality as
          twolevel_r2 --transform nscore). No time coupling: each window independent given pi. Per modality by EM over
          pi in [0, 1] (unique maximum: the log-likelihood is concave in pi); share = 1 - (1 - pi_visual)(1 - pi_speech),
          the OR fusion of M2 under independence; a modality without reads contributes 0.
            pi_mle       the MLE;  pi_laplace  the MAP under a Beta(2, 2) prior (EM update (sum r + 1) / (n + 2), the
                         rule of succession; keeps logit finite);  pi_vis_mle  the visual modality alone
  item 4  bf: log p(reads | V = 1, M2 segment model, per-video length average) - log p(reads | V = 0)
          = the joint log-odds of twolevel_r2 (--key raw writes it as p_video_logodds) minus the verdict term
          log(pi / (1 - pi)) + log N(z_video; m1, t2) - log N(z_video; m0, t2) (params.json);  bf_per_window = bf / n_windows
  references: pcc = mean over windows of expit(z_w) (probability classify-and-count of the window reads);
          expit_wm = expit(mean window z); joint_lo = the joint log-odds itself (pilot d of twolevel README section 10.2)
Then precheck.analyse on each (Spearman with GT coverage etc.) and key_eval.py replace --scale sd (K = z_video +
sd(wm)/sd(x) x, r6 calibration; ablation verdict_only) for logit pi_laplace, bf, bf_per_window.
Output: runs/20261025_m3_coverage/explore_shares/{values/, precheck/, keyeval/, cli_raw/, summary.txt, summary.json}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASETS, OUT_ROOT, R6, READS, ROOT, gt_sets, load_r6, load_reads, start_run, write_values  # noqa: E402
from explore_lib import nscore  # noqa: E402
from key_eval import r6_cli_args  # noqa: E402
from precheck import analyse, table  # noqa: E402

OUT = OUT_ROOT / "explore_shares"
MODS = ("z_visual", "z_speech")


def lognorm(y, mu, s2):
    return -0.5 * (np.log(2 * np.pi * s2) + (np.asarray(y) - mu) ** 2 / s2)


def mix_em(y, e, alpha=0.0, it=20000, tol=1e-13):
    """EM for the weight p of N(mu11, s2) against N(mu10, s2) on reads y, maximising the log-likelihood plus
    alpha log p + alpha log(1 - p) (alpha = 0: MLE; alpha = 1: MAP under Beta(2, 2)). Update p = (sum r + alpha) /
    (n + 2 alpha); the objective is concave in p, so the fixed point is the unique maximum."""
    if len(y) == 0:
        return 0.0
    d = lognorm(y, e["mu11"], e["s2"]) - lognorm(y, e["mu10"], e["s2"])
    p = 0.5
    for _ in range(it):
        r = expit(np.log(p) - np.log1p(-p) + d)
        q = (float(r.sum()) + alpha) / (len(y) + 2 * alpha)
        q = min(max(q, 1e-300), 1 - 1e-16)
        if abs(q - p) < tol:
            return q
        p = q
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    log = start_run(OUT, "experiments/20261025_m3_coverage/explore_shares.py", vars(a))
    P = json.load(open(R6 / "params.json"))
    rows = {}
    for line in open(READS / "predictions.jsonl"):
        r = json.loads(line)
        if not r.get("error") and r["dataset"] in DATASETS:
            rows[(r["dataset"], r["video_id"])] = r
    # normal scores per corpus and modality, over every window of the corpus with that modality (as twolevel_r2)
    ns = {}
    for ds in DATASETS:
        for m in MODS:
            refs = [(k, i) for k, r in sorted(rows.items()) if k[0] == ds for i, w in enumerate(r["extra"]["windows"]) if m in w]
            z = nscore([rows[k]["extra"]["windows"][i][m] for k, i in refs])
            for (k, i), x in zip(refs, z):
                ns[(k, i, m)] = float(x)
    vals = {n: {} for n in ("pi_mle", "pi_laplace", "pi_vis_mle", "logit_pi_laplace", "pcc", "expit_wm")}
    for k, r in rows.items():
        e = P[k[0]]["emit"]; ws = r["extra"]["windows"]
        mle, lap = {}, {}
        for m in MODS:
            y = np.array([ns[(k, i, m)] for i, w in enumerate(ws) if m in w])
            mle[m] = mix_em(y, e[m], 0.0); lap[m] = mix_em(y, e[m], 1.0)
        vals["pi_mle"][k] = 1 - (1 - mle["z_visual"]) * (1 - mle["z_speech"])
        vals["pi_laplace"][k] = 1 - (1 - lap["z_visual"]) * (1 - lap["z_speech"])
        vals["pi_vis_mle"][k] = mle["z_visual"]
        vals["logit_pi_laplace"][k] = math.log(vals["pi_laplace"][k]) - math.log1p(-vals["pi_laplace"][k])
        z = np.array([float(w["z"]) for w in ws])
        vals["pcc"][k] = float(expit(z).mean()); vals["expit_wm"][k] = float(expit(z.mean()))
    log("pi: share of videos with pi_mle < 1e-6: " + ", ".join(
        f"{ds} {np.mean([vals['pi_mle'][k] < 1e-6 for k in vals['pi_mle'] if k[0] == ds]):.3f}" for ds in DATASETS))
    # item 4: the joint log-odds from twolevel_r2 with --key raw (M2 and EM unchanged)
    cli = OUT / "cli_raw"
    args = r6_cli_args(R6, READS)
    args[args.index("--key") + 1] = "raw"
    cmd = [sys.executable, str(ROOT / "experiments/20260926_twolevel/twolevel_r2.py"), "--run", str(READS), *args,
           "--tag", "key_raw", "--out-root", str(cli), "--datasets", *DATASETS]
    log("CLI " + " ".join(cmd[1:]))
    cli.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=open(cli / "key_raw.stdout", "w"), stderr=subprocess.STDOUT)
    r6 = load_r6()
    vals["bf"], vals["bf_per_window"], vals["joint_lo"] = {}, {}, {}
    maxd = 0.0
    for line in open(cli / "key_raw" / "predictions.jsonl"):
        r = json.loads(line); k = (r["dataset"], r["video_id"]); p = P[k[0]]
        lo = float(r["extra"]["p_video_logodds"]); zv = float(r["extra"]["z_video"])
        lov = math.log(p["pi"]) - math.log(1 - p["pi"]) + float(lognorm(zv, p["m1"], p["t2"]) - lognorm(zv, p["m0"], p["t2"]))
        vals["joint_lo"][k] = lo; vals["bf"][k] = lo - lov
        vals["bf_per_window"][k] = (lo - lov) / max(1, len(rows[k]["extra"]["windows"]))
        maxd = max(maxd, float(np.abs(np.asarray(r["extra"]["cell_prob"]) - r6[k]["cell_prob"]).max()))
    log(f"CLI --key raw: M2 cell_prob vs r6_bma max abs diff {maxd:.2e}")
    for n, v in vals.items():
        write_values(OUT / "values" / f"{n}.jsonl", v)
    # pre-checks
    gts, reads = gt_sets(), load_reads()
    pre = {}
    for n, v in vals.items():
        res = analyse(v, gts, r6, reads, 2 * a.B, a.seed)
        d = OUT / "precheck" / n; d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(res, indent=1))
        (d / "summary.txt").write_text(table(res, f"explore_shares precheck {n}") + "\n")
        pre[n] = res
        log(f"precheck {n} done")
    # key evaluation (replace, sd scaling) for the log-odds-type quantities, with the verdict-only ablation
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    jobs = {}
    for n in ("logit_pi_laplace", "bf", "bf_per_window"):
        d = OUT / "keyeval" / n; d.mkdir(parents=True, exist_ok=True)
        c = [sys.executable, str(ROOT / "experiments/20261025_m3_coverage/key_eval.py"), "--values", str(OUT / "values" / f"{n}.jsonl"),
             "--name", n, "--forms", "replace", "verdict_only", "--scale", "sd", "--B", str(a.B), "--seed", str(a.seed), "--out", str(d)]
        jobs[n] = subprocess.Popen(c, cwd=ROOT, env=env, stdout=open(d / "stdout.log", "w"), stderr=subprocess.STDOUT)
    ke = {}
    for n, p in jobs.items():
        if p.wait() != 0:
            raise SystemExit(f"key_eval {n} FAILED; see {OUT / 'keyeval' / n / 'stdout.log'}")
        ke[n] = json.load(open(OUT / "keyeval" / n / "summary.json"))
        log(f"key_eval {n} done")
    # summary
    L = ["explore_shares (items 3, 4): development-selected test error analysis; Spearman with GT coverage "
         "(hateful = videos with a GT-positive frame); window mean reference .416 / .533"]
    L.append(f"\n{'quantity':18s} " + "  ".join(f"{g + ' hateful rho [CI]':>30s}" for g in gts) + "   rho(.,z_video) all   partial|wm hateful")
    for n, res in pre.items():
        cells = [f"{res[g]['hateful']['rho_value_cov']['point']:+.3f} [{res[g]['hateful']['rho_value_cov']['ci95'][0]:+.3f},"
                 f"{res[g]['hateful']['rho_value_cov']['ci95'][1]:+.3f}]" for g in gts]
        L.append(f"{n:18s} " + "  ".join(f"{c:>30s}" for c in cells) + "   " +
                 " / ".join(f"{res[ds]['all']['rho_value_zv']['point']:+.3f}" for ds in DATASETS) + "        " +
                 " / ".join(f"{res[ds]['hateful']['partial_value_cov_given_wm']['point']:+.3f}" for ds in DATASETS))
    L.append(f"{'(window mean)':18s} " + "  ".join(f"{pre['pcc'][g]['hateful']['rho_wm_cov']['point']:>+30.3f}" for g in gts))
    L.append("\nkey_eval replace (K = z_video + sd(wm)/sd(x) x, calibrated) and verdict_only (ablation: the extent term "
             "removed); pooled ROC / PR / within; d vs r6_bma [95% paired bootstrap]")
    for n, s in ke.items():
        for g, r in s["bootstrap"].items():
            for f in ("replace", "verdict_only"):
                p = r["point"][f]; x = r["delta"][f]
                L.append(f"  {n:16s} {f:12s} {g:19s} {p[0]:.4f} / {p[1]:.4f} / {p[2]:.4f}  dROC {x['dROC']:+.4f} "
                         f"[{x['ROC_ci95'][0]:+.4f},{x['ROC_ci95'][1]:+.4f}]  dPR {x['dPR']:+.4f} [{x['PR_ci95'][0]:+.4f},"
                         f"{x['PR_ci95'][1]:+.4f}]  dW {x['dWithin']:+.4f}")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    (OUT / "summary.json").write_text(json.dumps({"precheck": {n: {g: {s: {k: v for k, v in r[g][s].items()}
                                                                         for s in ("all", "hateful")} for g in gts}
                                                               for n, r in pre.items()},
                                                  "keyeval": {n: s["bootstrap"] for n, s in ke.items()}}, indent=1, default=str))
    print(txt)
    log("RUN_DONE")


if __name__ == "__main__":
    main()
