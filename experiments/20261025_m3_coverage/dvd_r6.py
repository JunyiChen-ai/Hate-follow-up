#!/usr/bin/env python3
"""DVD 'verdict + T' (experiments/20260927_dvd README section 9, arm dvd_noE, first tested on the r3_m2 time level)
re-decoded on the r6_bma time level (CPU; GT read for evaluation only).

  twolevel_r2.py with r6_bma's flags (from each corpus's r6_bma config.json) + --dvd-reads <DVD reads> --dvd-conds viol,T
  key = logit( P(V = 1 | K_r6) * P_T(z_target) ), P_T = the two-Gaussian calibration of the T read per corpus.
  HateMM, HateClipSeg: runs/20260926_glr/base_gridA + runs/20260927_dvd/reads_main/reads.jsonl
  DeHate:              runs/20260927_dehate_external/reads_gridA + runs/20260927_dvd/reads_dehate/reads.jsonl
Reports per corpus: pooled ROC / PR / within and the paired video bootstrap of the pooled differences against r6_bma
(src/eval/evaluate.pooled), the HateClipSeg hateful-only GT diagnostic, the T-removal ablation (= r6_bma, i.e. minus the
difference), G1 per corpus, M2 unchanged (cell_prob vs r6_bma), and the key maximum (the DVD path clamps at logit 27.6).
Output: runs/20261025_m3_coverage/dvd_r6/{cli/<tag>/..., summary.txt, summary.json, run.log, config.json}
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORPORA, OUT_ROOT, ROOT, gt_sets, load_r6, start_run  # noqa: E402
from explore_lib import NOISE_POOLED, NOISE_WITHIN, fmt, paired_bootstrap  # noqa: E402
from key_eval import r6_cli_args  # noqa: E402

OUT = OUT_ROOT / "dvd_r6"
GROUPS = {"main": (("HateMM", "HateClipSeg"), ROOT / "runs/20260927_dvd/reads_main/reads.jsonl"),
          "dehate": (("DeHate",), ROOT / "runs/20260927_dvd/reads_dehate/reads.jsonl")}
CLAMP = 27.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    log = start_run(OUT, "experiments/20261025_m3_coverage/dvd_r6.py", vars(a))
    jobs = {}
    for g, (dss, dvd) in GROUPS.items():
        r6_dir, reads_dir = CORPORA[dss[0]]["r6"], CORPORA[dss[0]]["reads"]
        cmd = [sys.executable, str(ROOT / "experiments/20260926_twolevel/twolevel_r2.py"), "--run", str(reads_dir),
               *r6_cli_args(r6_dir, reads_dir), "--dvd-reads", str(dvd), "--dvd-conds", "viol,T", "--tag", f"dvd_noE_r6_{g}",
               "--out-root", str(OUT / "cli"), "--datasets", *dss]
        (OUT / "cli").mkdir(parents=True, exist_ok=True)
        log("CLI " + " ".join(cmd[1:]))
        jobs[g] = subprocess.Popen(cmd, cwd=ROOT, stdout=open(OUT / "cli" / f"{g}.stdout", "w"), stderr=subprocess.STDOUT)
    for g, p in jobs.items():
        if p.wait() != 0:
            raise SystemExit(f"CLI {g} FAILED; see {OUT / 'cli' / (g + '.stdout')}")
        log(f"CLI {g} done")
    datasets = tuple(ds for dss, _ in GROUPS.values() for ds in dss)
    r6, gts = load_r6(datasets=datasets), gt_sets(datasets)
    arm, keymax, nclamp, dcell = {}, {}, {}, 0.0
    for g, (dss, _) in GROUPS.items():
        for line in open(OUT / "cli" / f"dvd_noE_r6_{g}" / "predictions.jsonl"):
            r = json.loads(line); k = (r["dataset"], r["video_id"])
            arm[k] = np.asarray(r["score_curve"], float)
            key = float(r["extra"]["p_video_logodds"])
            keymax[k[0]] = max(keymax.get(k[0], -np.inf), key); nclamp[k[0]] = nclamp.get(k[0], 0) + int(key > CLAMP)
            dcell = max(dcell, float(np.abs(np.asarray(r["extra"]["cell_prob"]) - r6[k]["cell_prob"]).max()))
    log(f"M2 cell_prob vs r6_bma max abs diff {dcell:.2e}; key max " + ", ".join(f"{ds} {v:.2f}" for ds, v in keymax.items()))
    base = {k: r6[k]["score"] for k in arm}
    res = paired_bootstrap({"dvd_noE_r6": arm}, base, gts, a.B, a.seed)
    cli_m = {}
    for g in GROUPS:
        for p in json.load(open(OUT / "cli" / f"dvd_noE_r6_{g}" / "metrics.json"))["per_dataset"]:
            cli_m[p["dataset"]] = [p["frame_ROC_AUC"], p["frame_PR_AUC"], p["within_video_macro_ROC_AUC"]]
    for ds in datasets:                                 # the bootstrap point equals the CLI's own evaluator output
        if max(abs(x - y) for x, y in zip(res[ds]["arms"]["dvd_noE_r6"]["point"], cli_m[ds])) > 1e-12:
            raise SystemExit(f"{ds}: bootstrap point differs from metrics.json")
    L = [f"DVD verdict + T (dvd_noE) on the r6_bma time level; d = arm - r6_bma [95% paired video bootstrap, B={a.B}]; "
         f"T-removal ablation = r6_bma - arm; G1 per corpus (pooled -.005, within -.01); development-selected",
         f"M2 cell_prob vs r6_bma max abs diff {dcell:.1e}; calibrated key max " +
         ", ".join(f"{ds} {v:.2f} ({nclamp[ds]} keys > {CLAMP:g})" for ds, v in keymax.items())]
    out = {}
    for gname, r in res.items():
        x = r["arms"]["dvd_noE_r6"]; p, d = x["point"], x["delta"]; b = r["r6_bma"]
        G1 = d[0] >= -NOISE_POOLED and d[1] >= -NOISE_POOLED and d[2] >= -NOISE_WITHIN
        out[gname] = {"arm": p, "r6_bma": b, "delta": d, "ROC_ci95": x["ROC_ci95"], "PR_ci95": x["PR_ci95"],
                      "T_removal_ablation": [-v for v in d], "G1": G1, "n_videos": r["n_videos"]}
        L.append(f"{gname:19s} arm {fmt(p[0])} / {fmt(p[1])} / {fmt(p[2])}  r6 {fmt(b[0])} / {fmt(b[1])} / {fmt(b[2])}  "
                 f"d {fmt(d[0], True)} [{fmt(x['ROC_ci95'][0], True)},{fmt(x['ROC_ci95'][1], True)}] / {fmt(d[1], True)} "
                 f"[{fmt(x['PR_ci95'][0], True)},{fmt(x['PR_ci95'][1], True)}] / {fmt(d[2], True)}  G1 {'PASS' if G1 else 'FAIL'}"
                 + ("  [diagnostic]" if gname.endswith("_strict") else ""))
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    (OUT / "summary.json").write_text(json.dumps({"results": out, "key_max": keymax, "n_keys_above_27": nclamp,
                                                  "m2_cell_prob_max_abs_diff": dcell}, indent=1))
    print(txt)
    log("RUN_DONE")


if __name__ == "__main__":
    main()
