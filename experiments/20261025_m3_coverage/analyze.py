#!/usr/bin/env python3
"""Candidates 47 and 48 analysis in one command (CPU; reads the test GT for pre-checks and evaluation only, rule 10).

Input: one or more read_extra.py runs (reads.jsonl; verify rows inside the records or in verify.jsonl; config.json),
e.g. runs/20261025_m3_coverage/main (HateMM + HateClipSeg) and runs/20261025_m3_coverage/dehate (DeHate, numbers only).
Steps, as declared in README sections 2-4 (candidate 47) and 9 (candidate 48); the constants below are copied there:
  1. readouts per video and reading (direct, stance), from the six options' log probabilities normalised over the
     options (log space throughout):
       E[c] = sum_k P_k m_k, m = (0, .05, .175, .375, .625, .875); written as logit E[c] (log-space logsumexp)
       mode = m of argmax_k P_k, .01 when the mode is A; written as logit(mode)
       eta  = mean over j = 1..5 of l_j = log P(option >= j) - log P(option < j)   (options ordered A..F)
     -> values/<reading>_<readout>.jsonl (dataset, video_id, value), per_video.jsonl (no GT);
  2. P0 (no GT; main corpora gated, DeHate reported): options single tokens; on the verify videos, cached vs plain
     call: max |dP| < .01, |dE[c]| < .005, |d eta| < .5; reproduced z_video vs the r6 archive |dz| < .05 on every video;
     also the full-vocabulary option mass, whether the full-vocabulary top-1 token is an option, cache-vs-plain |dz|;
  3. pre-checks (GT) per readout: precheck.analyse (Spearman with GT coverage, partial Spearman given the window
     mean / z_video, video bootstrap); P1 (hateful videos: rho(readout, coverage) > rho(window mean, coverage)), P3
     (all videos: rho(readout, z_video) < .90), P2 (candidate 47 A1 / A1eta only: videos with P(V=1 | K_r6) > .5, video
     AUC of the readout for "has a GT-positive frame" >= the window mean's); rho(readout, coverage) on videos with
     z_video > 0; the readouts' distribution on videos without / with GT-positive frames, option entropy, mode counts;
  4. key evaluation through key_eval.py (subprocesses, in parallel; main corpora together, DeHate separately): per
     readout, share_raw (K = readout, no calibration) and replace --scale sd (K = z_video + sd(wm)/sd(readout) *
     readout, then the r6 two-Gaussian calibration through twolevel_r2.py --key calib; no DVD path, so no clamp);
  5. gates: G1 (no metric below r6_bma by more than pooled .005 / within .01, both corpora), G2 (pooled ROC or pooled
     PR >= r6_bma + .01 on both corpora).
       candidate 47 (archived for P3, README section 8; computed for the record): A1 -> A2 -> A1eta -> A2eta,
         an arm passes with P0, its readout's P1 and P3 (A1 / A1eta also P2), G1, G2;
       candidate 48 (README section 9): B1 (stance eta, replace) -> B2 (stance logit E[c], replace), an arm passes with
         P0, its readout's P1 and P3, G1, G2; ablations: r6 key (= the deltas), condition (direct reading, same form =
         candidate 47 A2eta / A2), readout (stance logit c_mode, same form).
     Every calibrated key above 27 is flagged (code review B2).
Output: runs/20261025_m3_coverage/<reads run names joined by _>_analysis/{summary.json, summary.txt, per_video.jsonl,
values/, precheck/<readout>/, keyeval/<group>/<readout>/, run.log, config.json}.

  python experiments/20261025_m3_coverage/analyze.py --reads runs/20261025_m3_coverage/main [runs/.../dehate] [--B 1000]
  python experiments/20261025_m3_coverage/analyze.py --selftest [--selftest-datasets HateMM HateClipSeg DeHate]
      (fake reads whose direct and stance logit E[c] are affine maps of the window mean: A2 and B2 must reproduce
       r6_bma, P1 must tie with the window mean and fail, P2 must tie and pass, P0 must pass)
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
from scipy.special import expit, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CORPORA, DATASETS, EXTERNAL, OUT_ROOT, ROOT, ci, coverage, gt_sets, load_r6, load_reads,  # noqa: E402
                    spearman, start_run, video_auc, write_values)
from precheck import analyse, table  # noqa: E402

# ---- README sections 2-4 and 9 constants
MID = np.array([0.0, 0.05, 0.175, 0.375, 0.625, 0.875])
MODE_A = 0.01
OPTIONS = ["A", "B", "C", "D", "E", "F"]
P0_DP, P0_DEC, P0_DETA, P0_DZ = 0.01, 0.005, 0.5, 0.05
P3_MAX = 0.90
NOISE_POOLED, NOISE_WITHIN, G2_GAIN = 0.005, 0.01, 0.01
KEY_WARN = 27.0   # twolevel_r2.py's DVD path clamps at logit 27.6 (min(lp, -1e-12)); flag any key above 27 (review B2)
DECLARED_WM = {"P1": {"HateMM": .416, "HateClipSeg": .533}, "P2": {"HateMM": .821, "HateClipSeg": .710}}
# (name, readout, key_eval form, P2 required)
CAND47 = [("A1", "direct_logitEc", "share_raw", True), ("A2", "direct_logitEc", "replace", False),
          ("A1eta", "direct_eta", "share_raw", True), ("A2eta", "direct_eta", "replace", False)]
CAND48 = [("B1", "stance_eta", "replace", False), ("B2", "stance_logitEc", "replace", False)]
# (name, readout, form, what it is)
CAND48_ABL = [("B1_cond_direct", "direct_eta", "replace", "condition ablation of B1: direct reading (= 47 A2eta)"),
              ("B2_cond_direct", "direct_logitEc", "replace", "condition ablation of B2: direct reading (= 47 A2)"),
              ("B_readout_mode", "stance_logitMode", "replace", "readout ablation: stance logit c_mode")]
CAND47_EXTRA = [("A1_mode", "direct_logitMode", "share_raw", "readout ablation of A1"),
                ("A2_mode", "direct_logitMode", "replace", "readout ablation of A2")]
KEYEVAL_READOUTS = ("direct_logitEc", "direct_eta", "direct_logitMode", "stance_logitEc", "stance_eta", "stance_logitMode")
METRICS = ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC")


# ----------------------------------------------------------------------------------------------- readouts

def norm_logp(logp):
    lp = np.array([logp[o] for o in OPTIONS], float)
    return lp - logsumexp(lp)


def readouts(lp):
    """lp: the six options' log probabilities normalised over the options (order A..F)."""
    log_e = logsumexp(lp[1:] + np.log(MID[1:]))
    log_1me = logsumexp(lp + np.log1p(-MID))
    ell = [logsumexp(lp[j:]) - logsumexp(lp[:j]) for j in range(1, 6)]
    k = int(np.argmax(lp)); mode = MODE_A if k == 0 else float(MID[k])
    P = np.exp(lp)
    return {"Ec": float(np.exp(log_e)), "logitEc": float(log_e - log_1me), "eta": float(np.mean(ell)),
            "mode_option": OPTIONS[k], "mode_mid": mode, "logitMode": float(math.log(mode) - math.log1p(-mode)),
            "max_p": float(P.max()), "entropy": float(-(P * lp).sum())}


def load_runs(paths):
    """Merge one or more read_extra.py runs: records without error, error count, configs, verify rows."""
    recs, cfgs, ver, n_err, names = {}, {}, {}, 0, []
    for rp in paths:
        rp = Path(rp)
        d = rp if rp.is_dir() else rp.parent
        f = d / "reads.jsonl" if rp.is_dir() else rp
        names.append(d.name)
        cfg = json.load(open(d / "config.json")) if (d / "config.json").exists() else {}
        for line in open(f):
            r = json.loads(line)
            k = (r["dataset"], r["video_id"])
            if r.get("error"):
                n_err += 1; continue
            if k in recs:
                raise SystemExit(f"{k} appears in more than one reads file")
            recs[k] = r; cfgs[k[0]] = cfg
        if (d / "verify.jsonl").exists():
            for line in open(d / "verify.jsonl"):
                v = json.loads(line); ver[(v["dataset"], v["video_id"])] = v
    for k, r in recs.items():
        if "verify" in r:
            ver[k] = {**ver.get(k, {}), **r["verify"]}
    return names, recs, n_err, cfgs, ver


# ----------------------------------------------------------------------------------------------- P0

def p0(recs, cfgs, ver, datasets):
    out = {"checks": {}, "stats": {}}
    modes = sorted({m for r in recs.values() for m in r["reads"]})
    for ds in datasets:
        rep = cfgs.get(ds, {}).get("option_tokens")
        union = {i for x in rep for i in x["ids"]} if rep else None
        if rep:
            single = all(len(x["ids"]) >= 1 and x["label"] in x.get("forms", [x["label"]]) for x in rep) and \
                [x["label"] for x in rep] == OPTIONS
            out["checks"][f"{ds}_options_single_token"] = {"pass": bool(single)}
        else:
            out["checks"][f"{ds}_options_single_token"] = {"pass": None, "detail": "config.json without option_tokens"}
        ks = [k for k in recs if k[0] == ds]
        dz = np.array([recs[k]["dz_r6"] for k in ks if recs[k].get("dz_r6") is not None], float)
        st = {"n_videos": len(ks), "dz_r6_median": float(np.median(dz)) if len(dz) else None,
              "dz_r6_max": float(dz.max()) if len(dz) else None, "n_dz_r6_ge_gate": int((dz >= P0_DZ).sum()),
              "n_without_r6": len(ks) - len(dz)}
        for m in modes:
            kk = [k for k in ks if m in recs[k]["reads"]]
            mass = np.exp([recs[k]["reads"][m]["option_mass_full_vocab_log"] for k in kk])
            top = [recs[k]["reads"][m]["top_token_id"] in union if union is not None
                   else recs[k]["reads"][m]["top_token"].strip().upper() in OPTIONS for k in kk]
            st[m] = {"option_mass_full_vocab": {"min": float(mass.min()), "p10": float(np.percentile(mass, 10)),
                                                "median": float(np.median(mass))} if len(mass) else None,
                     "share_top1_is_option": float(np.mean(top)) if top else None}
        vrows = []
        for k in sorted(x for x in ver if x[0] == ds):
            v = ver[k]; row = {"video_id": k[1], "dz_video_cache_vs_plain": v.get("dz_video_cache_vs_plain")}
            for m in modes:
                if f"logp_plain_{m}" not in v or k not in recs or m not in recs[k]["reads"]:
                    continue
                a = norm_logp(recs[k]["reads"][m]["logp"]); b = norm_logp(v[f"logp_plain_{m}"])
                ra, rb = readouts(a), readouts(b)
                row[m] = {"max_dP": float(np.abs(np.exp(a) - np.exp(b)).max()), "dEc": abs(ra["Ec"] - rb["Ec"]),
                          "deta": abs(ra["eta"] - rb["eta"]), "seam_ok": v.get(f"seam_ok_{m}")}
            vrows.append(row)
        st["verify"] = vrows
        out["stats"][ds] = st
        ok_v = bool(vrows) and all(any(m in r for m in modes) and all(
            r[m]["max_dP"] < P0_DP and r[m]["dEc"] < P0_DEC and r[m]["deta"] < P0_DETA and r[m]["seam_ok"] is not False
            for m in modes if m in r) for r in vrows)
        out["checks"][f"{ds}_verify_cached_vs_plain"] = {"pass": ok_v, "n_verify_videos": len(vrows)}
        out["checks"][f"{ds}_z_video_vs_r6_archive"] = {"pass": bool(len(dz) == len(ks) and (dz < P0_DZ).all()),
                                                       "max": st["dz_r6_max"]}
    gated = [c for c in out["checks"] if not any(c.startswith(ds + "_") for ds in EXTERNAL)]
    out["gated_checks"] = gated
    out["pass"] = bool(gated) and all(out["checks"][c]["pass"] is True for c in gated)
    return out


# ----------------------------------------------------------------------------------------------- pre-checks

def distributions(per, gts, r6, datasets):
    """Readouts on videos without / with a GT-positive frame (protocol GT), entropy, max p and mode counts."""
    out = {}
    for ds in datasets:
        Y = gts[ds][1]
        for m in sorted({m for p in per.values() for m in p["reads"]}):
            ks = [k for k in per if k[0] == ds and k in r6 and k[1] in Y and m in per[k]["reads"]]
            grp = {"no_positive_frame": [k for k in ks if coverage(Y[k[1]], len(r6[k]["score"])) == 0],
                   "has_positive_frame": [k for k in ks if coverage(Y[k[1]], len(r6[k]["score"])) > 0]}
            o = {}
            for g, kk in grp.items():
                o[g] = {"n": len(kk)}
                for q in ("Ec", "eta", "entropy"):
                    x = np.array([per[k]["reads"][m][q] for k in kk], float)
                    o[g][q] = {"mean": float(x.mean()), **{f"p{p}": float(np.percentile(x, p)) for p in (10, 25, 50, 75, 90)}} \
                        if len(x) else None
                o[g]["share_max_p_gt_.99"] = float(np.mean([per[k]["reads"][m]["max_p"] > .99 for k in kk])) if kk else None
                o[g]["mode_counts"] = {opt: sum(per[k]["reads"][m]["mode_option"] == opt for k in kk) for opt in OPTIONS}
            out[f"{ds}/{m}"] = o
    return out


def p2_auc(values, gts, r6, reads, ds):
    Y = gts[ds][1]
    ks = [k for k in sorted(r6) if k[0] == ds and k[1] in Y and k in values and r6[k]["lo"] > 0]
    lab = [int(coverage(Y[k[1]], len(r6[k]["score"])) > 0) for k in ks]
    return {"n": len(ks), "n_positive": int(sum(lab)),
            "auc_readout": video_auc([values[k] for k in ks], lab) if ks else float("nan"),
            "auc_wm": video_auc([reads[k]["wm"] for k in ks], lab) if ks else float("nan")}


def rho_zv_positive(values, gts, r6, reads, ds, B, rng):
    """rho(readout, coverage) and rho(window mean, coverage) on the videos with z_video > 0, with a video bootstrap."""
    Y = gts[ds][1]
    ks = [k for k in sorted(r6) if k[0] == ds and k[1] in Y and k in values and r6[k]["zv"] > 0]
    v = np.array([values[k] for k in ks]); w = np.array([reads[k]["wm"] for k in ks])
    c = np.array([coverage(Y[k[1]], len(r6[k]["score"])) for k in ks])
    out = {"n": len(ks), "rho_readout": spearman(v, c), "rho_wm": spearman(w, c)}
    if len(ks) >= 3:
        bs = []
        for _ in range(B):
            i = rng.integers(0, len(ks), len(ks)); bs.append((spearman(v[i], c[i]), spearman(w[i], c[i])))
        bs = np.array(bs)
        out.update({"rho_readout_ci95": ci(bs[:, 0]), "rho_wm_ci95": ci(bs[:, 1])})
    else:
        out.update({"rho_readout_ci95": [float("nan")] * 2, "rho_wm_ci95": [float("nan")] * 2})
    return out


def gate(x, y, op):
    if x is None or y is None or not (np.isfinite(x) and np.isfinite(y)):
        return None
    return bool(x > y) if op == ">" else bool(x >= y) if op == ">=" else bool(x < y)


def prechecks(values_by, gts, r6, reads, datasets, B, seed, out, log):
    res = {}
    rng = np.random.default_rng(seed)
    for name, values in values_by.items():
        r = analyse(values, {g: v for g, v in gts.items() if v[0] in datasets}, r6, reads, B, seed)
        d = out / "precheck" / name; d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(r, indent=1))
        (d / "summary.txt").write_text(table(r, f"precheck {name} (analyze.py); video bootstrap B={B}") + "\n")
        res[name] = {"precheck": r, "P": {}}
        for ds in datasets:
            h, a = r[ds]["hateful"], r[ds]["all"]
            p2 = p2_auc(values, gts, r6, reads, ds)
            pt = lambda x, k: x[k]["point"]
            res[name]["P"][ds] = {
                "P1": {"rho_readout": pt(h, "rho_value_cov"), "rho_readout_ci95": h["rho_value_cov"]["ci95"],
                       "rho_wm": pt(h, "rho_wm_cov"), "declared_wm": DECLARED_WM["P1"].get(ds), "n": h["n"],
                       "diff_ci95": h["diff_value_minus_wm"]["ci95"],
                       "pass": gate(pt(h, "rho_value_cov"), pt(h, "rho_wm_cov"), ">")},
                "P3": {"rho_readout_zvideo": pt(a, "rho_value_zv"), "ci95": a["rho_value_zv"]["ci95"],
                       "threshold": P3_MAX, "n": a["n"], "pass": gate(pt(a, "rho_value_zv"), P3_MAX, "<")},
                "P2": {**p2, "declared_wm": DECLARED_WM["P2"].get(ds), "pass": gate(p2["auc_readout"], p2["auc_wm"], ">=")},
                "partial_given_wm_hateful": pt(h, "partial_value_cov_given_wm"),
                "partial_given_wm_hateful_ci95": h["partial_value_cov_given_wm"]["ci95"],
                "partial_given_zv_hateful": pt(h, "partial_value_cov_given_zv"),
                "partial_given_zv_hateful_ci95": h["partial_value_cov_given_zv"]["ci95"],
                "partial_given_zv_wm_hateful": pt(h, "partial_value_cov_given_zv_wm"),
                "zv_positive": rho_zv_positive(values, gts, r6, reads, ds, B, rng)}
        log(f"precheck {name} done")
    return res


# ----------------------------------------------------------------------------------------------- key evaluation

def launch_keyevals(vfiles, groups, B, seed, out, log):
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    jobs = {}
    for g, dss in groups.items():
        for rd in KEYEVAL_READOUTS:
            if rd not in vfiles:
                continue
            d = out / "keyeval" / g / rd; d.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / "experiments/20261025_m3_coverage/key_eval.py"), "--values", str(vfiles[rd]),
                   "--name", rd, "--forms", "share_raw", "replace", "--transform", "none", "--scale", "sd",
                   "--datasets", *dss, "--B", str(B), "--seed", str(seed), "--out", str(d)]
            log(f"key_eval {g}/{rd}: {' '.join(cmd[1:])}")
            jobs[(g, rd)] = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(d / "stdout.log", "w"), stderr=subprocess.STDOUT)
    return jobs


def wait_keyevals(jobs, out, log):
    res = {}
    for (g, rd), p in jobs.items():
        if p.wait() != 0:
            raise SystemExit(f"key_eval {g}/{rd} FAILED (exit {p.returncode}); see {out / 'keyeval' / g / rd / 'stdout.log'}")
        res[(g, rd)] = json.load(open(out / "keyeval" / g / rd / "summary.json"))
        log(f"key_eval {g}/{rd} done")
    return res


def arm_numbers(ke, form, datasets, gnames):
    """Evaluator point metrics of the arm and of r6_bma on the same videos, plus bootstrap intervals, per GT set."""
    if ke is None or form not in ke["forms"]:
        return None
    out = {}
    for ds in datasets:
        a, b = ke["evaluator_metrics"][form][ds], ke["evaluator_metrics"]["r6_bma"][ds]
        bt = ke["bootstrap"].get(ds, {}).get("delta", {}).get(form, {})
        out[ds] = {"arm": [a[m] for m in METRICS], "r6": [b[m] for m in METRICS],
                   "delta": [None if a[m] is None or b[m] is None else a[m] - b[m] for m in METRICS],
                   "ROC_ci95": bt.get("ROC_ci95"), "PR_ci95": bt.get("PR_ci95")}
    for gname in gnames:
        if gname in ke["bootstrap"] and gname not in out:
            bt = ke["bootstrap"][gname]
            out[gname] = {"arm": bt["point"][form][:3], "r6": bt["point"]["r6_bma"][:3],
                          "delta": [bt["delta"][form]["dROC"], bt["delta"][form]["dPR"], bt["delta"][form]["dWithin"]],
                          "ROC_ci95": bt["delta"][form]["ROC_ci95"], "PR_ci95": bt["delta"][form]["PR_ci95"]}
    return out


def key_max(ke, form):
    return None if ke is None or form not in ke.get("key_max", {}) else ke["key_max"][form]


def g_gates(nums, main):
    if nums is None or any(ds not in nums for ds in main):
        return {"G1": None, "G2": None}
    g1 = {}
    for ds in main:
        d = nums[ds]["delta"]
        g1[ds] = [gate(d[0], -NOISE_POOLED, ">="), gate(d[1], -NOISE_POOLED, ">="), gate(d[2], -NOISE_WITHIN, ">=")]
    G1 = None if any(x is None for v in g1.values() for x in v) else all(x for v in g1.values() for x in v)
    roc = [gate(nums[ds]["delta"][0], G2_GAIN - 1e-12, ">=") for ds in main]
    pr = [gate(nums[ds]["delta"][1], G2_GAIN - 1e-12, ">=") for ds in main]
    G2 = None if None in roc + pr else (all(roc) or all(pr))
    return {"G1": G1, "G1_detail": g1, "G2": G2, "G2_roc_both": all(roc) if None not in roc else None,
            "G2_pr_both": all(pr) if None not in pr else None}


def allp(xs):
    return None if (not xs or None in xs) else all(xs)


def gated_arms(spec, pre, ke, P0, main, ext, strict):
    arms = {}
    for name, rd, form, need_p2 in spec:
        P = pre.get(rd, {}).get("P", {})
        pP1 = [P.get(ds, {}).get("P1", {}).get("pass") for ds in main]
        pP3 = [P.get(ds, {}).get("P3", {}).get("pass") for ds in main]
        pP2 = [P.get(ds, {}).get("P2", {}).get("pass") for ds in main] if need_p2 else []
        nums = arm_numbers(ke.get(("main", rd)), form, main, strict)
        g = g_gates(nums, main)
        conds = {"P0": P0["pass"], "P1": allp(pP1), "P3": allp(pP3), **({"P2": allp(pP2)} if need_p2 else {}),
                 "G1": g["G1"], "G2": g["G2"]}
        arms[name] = {"readout": rd, "form": form, **conds, "P2_required": need_p2, "G1_detail": g.get("G1_detail"),
                      "G2_roc_both": g.get("G2_roc_both"), "G2_pr_both": g.get("G2_pr_both"),
                      "failing": [c for c, v in conds.items() if v is False],
                      "pass": None if None in conds.values() else all(conds.values()), "numbers": nums,
                      "external": arm_numbers(ke.get(("external", rd)), form, ext, []),
                      "key_max": {**(key_max(ke.get(("main", rd)), form) or {}),
                                  **(key_max(ke.get(("external", rd)), form) or {})}}
    return arms


def plain_arms(spec, ke, main, ext, strict):
    return {name: {"readout": rd, "form": form, "what": what, "numbers": arm_numbers(ke.get(("main", rd)), form, main, strict),
                   "external": arm_numbers(ke.get(("external", rd)), form, ext, []),
                   "key_max": {**(key_max(ke.get(("main", rd)), form) or {}), **(key_max(ke.get(("external", rd)), form) or {})}}
            for name, rd, form, what in spec}


def decide(arms, P0, main, label):
    sel = next((a for a, x in arms.items() if x["pass"]), None)
    first = next(iter(arms))
    if not main:
        v = "no main corpus in the reads: gates not evaluated"
    elif not P0["pass"]:
        v = "P0 FAIL: pipeline check failed; fix the code (not a candidate failure); gates are for the record only"
    elif sel == first:
        v = f"{sel} (declared first arm) passes its pre-checks and G1, G2"
    elif sel:
        v = f"{sel} passes its pre-checks and G1, G2; selected among the {label} declared arms by test results (development-selected)"
    elif any(x["pass"] is None for x in arms.values()):
        v = "undetermined: some conditions could not be evaluated (n/a)"
    else:
        v = "no declared arm passes: archive per rule 12 (" + "; ".join(f"{a}: {', '.join(x['failing'])}" for a, x in arms.items()) + ")"
    return sel, v


# ----------------------------------------------------------------------------------------------- formatting

def tri(x):
    return "n/a" if x is None else ("PASS" if x else "FAIL")


def f4(x, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.4f}" if sign else f"{x:.4f}"


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.3f}"


def arm_lines(name, x, gated):
    head = f"  {name:15s} {x['readout']:17s} {x['form']:9s}"
    if gated:
        head += " " + " ".join(f"{c} {tri(x[c])}" for c in ("P0", "P1", "P3") + (("P2",) if x["P2_required"] else ()) + ("G1", "G2"))
        head += f" -> {tri(x['pass'])}"
    else:
        head += f" ({x['what']})"
    head += "  max key " + (", ".join(f"{ds} {v:.2f}" for ds, v in x["key_max"].items()) or "n/a")
    L = [head]
    for part in ("numbers", "external"):
        for gname, n in (x[part] or {}).items():
            rc, pc = n["ROC_ci95"] or [None, None], n["PR_ci95"] or [None, None]
            L.append(f"      {gname:19s} {f4(n['arm'][0])} / {f4(n['arm'][1])} / {f4(n['arm'][2])}   d {f4(n['delta'][0], True)}"
                     f" [{f4(rc[0], True)},{f4(rc[1], True)}] / {f4(n['delta'][1], True)} [{f4(pc[0], True)},{f4(pc[1], True)}]"
                     f" / {f4(n['delta'][2], True)}   (r6 {f4(n['r6'][0])} / {f4(n['r6'][1])} / {f4(n['r6'][2])})"
                     + ("  [external, report only]" if part == "external" else "")
                     + ("  [diagnostic]" if gname.endswith("_strict") else ""))
    return L


def render(s, ext):
    c47, c48 = s["candidate_47"], s["candidate_48"]
    L = [f"candidates 47 / 48 analysis: {s['reads']}  ({s['n_records']} records, {s['n_error_records']} errors; "
         f"corpora {s['datasets']}); development-selected; HateClipSeg_strict = hateful-only GT (diagnostic)",
         f"\nCANDIDATE 48 VERDICT: {c48['verdict']}   (order {' -> '.join(c48['selection_order'])}; selected: {c48['selected']})",
         f"CANDIDATE 47 (record only, {c47['status']}): {c47['verdict']}   (order {' -> '.join(c47['selection_order'])}; "
         f"would-be selected: {c47['selected']})"]
    for w in s["key_max_warnings"]:
        L.append(f"WARNING (key above {KEY_WARN:g}; twolevel_r2.py DVD-path clamp at 27.6): {w}")
    L.append(f"\n== P0 (no GT; main corpora gated): {tri(s['P0']['pass'])}")
    for c, x in s["P0"]["checks"].items():
        L.append(f"  {c:40s} {tri(x['pass'])}{'' if c in s['P0']['gated_checks'] else ' (external, report only)'}  "
                 + ", ".join(f"{k} {v}" for k, v in x.items() if k not in ("pass", "detail")))
    for ds, st in s["P0"]["stats"].items():
        L.append(f"  {ds}: n {st['n_videos']}, |dz_r6| median {f4(st['dz_r6_median'])} max {f4(st['dz_r6_max'])} "
                 f"(>= {P0_DZ}: {st['n_dz_r6_ge_gate']})")
        for m in ("direct", "stance"):
            if m in st and st[m]["option_mass_full_vocab"]:
                om = st[m]["option_mass_full_vocab"]
                L.append(f"    {m:6s} option mass (full vocab) min {om['min']:.4f} p10 {om['p10']:.4f} median {om['median']:.4f}; "
                         f"top-1 is an option {st[m]['share_top1_is_option']:.3f}")
        for r in st["verify"]:
            L.append(f"    verify {r['video_id']}: dz cache-vs-plain {f4(r['dz_video_cache_vs_plain'])}  " + "  ".join(
                f"{m}: max|dP| {r[m]['max_dP']:.2e} |dE[c]| {r[m]['dEc']:.2e} |d eta| {r[m]['deta']:.2e} seam {r[m]['seam_ok']}"
                for m in ("direct", "stance") if m in r))
    L.append("\n== pre-checks (protocol GT; P1 hateful videos vs the window mean, P3 all videos, P2 videos with "
             "P(V=1|K_r6) > .5; partial Spearman on hateful videos; zv>0 = videos with z_video > 0)")
    for rd, P in s["prechecks"].items():
        for ds, x in P.items():
            tag = " (report only)" if ds in ext else ""
            z = x["zv_positive"]
            L.append(f"  {rd:18s} {ds:12s}{tag} P1 {f3(x['P1']['rho_readout'])} [{f3(x['P1']['rho_readout_ci95'][0])},"
                     f"{f3(x['P1']['rho_readout_ci95'][1])}] vs wm {f3(x['P1']['rho_wm'])} {tri(x['P1']['pass'])} | "
                     f"P3 {f3(x['P3']['rho_readout_zvideo'])} {tri(x['P3']['pass'])} | "
                     f"P2 {f3(x['P2']['auc_readout'])} vs wm {f3(x['P2']['auc_wm'])} (n {x['P2']['n']}) {tri(x['P2']['pass'])}")
            L.append(f"  {'':18s} {'':12s}{tag} partial|wm {f3(x['partial_given_wm_hateful'])} "
                     f"[{f3(x['partial_given_wm_hateful_ci95'][0])},{f3(x['partial_given_wm_hateful_ci95'][1])}]  "
                     f"partial|z_video {f3(x['partial_given_zv_hateful'])} [{f3(x['partial_given_zv_hateful_ci95'][0])},"
                     f"{f3(x['partial_given_zv_hateful_ci95'][1])}]  zv>0 (n {z['n']}) rho {f3(z['rho_readout'])} "
                     f"[{f3(z['rho_readout_ci95'][0])},{f3(z['rho_readout_ci95'][1])}] vs wm {f3(z['rho_wm'])}")
    L.append("\n== readout distributions (protocol GT groups): E[c] median [p10, p90], eta median, entropy median, "
             "share max p > .99, mode counts A..F")
    for key, o in s["distributions"].items():
        for g, x in o.items():
            if not x["n"]:
                continue
            L.append(f"  {key:22s} {g:19s} n {x['n']:4d}  E[c] {x['Ec']['p50']:.3f} [{x['Ec']['p10']:.3f}, {x['Ec']['p90']:.3f}]  "
                     f"eta {x['eta']['p50']:+.2f}  H {x['entropy']['p50']:.3f}  p>.99 {x['share_max_p_gt_.99']:.2f}  "
                     f"modes {list(x['mode_counts'].values())}")
    L.append("\n== candidate 48 arms (point = evaluator metrics on the same videos as r6_bma; d = arm - r6_bma = the 14g "
             "ablation 'key back to r6' with the sign reversed; 95% paired video bootstrap)")
    for name, x in c48["arms"].items():
        L += arm_lines(name, x, True)
    L.append("  ablations:")
    for name, x in c48["ablations"].items():
        L += arm_lines(name, x, False)
    L.append(f"\n== candidate 47 arms (record only; {c47['status']})")
    for name, x in c47["arms"].items():
        L += arm_lines(name, x, True)
    for name, x in c47["extra"].items():
        L += arm_lines(name, x, False)
    L.append("\nnot checked here: rule 4 novelty review, rule 6 code review (README sections 4 and 9.4)")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------- main

def run(reads_paths, datasets_req, B, seed, out=None, selftest=False):
    names, recs, n_err, cfgs, ver = load_runs(reads_paths)
    present = sorted({k[0] for k in recs})
    datasets = [ds for ds in (datasets_req or present) if ds in present]
    main = [ds for ds in DATASETS if ds in datasets]; ext = [ds for ds in EXTERNAL if ds in datasets]
    out = Path(out) if out else OUT_ROOT / f"{'_'.join(names)}_analysis"
    log = start_run(out, "experiments/20261025_m3_coverage/analyze.py",
                    {"reads": [str(p) for p in reads_paths], "datasets": datasets, "B": B, "seed": seed, "selftest": selftest,
                     "constants": {"MID": MID.tolist(), "MODE_A": MODE_A, "P0": [P0_DP, P0_DEC, P0_DETA, P0_DZ], "P3": P3_MAX,
                                   "noise": [NOISE_POOLED, NOISE_WITHIN], "G2": G2_GAIN, "key_warn": KEY_WARN}})
    log(f"reads {[str(p) for p in reads_paths]}: {len(recs)} records without error, {n_err} error records; corpora {datasets}")
    if any(r["options"] != OPTIONS for r in recs.values()):
        raise SystemExit(f"options are not {OPTIONS}")
    # 1. readouts
    per = {k: {"reads": {m: readouts(norm_logp(x["logp"])) for m, x in r["reads"].items()}}
           for k, r in recs.items() if k[0] in datasets}
    with open(out / "per_video.jsonl", "w") as fh:
        for (ds, v), p in sorted(per.items()):
            fh.write(json.dumps({"dataset": ds, "video_id": v, "dz_r6": recs[(ds, v)].get("dz_r6"), **p["reads"]}) + "\n")
    modes = sorted({m for p in per.values() for m in p["reads"]})
    values_by, vfiles = {}, {}
    for m in modes:
        for q in ("logitEc", "eta", "logitMode"):
            name = f"{m}_{q}"
            values_by[name] = {k: p["reads"][m][q] for k, p in per.items() if m in p["reads"]}
            vfiles[name] = out / "values" / f"{name}.jsonl"; write_values(vfiles[name], values_by[name])
    # 2. P0
    P0 = p0({k: recs[k] for k in per}, cfgs, ver, datasets)
    log(f"P0 {'PASS' if P0['pass'] else 'FAIL'}: " + ", ".join(f"{c} {tri(x['pass'])}" for c, x in P0["checks"].items()))
    # 4. key evaluation in the background, 3. pre-checks meanwhile
    gts = gt_sets(tuple(datasets))
    r6 = load_r6(datasets=tuple(datasets)); reads = load_reads(datasets=tuple(datasets))
    groups = {g: dss for g, dss in (("main", main), ("external", ext)) if dss}
    jobs = launch_keyevals(vfiles, groups, B, seed, out, log)
    pre = prechecks(values_by, gts, r6, reads, datasets, B * 2, seed, out, log)
    dist = distributions(per, gts, r6, datasets)
    ke = wait_keyevals(jobs, out, log)
    # 5. gates
    strict = ["HateClipSeg_strict"] if "HateClipSeg" in main else []
    a48 = gated_arms(CAND48, pre, ke, P0, main, ext, strict)
    a47 = gated_arms(CAND47, pre, ke, P0, main, ext, strict)
    s48, v48 = decide(a48, P0, main, "two")
    s47, v47 = decide(a47, P0, main, "four")
    abl48 = plain_arms(CAND48_ABL, ke, main, ext, strict)
    ext47 = plain_arms(CAND47_EXTRA, ke, main, ext, strict)
    warns = [f"{a} ({x['readout']}, {x['form']}) {ds}: max key {v:.2f}"
             for group in (a48, abl48, a47, ext47) for a, x in group.items() for ds, v in x["key_max"].items() if v > KEY_WARN]
    for w in warns:
        log(f"WARNING key above {KEY_WARN:g}: {w}")
    summary = {"reads": [str(p) for p in reads_paths], "datasets": datasets, "n_records": len(recs), "n_error_records": n_err,
               "P0": P0, "prechecks": {rd: x["P"] for rd, x in pre.items()}, "distributions": dist,
               "candidate_48": {"arms": a48, "ablations": abl48, "selection_order": [a[0] for a in CAND48],
                                "selected": s48, "verdict": v48},
               "candidate_47": {"status": "archived for P3 (README section 8, item 37)", "arms": a47, "extra": ext47,
                                "selection_order": [a[0] for a in CAND47], "selected": s47, "verdict": v47},
               "key_max_warnings": warns, "not_checked_here": ["rule 4 novelty review", "rule 6 code review"],
               "B": B, "seed": seed}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    txt = render(summary, ext)
    (out / "summary.txt").write_text(txt + "\n")
    print(txt)
    log(f"CANDIDATE 48 VERDICT {v48}")
    log(f"CANDIDATE 47 (record) {v47}")
    return summary, out, log


# ----------------------------------------------------------------------------------------------- self-test

def fake_reads(out, datasets):
    """Fake read_extra.py output: direct E[c] = expit(.05 wm - 1) and stance E[c] = expit(.04 wm - .5), exactly (two
    adjacent options plus 1e-6 on the others); z_video and dz_r6 from the r6 reads; verify rows on the first 2 videos
    per corpus with a 1e-7 perturbation of the log probabilities."""
    reads = load_reads(datasets=datasets)
    out.mkdir(parents=True, exist_ok=True)
    eps = 1e-6

    def dist(c):
        k = int(np.clip(np.searchsorted(MID, c, side="right") - 1, 0, 4))
        others = [j for j in range(6) if j not in (k, k + 1)]
        rest = 1 - 4 * eps; target = c - eps * MID[others].sum()
        pk1 = (target - rest * MID[k]) / (MID[k + 1] - MID[k]); pk = rest - pk1
        P = np.full(6, eps); P[k], P[k + 1] = pk, pk1
        assert (P > 0).all() and abs((P * MID).sum() - c) < 1e-12, c
        return {o: float(math.log(p)) for o, p in zip(OPTIONS, P)}

    cfg = {"option_tokens": [{"label": o, "ids": [9000 + i], "forms": [o], "decoded": [o]} for i, o in enumerate(OPTIONS)],
           "modes": ["stance", "direct"], "note": "analyze.py --selftest fake reads"}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    rng = np.random.default_rng(0)
    first = {}
    with open(out / "reads.jsonl", "w") as fh, open(out / "verify.jsonl", "w") as vf:
        for (ds, v), r in sorted(reads.items()):
            rd = {}
            for m, c in (("direct", float(expit(0.05 * r["wm"] - 1))), ("stance", float(expit(0.04 * r["wm"] - 0.5)))):
                lp = dist(c); k = max(lp, key=lp.get)
                rd[m] = {"logp": lp, "option_mass_full_vocab_log": math.log(0.98), "top_token": k,
                         "top_token_id": 9000 + OPTIONS.index(k), "top_token_logp_full_vocab": lp[k], "branch_tokens": 60}
            rec = {"dataset": ds, "video_id": v, "error": None, "options": OPTIONS, "modes": ["stance", "direct"],
                   "z_video": r["zv"], "reads": rd, "z_video_r6": r["zv"], "dz_r6": 0.0}
            first[ds] = first.get(ds, 0) + 1
            if first[ds] <= 2:
                ver = {"dz_video_cache_vs_plain": 0.0}
                for m in rd:
                    ver[f"seam_ok_{m}"] = True
                    ver[f"logp_plain_{m}"] = {o: x + 1e-7 * float(rng.normal()) for o, x in rd[m]["logp"].items()}
                    ver[f"max_abs_dlogp_{m}"] = max(abs(ver[f"logp_plain_{m}"][o] - rd[m]["logp"][o]) for o in OPTIONS)
                rec["verify"] = ver
                vf.write(json.dumps({"dataset": ds, "video_id": v, **ver}) + "\n")
            fh.write(json.dumps(rec) + "\n")
    return out / "reads.jsonl"


def selftest(a):
    rp = fake_reads(OUT_ROOT / "selftest_fake_reads", a.selftest_datasets)
    s, out, log = run([rp], None, a.B, a.seed, a.out, selftest=True)
    ok = {"P0 pass": s["P0"]["pass"] is True}
    r6 = load_r6(datasets=tuple(a.selftest_datasets)); reads = load_reads(datasets=tuple(a.selftest_datasets))
    for g in ("main", "external"):
        for rd, arm in (("direct_logitEc", "A2"), ("stance_logitEc", "B2")):
            d = out / "keyeval" / g / rd
            if not d.exists():
                continue
            ke = json.load(open(d / "summary.json"))
            sc = {(r["dataset"], r["video_id"]): np.asarray(r["score_curve"]) for r in
                  map(json.loads, open(d / "arms" / f"{rd}__replace" / "predictions.jsonl"))}
            dmax = max(float(np.abs(sc[k] - r6[k]["score"]).max()) for k in sc)
            dmet = max(abs(ke["evaluator_metrics"]["replace"][ds][m] - ke["r6_metrics_file"][ds][m])
                       for ds in ke["datasets"] for m in METRICS)
            ok[f"{g}: {arm} on an affine map of the window mean reproduces r6_bma (frame {dmax:.1e}, metrics {dmet:.1e})"] = \
                dmax < 1e-6 and dmet < 1e-6 and len(sc) == sum(k[0] in ke["datasets"] for k in r6)
    for ds in [x for x in DATASETS if x in a.selftest_datasets]:
        for rd in ("direct_logitEc", "stance_logitEc"):
            P = s["prechecks"][rd][ds]
            ok[f"{ds} {rd}: P1 rho ties with the window mean ({P['P1']['rho_readout']:.6f} vs {P['P1']['rho_wm']:.6f}) "
               f"and fails (strict >)"] = abs(P["P1"]["rho_readout"] - P["P1"]["rho_wm"]) < 1e-12 and P["P1"]["pass"] is False
            z = P["zv_positive"]
            ok[f"{ds} {rd}: rho on z_video > 0 ties with the window mean ({z['rho_readout']:.6f}, n {z['n']})"] = \
                abs(z["rho_readout"] - z["rho_wm"]) < 1e-12
            ok[f"{ds} {rd}: partial|wm = 0 ({P['partial_given_wm_hateful']:.1e})"] = abs(P["partial_given_wm_hateful"]) < 1e-9
        P2 = s["prechecks"]["direct_logitEc"][ds]["P2"]
        ok[f"{ds}: P2 AUC ties with the window mean ({P2['auc_readout']:.4f}, wm {P2['auc_wm']:.4f}) and passes (>=)"] = \
            abs(P2["auc_readout"] - P2["auc_wm"]) < 1e-12 and P2["pass"] is True
    for arm, c in (("A2", "candidate_47"), ("B2", "candidate_48")):
        x = s[c]["arms"][arm]
        ok[f"{arm}: G1 PASS (zero differences), G2 FAIL, P1 FAIL, arm FAIL"] = \
            x["G1"] is True and x["G2"] is False and x["P1"] is False and x["pass"] is False
    for c, v in ok.items():
        log(f"SELFTEST {c}: {'PASS' if v else 'FAIL'}")
    log(f"SELFTEST {'PASS' if all(ok.values()) else 'FAIL'}")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", nargs="+", help="read_extra.py run directories or their reads.jsonl (main, dehate)")
    ap.add_argument("--datasets", nargs="+", default=None, help="default: every corpus in the reads")
    ap.add_argument("--B", type=int, default=1000, help="key_eval bootstrap; pre-checks use 2B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-datasets", nargs="+", default=list(DATASETS), choices=sorted(CORPORA))
    a = ap.parse_args()
    if a.selftest:
        log = selftest(a)
    else:
        if not a.reads:
            raise SystemExit("--reads or --selftest")
        _, _, log = run(a.reads, a.datasets, a.B, a.seed, a.out)
    log("RUN_DONE")


if __name__ == "__main__":
    main()
