#!/usr/bin/env python3
"""Candidate 49 analysis in one command (CPU; reads the test GT for pre-checks and evaluation only, rule 10).

Input: one or more read_segments.py runs (segments.jsonl; verify rows in the records or in verify.jsonl), e.g.
runs/20261025_m3_coverage/seg_main (HateMM + HateClipSeg) and runs/20261025_m3_coverage/seg_dehate (DeHate, numbers
only). README section 11 (constants copied from there):
  extent terms per video, from the J = 4 mutually invisible segment verdicts z_j:
    q_v       = mean_j z_j
    p_hat_v   = 1 - (1 - f)^(1 / m),  f = mean_j sigmoid(z_j), m = mean_j m_overlap;  p_hat = 1 - 1e-6 when f = 1
                (float64); written as logit p_hat
  arms (calibrated by the r6 two-Gaussian mixture through twolevel_r2.py --key calib; frame score = key + r6_bma rank):
    C1  K = z_video + s q_v,            s  = sd(window mean) / sd(q_v)            (key_eval replace --scale sd)
    C2  K = z_video + s' logit p_hat_v, s' = sd(window mean) / sd(logit p_hat_v)
    selection C1 -> C2
  P0 (no GT; main corpora gated, DeHate reported): verify rows (first 2 videos per corpus): token seam of the full
     prefix and of every segment, |dz| < 1 for each; every video: reproduced z_video vs the r6 archive |dz_r6| < .05
     (from segments.jsonl); J = 4 and four z_j per record
  P1 hateful videos: rho(extent term, GT coverage) > rho(window mean, coverage) (.416 / .533), both corpora
  P3 all videos: rho(extent term, z_video) < .90, both corpora
  G1 no main metric below r6_bma by more than pooled .005 / within .01
  G2 (revised charter) removing the extent term (K = z_video, calibrated; key_eval verdict_only) drops the same main
     metric by >= .01 on both corpora
  diagnostics (not in the method): C3 = calibrated s q_v alone (key_eval calib_scalar); additive K = z_video + window
     mean + s q_v (key_eval add --scale sd); winonly K = z_video + s mean(single-window reads z_w), with the old-ASR cache
     runs/20260910_spvl/mllm/q3vl-8b/winonly (HateMM, HCS only); HCS hateful-only GT; DeHate; rho(q_v, window mean),
     rho(q_v, duration), rho(logit p_hat, duration); share of videos whose four segments all read Yes / all No;
     partial Spearman given the window mean; rho on videos with z_video > 0; single-window reference .362 / .224.
Output: runs/20261025_m3_coverage/<run names joined by _>_analysis/{summary.json, summary.txt, per_video.jsonl,
values/, precheck/<term>/, keyeval/<group>/<term>/, run.log, config.json}.

  python experiments/20261025_m3_coverage/analyze_segments.py --reads runs/20261025_m3_coverage/seg_main [.../seg_dehate]
  python experiments/20261025_m3_coverage/analyze_segments.py --selftest [--selftest-datasets HateMM HateClipSeg DeHate]
      (fake segment reads z_j = .5 wm + c_j: C1 must reproduce r6_bma, P1 must tie with the window mean and fail,
       G1 must pass, G2 must equal r6_bma vs verdict-only)
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
from analyze import (KEY_WARN, NOISE_POOLED, NOISE_WITHIN, arm_numbers, f3, f4, gate, key_max,  # noqa: E402
                     rho_zv_positive, tri)
from common import (CORPORA, DATASETS, EXTERNAL, OUT_ROOT, ROOT, gt_sets, load_r6, load_reads, spearman,  # noqa: E402
                    start_run, write_values)
from precheck import analyse, table  # noqa: E402

J = 4
P0_DZ_SEG, P0_DZ_R6 = 1.0, 0.05
P3_MAX, ABL_DROP = 0.90, 0.01
DECLARED_WM = {"HateMM": .416, "HateClipSeg": .533}
WINONLY = ROOT / "runs/20260910_spvl/mllm/q3vl-8b/winonly"
METRICS = ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC")
# key_eval runs: (term, forms); the main arms and diagnostics are read from them
RUNS = {"q": ["replace", "verdict_only", "add", "calib_scalar"], "logit_phat": ["replace"], "winonly_mean": ["replace"]}
ARMS = [("C1", "q", "replace"), ("C2", "logit_phat", "replace")]
DIAG = [("C3_q_alone", "q", "calib_scalar", "calibrated s q_v alone, no z_video"),
        ("additive", "q", "add", "z_video + window mean + s q_v"),
        ("verdict_only", "q", "verdict_only", "G2 ablation: extent term removed, K = z_video"),
        ("winonly", "winonly_mean", "replace", "z_video + s mean(single-window reads), old ASR (20260910_spvl winonly)")]


# ----------------------------------------------------------------------------------------------- inputs

def load_runs(paths):
    recs, ver, n_err, names = {}, {}, 0, []
    for rp in paths:
        rp = Path(rp)
        d = rp if rp.is_dir() else rp.parent
        f = d / "segments.jsonl" if rp.is_dir() else rp
        names.append(d.name)
        for line in open(f):
            r = json.loads(line); k = (r["dataset"], r["video_id"])
            if r.get("error"):
                n_err += 1; continue
            if k in recs:
                raise SystemExit(f"{k} appears in more than one segments file")
            recs[k] = r
        if (d / "verify.jsonl").exists():
            for line in open(d / "verify.jsonl"):
                v = json.loads(line); ver[(v["dataset"], v["video_id"])] = v
    for k, r in recs.items():
        if "verify" in r:
            ver[k] = {**ver.get(k, {}), **r["verify"]}
    return names, recs, n_err, ver


def terms(r):
    z = np.asarray(r["z_seg"], float)
    f = float(np.mean(expit(z))); m = float(np.mean([s["m_overlap"] for s in r["segments"]]))
    clamped = f >= 1.0
    p = 1 - 1e-6 if clamped else 1 - (1 - f) ** (1 / m)
    return {"q": float(z.mean()), "f": f, "m": m, "p_hat": p, "logit_phat": float(math.log(p) - math.log1p(-p)),
            "phat_clamped": clamped, "all_yes": bool((z > 0).all()), "all_no": bool((z < 0).all()),
            "duration": float(r["duration"])}


def winonly_means():
    out = {}
    for line in open(WINONLY / "predictions.jsonl"):
        r = json.loads(line)
        if not r.get("error"):
            out[(r["dataset"], r["video_id"])] = float(np.mean([w["z"] for w in r["extra"]["windows"]]))
    return out


def p0(recs, ver, datasets):
    out = {"checks": {}, "stats": {}}
    for ds in datasets:
        ks = [k for k in recs if k[0] == ds]
        dz = np.array([recs[k]["dz_r6"] for k in ks if recs[k].get("dz_r6") is not None], float)
        shape_ok = all(recs[k].get("J") == J and len(recs[k]["z_seg"]) == J and len(recs[k]["segments"]) == J for k in ks)
        frames_ok = sum(sum(s["n_frames"] for s in recs[k]["segments"]) == recs[k]["n_frames"] for k in ks)
        vrows = []
        for k in sorted(x for x in ver if x[0] == ds):
            v = ver[k]; segs = v.get("segments", [])
            vrows.append({"video_id": k[1], "seam_ok_video": v["video"]["seam_ok"], "dz_video": v["video"]["dz"],
                          "seam_ok_segments": [s["seam_ok"] for s in segs], "dz_segments": [s["dz"] for s in segs]})
        ok_v = bool(vrows) and all(r["seam_ok_video"] and all(r["seam_ok_segments"]) and len(r["dz_segments"]) == J
                                   and r["dz_video"] < P0_DZ_SEG and max(r["dz_segments"]) < P0_DZ_SEG for r in vrows)
        out["checks"][f"{ds}_verify_seam_and_dz"] = {"pass": ok_v, "n_verify_videos": len(vrows)}
        out["checks"][f"{ds}_z_video_vs_r6_archive"] = {"pass": bool(len(dz) == len(ks) and (dz < P0_DZ_R6).all()),
                                                       "max": float(dz.max()) if len(dz) else None,
                                                       "n_ge_gate": int((dz >= P0_DZ_R6).sum()), "n_missing": len(ks) - len(dz)}
        out["checks"][f"{ds}_four_segments"] = {"pass": shape_ok}
        out["stats"][ds] = {"n_videos": len(ks), "n_frames_partition_ok": frames_ok, "verify": vrows}
    gated = [c for c in out["checks"] if not any(c.startswith(x + "_") for x in EXTERNAL)]
    out["gated_checks"] = gated
    out["pass"] = bool(gated) and all(out["checks"][c]["pass"] is True for c in gated)
    return out


# ----------------------------------------------------------------------------------------------- key evaluation

def launch(vfiles, groups, B, seed, out, log):
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    jobs = {}
    for g, dss in groups.items():
        for term, forms in RUNS.items():
            use = [ds for ds in dss if any(k[0] == ds for k in vfiles[term][1])]
            if not use:
                continue
            d = out / "keyeval" / g / term; d.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(ROOT / "experiments/20261025_m3_coverage/key_eval.py"), "--values", str(vfiles[term][0]),
                   "--name", term, "--forms", *forms, "--transform", "none", "--scale", "sd", "--datasets", *use,
                   "--B", str(B), "--seed", str(seed), "--out", str(d)]
            log(f"key_eval {g}/{term}: {' '.join(cmd[1:])}")
            jobs[(g, term)] = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=open(d / "stdout.log", "w"), stderr=subprocess.STDOUT)
    return jobs


def wait(jobs, out, log):
    res = {}
    for (g, term), p in jobs.items():
        if p.wait() != 0:
            raise SystemExit(f"key_eval {g}/{term} FAILED (exit {p.returncode}); see {out / 'keyeval' / g / term / 'stdout.log'}")
        res[(g, term)] = json.load(open(out / "keyeval" / g / term / "summary.json"))
        log(f"key_eval {g}/{term} done")
    return res


def g1(nums, main):
    if nums is None or any(ds not in nums for ds in main):
        return None
    xs = [gate(nums[ds]["delta"][0], -NOISE_POOLED, ">=") for ds in main] + \
         [gate(nums[ds]["delta"][1], -NOISE_POOLED, ">=") for ds in main] + \
         [gate(nums[ds]["delta"][2], -NOISE_WITHIN, ">=") for ds in main]
    return None if None in xs else all(xs)


def g2(ke_arm, form, ke_abl, main):
    """Removing the extent term (form -> verdict_only): the same main metric drops by >= .01 on both corpora."""
    if ke_arm is None or ke_abl is None or form not in ke_arm["forms"] or "verdict_only" not in ke_abl["forms"]:
        return None, {}
    det = {}
    for m in METRICS:
        d = [None if ke_arm["evaluator_metrics"][form][ds][m] is None or ke_abl["evaluator_metrics"]["verdict_only"][ds][m] is None
             else ke_arm["evaluator_metrics"][form][ds][m] - ke_abl["evaluator_metrics"]["verdict_only"][ds][m] for ds in main]
        det[m] = {"drop_when_removed": d, "pass": None if None in d else all(x >= ABL_DROP - 1e-12 for x in d)}
    vals = [x["pass"] for x in det.values()]
    return (None if None in vals else any(vals)), det


# ----------------------------------------------------------------------------------------------- main

def run(paths, datasets_req, B, seed, out=None, selftest=False):
    names, recs, n_err, ver = load_runs(paths)
    present = sorted({k[0] for k in recs})
    datasets = [ds for ds in (datasets_req or present) if ds in present]
    main = [ds for ds in DATASETS if ds in datasets]; ext = [ds for ds in EXTERNAL if ds in datasets]
    out = Path(out) if out else OUT_ROOT / f"{'_'.join(names)}_analysis"
    log = start_run(out, "experiments/20261025_m3_coverage/analyze_segments.py",
                    {"reads": [str(p) for p in paths], "datasets": datasets, "B": B, "seed": seed, "selftest": selftest,
                     "constants": {"J": J, "P0": [P0_DZ_SEG, P0_DZ_R6], "P3": P3_MAX, "noise": [NOISE_POOLED, NOISE_WITHIN],
                                   "G2_drop": ABL_DROP, "phat_clamp": 1e-6, "key_warn": KEY_WARN}})
    log(f"reads {[str(p) for p in paths]}: {len(recs)} records without error, {n_err} error records; corpora {datasets}")
    per = {k: terms(r) for k, r in recs.items() if k[0] in datasets}
    with open(out / "per_video.jsonl", "w") as fh:
        for (ds, v), t in sorted(per.items()):
            fh.write(json.dumps({"dataset": ds, "video_id": v, "z_seg": recs[(ds, v)]["z_seg"], **t}) + "\n")
    wo = {k: x for k, x in winonly_means().items() if k[0] in main}
    values = {"q": {k: t["q"] for k, t in per.items()}, "logit_phat": {k: t["logit_phat"] for k, t in per.items()},
              "winonly_mean": wo}
    vfiles = {}
    for n, v in values.items():
        vfiles[n] = (out / "values" / f"{n}.jsonl", v); write_values(vfiles[n][0], v)
    P0 = p0({k: recs[k] for k in per}, ver, datasets)
    log(f"P0 {'PASS' if P0['pass'] else 'FAIL'}: " + ", ".join(f"{c} {tri(x['pass'])}" for c, x in P0["checks"].items()))
    gts = gt_sets(tuple(datasets)); r6 = load_r6(datasets=tuple(datasets)); reads = load_reads(datasets=tuple(datasets))
    groups = {g: dss for g, dss in (("main", main), ("external", ext)) if dss}
    jobs = launch(vfiles, groups, B, seed, out, log)
    # pre-checks and descriptive statistics
    pre, rng = {}, np.random.default_rng(seed)
    for n, v in values.items():
        dss = [ds for ds in datasets if any(k[0] == ds for k in v)]
        if not dss:
            continue
        r = analyse(v, {g: x for g, x in gts.items() if x[0] in dss}, r6, reads, 2 * B, seed)
        d = out / "precheck" / n; d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(r, indent=1))
        (d / "summary.txt").write_text(table(r, f"precheck {n} (analyze_segments.py); video bootstrap B={2 * B}") + "\n")
        P = {}
        for ds in dss:
            h, a = r[ds]["hateful"], r[ds]["all"]
            P[ds] = {"P1": {"rho": h["rho_value_cov"]["point"], "ci95": h["rho_value_cov"]["ci95"], "rho_wm": h["rho_wm_cov"]["point"],
                            "declared_wm": DECLARED_WM.get(ds), "n": h["n"],
                            "pass": gate(h["rho_value_cov"]["point"], h["rho_wm_cov"]["point"], ">")},
                     "P3": {"rho_zvideo": a["rho_value_zv"]["point"], "ci95": a["rho_value_zv"]["ci95"],
                            "pass": gate(a["rho_value_zv"]["point"], P3_MAX, "<")},
                     "rho_wm_all": a["rho_value_wm"]["point"],
                     "partial_given_wm_hateful": h["partial_value_cov_given_wm"]["point"],
                     "partial_given_wm_hateful_ci95": h["partial_value_cov_given_wm"]["ci95"],
                     "zv_positive": rho_zv_positive(v, gts, r6, reads, ds, 2 * B, rng)}
        pre[n] = P
        log(f"precheck {n} done")
    desc = {}
    for ds in datasets:
        ks = [k for k in per if k[0] == ds]
        dur = [per[k]["duration"] for k in ks]
        desc[ds] = {"n": len(ks), "rho_q_wm": spearman([per[k]["q"] for k in ks], [reads[k]["wm"] for k in ks]),
                    "rho_q_duration": spearman([per[k]["q"] for k in ks], dur),
                    "rho_logit_phat_duration": spearman([per[k]["logit_phat"] for k in ks], dur),
                    "rho_wm_duration": spearman([reads[k]["wm"] for k in ks], dur),
                    "share_all_yes": float(np.mean([per[k]["all_yes"] for k in ks])),
                    "share_all_no": float(np.mean([per[k]["all_no"] for k in ks])),
                    "n_phat_clamped": int(sum(per[k]["phat_clamped"] for k in ks))}
    ke = wait(jobs, out, log)
    strict = ["HateClipSeg_strict"] if "HateClipSeg" in main else []
    arms = {}
    for name, term, form in ARMS:
        kk = ke.get(("main", term)); nums = arm_numbers(kk, form, main, strict)
        Pm = pre.get(term, {})
        p1 = [Pm.get(ds, {}).get("P1", {}).get("pass") for ds in main]
        p3 = [Pm.get(ds, {}).get("P3", {}).get("pass") for ds in main]
        G2, det = g2(kk, form, ke.get(("main", "q")), main)
        conds = {"P0": P0["pass"], "P1": None if (not p1 or None in p1) else all(p1),
                 "P3": None if (not p3 or None in p3) else all(p3), "G1": g1(nums, main), "G2": G2}
        arms[name] = {"term": term, "form": form, **conds, "G2_detail": det, "failing": [c for c, x in conds.items() if x is False],
                      "pass": None if None in conds.values() else all(conds.values()), "numbers": nums,
                      "external": arm_numbers(ke.get(("external", term)), form, ext, []),
                      "key_max": {**(key_max(kk, form) or {}), **(key_max(ke.get(("external", term)), form) or {})}}
    diag = {name: {"term": term, "form": form, "what": what,
                   "numbers": arm_numbers(ke.get(("main", term)), form, main, strict),
                   "external": arm_numbers(ke.get(("external", term)), form, ext, []),
                   "key_max": {**(key_max(ke.get(("main", term)), form) or {}), **(key_max(ke.get(("external", term)), form) or {})}}
            for name, term, form, what in DIAG}
    sel = next((a for a, x in arms.items() if x["pass"]), None)
    if not main:
        verdict = "no main corpus in the reads: gates not evaluated"
    elif not P0["pass"]:
        verdict = "P0 FAIL: pipeline check failed; fix the code (not a candidate failure); gates are for the record only"
    elif sel == "C1":
        verdict = "C1 (declared first arm) passes P1, P3, G1, G2"
    elif sel:
        verdict = f"{sel} passes P1, P3, G1, G2; selected between the two declared arms by test results (development-selected)"
    elif any(x["pass"] is None for x in arms.values()):
        verdict = "undetermined: some conditions could not be evaluated (n/a)"
    else:
        verdict = "no declared arm passes: archive per rule 12 (" + "; ".join(f"{a}: {', '.join(x['failing'])}" for a, x in arms.items()) + ")"
    warns = [f"{a} ({x['term']}, {x['form']}) {ds}: max key {v:.2f}" for grp in (arms, diag) for a, x in grp.items()
             for ds, v in x["key_max"].items() if v > KEY_WARN]
    for w in warns:
        log(f"WARNING key above {KEY_WARN:g}: {w}")
    summary = {"reads": [str(p) for p in paths], "datasets": datasets, "n_records": len(recs), "n_error_records": n_err,
               "P0": P0, "prechecks": pre, "descriptive": desc, "arms": arms, "diagnostics": diag, "selection_order": ["C1", "C2"],
               "selected": sel, "verdict": verdict, "key_max_warnings": warns,
               "references": {"single_window_rho_declared": {"HateMM": .362, "HateClipSeg": .224}},
               "not_checked_here": ["rule 4 novelty review", "rule 6 code review"], "B": B, "seed": seed}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    txt = render(summary, ext)
    (out / "summary.txt").write_text(txt + "\n")
    print(txt)
    log(f"CANDIDATE 49 VERDICT {verdict}")
    return summary, out, log


def num_lines(x):
    L = []
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
    L = [f"candidate 49 analysis: {s['reads']}  ({s['n_records']} records, {s['n_error_records']} errors; corpora "
         f"{s['datasets']}); development-selected; HateClipSeg_strict = hateful-only GT (diagnostic)",
         f"\nCANDIDATE 49 VERDICT: {s['verdict']}   (order C1 -> C2; selected: {s['selected']})"]
    for w in s["key_max_warnings"]:
        L.append(f"WARNING (key above {KEY_WARN:g}): {w}")
    L.append(f"\n== P0 (no GT; main corpora gated): {tri(s['P0']['pass'])}")
    for c, x in s["P0"]["checks"].items():
        L.append(f"  {c:34s} {tri(x['pass'])}{'' if c in s['P0']['gated_checks'] else ' (external, report only)'}  "
                 + ", ".join(f"{k} {v}" for k, v in x.items() if k != "pass"))
    for ds, st in s["P0"]["stats"].items():
        L.append(f"  {ds}: n {st['n_videos']}, frame partition ok on {st['n_frames_partition_ok']}")
        for r in st["verify"]:
            L.append(f"    verify {r['video_id']}: seam video {r['seam_ok_video']} segments {r['seam_ok_segments']}  "
                     f"|dz| video {r['dz_video']:.3f} segments {[round(x, 3) for x in r['dz_segments']]}")
    L.append("\n== pre-checks (protocol GT): P1 hateful rho vs window mean; P3 rho with z_video (all videos); partial|wm "
             "(hateful); zv>0 = videos with z_video > 0")
    for n, P in s["prechecks"].items():
        for ds, x in P.items():
            z = x["zv_positive"]; tag = " (report only)" if ds in ext else ""
            L.append(f"  {n:13s} {ds:12s}{tag} P1 {f3(x['P1']['rho'])} [{f3(x['P1']['ci95'][0])},{f3(x['P1']['ci95'][1])}] vs wm "
                     f"{f3(x['P1']['rho_wm'])} {tri(x['P1']['pass'])} | P3 {f3(x['P3']['rho_zvideo'])} {tri(x['P3']['pass'])} | "
                     f"rho(.,wm) {f3(x['rho_wm_all'])} | partial|wm {f3(x['partial_given_wm_hateful'])} "
                     f"[{f3(x['partial_given_wm_hateful_ci95'][0])},{f3(x['partial_given_wm_hateful_ci95'][1])}] | "
                     f"zv>0 (n {z['n']}) {f3(z['rho_readout'])} vs wm {f3(z['rho_wm'])}")
    L.append("  (winonly_mean = single-window reads, old ASR cache; declared reference .362 / .224)")
    L.append("\n== descriptive (no GT)")
    for ds, x in s["descriptive"].items():
        L.append(f"  {ds:12s} n {x['n']}: rho(q, wm) {f3(x['rho_q_wm'])}  rho(q, duration) {f3(x['rho_q_duration'])}  "
                 f"rho(logit p_hat, duration) {f3(x['rho_logit_phat_duration'])}  (rho(wm, duration) {f3(x['rho_wm_duration'])})  "
                 f"all four Yes {x['share_all_yes']:.3f}  all four No {x['share_all_no']:.3f}  p_hat clamped {x['n_phat_clamped']}")
    L.append("\n== arms (point = evaluator metrics on the same videos as r6_bma; d = arm - r6_bma [95% paired video bootstrap]; "
             "G2 = arm - verdict_only per metric)")
    for name, x in s["arms"].items():
        g2d = "; ".join(f"{m.split('_')[1] if m != 'within_video_macro_ROC_AUC' else 'within'} "
                        + "/".join(f4(v, True) for v in d["drop_when_removed"]) for m, d in x["G2_detail"].items())
        L.append(f"  {name:13s} {x['term']:11s} {x['form']:12s} " + " ".join(f"{c} {tri(x[c])}" for c in ("P0", "P1", "P3", "G1", "G2"))
                 + f" -> {tri(x['pass'])}  max key " + (", ".join(f"{ds} {v:.2f}" for ds, v in x["key_max"].items()) or "n/a"))
        L.append(f"      G2 (arm - verdict_only, HateMM/HCS): {g2d}")
        L += num_lines(x)
    L.append("  diagnostics:")
    for name, x in s["diagnostics"].items():
        L.append(f"  {name:13s} {x['term']:11s} {x['form']:12s} ({x['what']})  max key "
                 + (", ".join(f"{ds} {v:.2f}" for ds, v in x["key_max"].items()) or "n/a"))
        L += num_lines(x)
    L.append("\nnot checked here: rule 4 novelty review, rule 6 code review (README section 11.4)")
    return "\n".join(L)


# ----------------------------------------------------------------------------------------------- self-test

def fake_segments(out, datasets):
    """Fake read_segments.py output: z_j = .5 wm + c_j, c = (-1, 0, .5, 1), so q_v = .5 wm + .125 (affine);
    m_overlap from the 8 s grid geometry; z_video and dz_r6 from the r6 reads; verify rows on the first 2 videos."""
    reads = load_reads(datasets=datasets)
    dur = {}
    for ds in datasets:
        for line in open(Path(CORPORA[ds]["reads"]) / "predictions.jsonl"):
            r = json.loads(line); dur[(r["dataset"], r["video_id"])] = float(r["duration"])
    out.mkdir(parents=True, exist_ok=True)
    c = [-1.0, 0.0, 0.5, 1.0]; first = {}
    with open(out / "segments.jsonl", "w") as fh, open(out / "verify.jsonl", "w") as vf:
        for (ds, v), r in sorted(reads.items()):
            D = dur[(ds, v)]
            nw = max(1, int(math.ceil(D / 8 - 1e-9))); wins = [(8 * i, min(8 * (i + 1), D)) for i in range(nw)]
            segs = []
            for k in range(J):
                t1, t2 = k * D / J, (k + 1) * D / J
                segs.append({"k": k, "t1": t1, "t2": t2, "n_frames": 5, "n_words": 0,
                             "m_overlap": sum(1 for a, b in wins if min(b, t2) - max(a, t1) > 1e-9), "z": 0.5 * r["wm"] + c[k]})
            rec = {"dataset": ds, "video_id": v, "J": J, "duration": D, "error": None, "z_video": r["zv"], "n_frames": 20,
                   "z_seg": [s["z"] for s in segs], "segments": segs, "z_video_r6": r["zv"], "dz_r6": 0.0}
            first[ds] = first.get(ds, 0) + 1
            if first[ds] <= 2:
                rec["verify"] = {"video": {"seam_ok": True, "dz": 0.01}, "segments": [{"seam_ok": True, "dz": 0.02}] * J}
                vf.write(json.dumps({"dataset": ds, "video_id": v, **rec["verify"]}) + "\n")
            fh.write(json.dumps(rec) + "\n")
    return out


def selftest(a):
    rp = fake_segments(OUT_ROOT / "selftest_fake_segments", a.selftest_datasets)
    s, out, log = run([rp], None, a.B, a.seed, a.out, selftest=True)
    ok = {"P0 pass": s["P0"]["pass"] is True}
    r6 = load_r6(datasets=tuple(a.selftest_datasets))
    for g in ("main", "external"):
        d = out / "keyeval" / g / "q"
        if not d.exists():
            continue
        ke = json.load(open(d / "summary.json"))
        sc = {(r["dataset"], r["video_id"]): np.asarray(r["score_curve"]) for r in
              map(json.loads, open(d / "arms" / "q__replace" / "predictions.jsonl"))}
        dmax = max(float(np.abs(sc[k] - r6[k]["score"]).max()) for k in sc)
        dmet = max(abs(ke["evaluator_metrics"]["replace"][ds][m] - ke["r6_metrics_file"][ds][m]) for ds in ke["datasets"] for m in METRICS)
        ok[f"{g}: C1 on q = .5 wm + .125 reproduces r6_bma (frame {dmax:.1e}, metrics {dmet:.1e})"] = \
            dmax < 1e-6 and dmet < 1e-6 and len(sc) == sum(k[0] in ke["datasets"] for k in r6)
    for ds in [x for x in DATASETS if x in a.selftest_datasets]:
        P = s["prechecks"]["q"][ds]
        ok[f"{ds}: P1 ties with the window mean ({P['P1']['rho']:.6f} vs {P['P1']['rho_wm']:.6f}) and fails"] = \
            abs(P["P1"]["rho"] - P["P1"]["rho_wm"]) < 1e-12 and P["P1"]["pass"] is False
        ok[f"{ds}: rho(q, wm) = 1 ({s['descriptive'][ds]['rho_q_wm']:.6f}); partial|wm = 0"] = \
            abs(s["descriptive"][ds]["rho_q_wm"] - 1) < 1e-12 and abs(P["partial_given_wm_hateful"]) < 1e-9
    c1 = s["arms"]["C1"]
    ok["C1: G1 PASS (zero differences), P1 FAIL, arm FAIL"] = c1["G1"] is True and c1["P1"] is False and c1["pass"] is False
    vo = s["diagnostics"]["verdict_only"]["numbers"]
    exp = [-vo[ds]["delta"][1] for ds in DATASETS if ds in a.selftest_datasets]
    got = c1["G2_detail"]["frame_PR_AUC"]["drop_when_removed"]
    ok[f"C1: G2 PR drop equals r6_bma - verdict_only ({[round(x, 4) for x in got]})"] = \
        all(abs(x - y) < 1e-12 for x, y in zip(got, exp)) and c1["G2"] is True
    for c, v in ok.items():
        log(f"SELFTEST {c}: {'PASS' if v else 'FAIL'}")
    log(f"SELFTEST {'PASS' if all(ok.values()) else 'FAIL'}")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", nargs="+", help="read_segments.py run directories or their segments.jsonl")
    ap.add_argument("--datasets", nargs="+", default=None)
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
