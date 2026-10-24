#!/usr/bin/env python3
"""Key variants for a per-video scalar, with r6_bma's M2 term held fixed (CPU, analysis only).

r6_bma frame score (experiments/20260926_twolevel/twolevel_r2.py, --key calib --arm m2):
    score = key + centred within-video rank of logit P(hate | V = 1, reads)
    key   = a K + b,  K = z_video + mean window read z,  (a, b) = label-free two-Gaussian calibration of K per corpus.
Here the rank term is taken unchanged from the corpus's r6_bma predictions (score - key; common.CORPORA) and only the
key is replaced. Keys that need the label-free calibration are produced by the twolevel_r2.py command line itself (not
imported, not copied): its K reads only the windows' "z" field (the M2 chain reads z_visual / z_speech), so a copy of
the read file with the windows' z replaced gives the calibrated key of a new K; its --dvd-reads / --dvd-conds interface
gives the product forms. The CLI's own M2 output is compared with r6_bma's as a check.

Forms (s = the scalar, t(s) = s or logit(clip(s, 1e-4, 1 - 1e-4)) by --transform; scaling statistics per corpus over
all its videos in the r6 run that have a value, label-free):
  calib_scalar  key = a t(s) + b           t(s) alone with r6's two-Gaussian calibration fitted on it (no z_video;
                                           candidate 49 diagnostic C3; scale-invariant, so s * q_v gives the same key)
  share_raw     key = t(s)                 no z_video, no calibration (candidate 47 A1 with s = logit E[c], A1-eta
                                           with s = eta; README section 2.2)
  replace       K' = z_video + g(s)        g = t(s) (--scale raw); sd(wm) / sd(t) * t (--scale sd; candidate 47 A2 /
                                           A2-eta); or t affinely matched to the window mean's mean and sd (--scale
                                           match). sd and match differ by a per-corpus constant in K', which the
                                           two-Gaussian calibration absorbs (checked by --selftest)
  add           K' = z_video + wm + g0(s)  g0 = t(s) (raw) or sd(wm) * standardised t(s) (sd, match)
  product_K     key = logit(P(V=1 | z_video + wm) * Q(s)),  Q = two-Gaussian calibration of t(s) (CLI DVD path)
  product_zv    key = logit(P(V=1 | z_video) * Q(s))
  product_K_prob / product_zv_prob   (only when every s is in [0, 1]) key = logit(P(V=1 | .) * clip(s, 1e-4, 1))
  verdict_only  K' = z_video (reference: the extent term removed; scalar-free)
Each arm's predictions, and r6_bma restricted to the same videos, are scored by src/eval/evaluate_four_datasets.py
(metrics.json); the paired video bootstrap of the pooled differences against r6_bma uses src/eval/evaluate.pooled;
within is unchanged by construction (the key is constant within a video) and is checked. HateClipSeg hateful-only GT is
reported as a diagnostic. Videos without a value are left out of every arm and of the r6 reference; CLI forms need
values for >= 90% of every corpus in the call (they then fit on the covered videos) and are skipped otherwise.

  python experiments/20261025_m3_coverage/key_eval.py --values X.jsonl --name X [--forms ...] [--transform logit]
         [--scale sd] [--datasets DeHate]
  python experiments/20261025_m3_coverage/key_eval.py --selftest   # s = window mean: replace (raw / sd / match) = r6_bma
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import log_expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CORPORA, DATASETS, OUT_ROOT, ROOT, ci, gt_sets, load_r6, load_reads, load_values,  # noqa: E402
                    start_run, write_values)

sys.path.insert(0, str(ROOT))
from src.eval.evaluate import pooled, within_video_macro  # noqa: E402

CLI = ROOT / "experiments/20260926_twolevel/twolevel_r2.py"
CLIP = 1e-4
MIN_COVER = 0.9
CLI_FORMS = ("replace", "add", "product_K", "product_zv", "verdict_only")
ALL_FORMS = ("share_raw", "calib_scalar") + CLI_FORMS + ("product_K_prob", "product_zv_prob")
METRICS = ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC")


def sources(datasets):
    r6s = {str(CORPORA[ds]["r6"]) for ds in datasets}; rds = {str(CORPORA[ds]["reads"]) for ds in datasets}
    if len(r6s) != 1 or len(rds) != 1:
        raise SystemExit(f"{datasets}: corpora with different r6 runs must be evaluated in separate calls")
    return Path(r6s.pop()), Path(rds.pop())


def r6_cli_args(r6_dir, reads_dir):
    """twolevel_r2.py flags of r6_bma, from its config.json."""
    c = json.load(open(r6_dir / "config.json"))
    run = Path(c["run"]); run = run if run.is_absolute() else ROOT / run
    if c["dvd_reads"] or c["dvd_conds"] or run.resolve() != reads_dir.resolve():
        raise SystemExit(f"{r6_dir}: r6_bma config is not the expected one")
    args = ["--k", str(c["k"]), "--d-gap", str(c["d_gap"]), "--d-hate", str(c["d_hate"]), "--evidence", c["evidence"],
            "--transform", c["transform"], "--fusion", c["fusion"], "--arm", c["arm"], "--key", c["key"],
            "--key-scale", str(c["key_scale"]), "--vlevel", c["vlevel"], "--vtemper", c["vtemper"], "--kappa", str(c["kappa"]),
            "--duration", c["duration"], "--bma-grid", str(c["bma_grid"]), "--bma-prior", c["bma_prior"],
            "--bma-fixed", str(c["bma_fixed"]), "--min-windows", str(c["min_windows"])]
    for flag, key in (("--center", "center"), ("--sharedchain", "sharedchain"), ("--nocoupling", "nocoupling"),
                      ("--noleak", "noleak"), ("--iota-stationary", "iota_stationary"), ("--norank", "norank")):
        if c[key]:
            args.append(flag)
    return args


def transform(s, how):
    s = np.asarray(s, float)
    return np.log(np.clip(s, CLIP, 1 - CLIP)) - np.log1p(-np.clip(s, CLIP, 1 - CLIP)) if how == "logit" else s


def window_z_maps(values, reads, keys, datasets, how, scale):
    """Per form, the new per-video window z (replace / zero), the per-video shift of every window z (add), the DVD
    field (product forms) and the share key t(s)."""
    out = {"replace": {}, "add": {}, "zero": {}, "dvd": {}, "share": {}, "constant": []}
    for ds in datasets:
        ks = [k for k in keys if k[0] == ds]
        if not ks:
            continue
        t = transform([values[k] for k in ks], how)
        wm = np.array([reads[k]["wm"] for k in ks])
        if scale in ("match", "sd") and t.std() == 0:
            out["constant"].append(ds)                 # no scaling defined: replace / add are skipped
            rep = add = np.full(len(ks), np.nan)
        elif scale in ("match", "sd"):
            if scale == "match":
                zs = (t - t.mean()) / t.std()
                rep, add = wm.mean() + wm.std() * zs, wm.std() * zs
            else:
                rep = add = wm.std() / t.std() * t
        else:
            rep, add = t, t
        for k, r_, a_, t_ in zip(ks, rep, add, t):
            out["replace"][k] = float(r_); out["add"][k] = float(a_); out["zero"][k] = 0.0
            out["dvd"][k] = float(t_); out["share"][k] = float(t_)
    return out


def write_synthetic(path, reads_dir, mode, m):
    """Copy of the r6 reads (videos in m only) with the windows' z replaced (set) or shifted (shift); the score curve is
    replaced by zeros of the same length (twolevel_r2.py reads only its length)."""
    path.mkdir(parents=True, exist_ok=True)
    with open(reads_dir / "predictions.jsonl") as src, open(path / "predictions.jsonl", "w") as dst:
        for line in src:
            r = json.loads(line)
            k = (r["dataset"], r["video_id"])
            if r.get("error") or k not in m:
                continue
            for w in r["extra"]["windows"]:
                w["z"] = m[k] if mode == "set" else float(w["z"]) + m[k]
            r["score_curve"] = [0.0] * len(r["score_curve"])
            dst.write(json.dumps(r) + "\n")


def run_cli(forms, maps, datasets, r6_dir, reads_dir, out, log):
    """Launch twolevel_r2.py for the CLI forms in parallel; returns {form: CLI output dir}."""
    inp, cli_out = out / "cli_inputs", out / "cli"
    cli_out.mkdir(parents=True, exist_ok=True)
    base = r6_cli_args(r6_dir, reads_dir)
    jobs = {}
    for f in forms:
        extra = []
        if f in ("replace", "add"):
            write_synthetic(inp / f, reads_dir, "set" if f == "replace" else "shift", maps[f]); run = inp / f
        elif f in ("verdict_only", "product_zv"):
            if not (inp / "zero" / "predictions.jsonl").exists():
                write_synthetic(inp / "zero", reads_dir, "set", maps["zero"])
            run = inp / "zero"
        else:                                   # product_K: the r6 reads, restricted to the covered videos
            if not (inp / "same" / "predictions.jsonl").exists():
                write_synthetic(inp / "same", reads_dir, "shift", maps["zero"])
            run = inp / "same"
        if f.startswith("product"):
            dv = inp / "dvd_scalar.jsonl"
            if not dv.exists():
                with open(dv, "w") as fh:
                    for (ds, v), x in sorted(maps["dvd"].items()):
                        fh.write(json.dumps({"dataset": ds, "video_id": v, "z_target": x}) + "\n")
            extra = ["--dvd-reads", str(dv), "--dvd-conds", "viol,T"]
        cmd = [sys.executable, str(CLI), "--run", str(run), *base, *extra, "--tag", f,
               "--out-root", str(cli_out), "--datasets", *datasets]
        log(f"CLI {f}: {' '.join(cmd[1:])}")
        jobs[f] = subprocess.Popen(cmd, cwd=ROOT, stdout=open(cli_out / f"{f}.stdout", "w"), stderr=subprocess.STDOUT)
    for f, p in jobs.items():
        if p.wait() != 0:
            raise SystemExit(f"CLI {f} FAILED (exit {p.returncode}); see {cli_out / (f + '.stdout')}")
        log(f"CLI {f} done")
    return {f: cli_out / f for f in forms}


def cli_keys(d):
    out = {}
    for line in open(d / "predictions.jsonl"):
        r = json.loads(line)
        out[(r["dataset"], r["video_id"])] = (float(r["extra"]["p_video_logodds"]), np.asarray(r["extra"]["cell_prob"], float))
    return out


def evaluate_arm(name, scores, datasets, out):
    """Write the frame scores as predictions and score them with the repository evaluator."""
    d = out / "arms" / name; d.mkdir(parents=True, exist_ok=True)
    with open(d / "predictions.jsonl", "w") as fh:
        for (ds, v), s in sorted(scores.items()):
            fh.write(json.dumps({"schema_version": 1, "method": f"m3cov__{name}", "dataset": ds, "video_id": v,
                                 "score_curve": [float(x) for x in s], "intervals": []}) + "\n")
    subprocess.run([sys.executable, "-m", "src.eval.evaluate_four_datasets", "--predictions", str(d / "predictions.jsonl"),
                    "--gt-dir", str(ROOT / "data/gt_4fps"), "--out", str(d / "metrics.json"), "--datasets", *datasets],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return {p["dataset"]: p for p in json.load(open(d / "metrics.json"))["per_dataset"]}


def bootstrap(arms, r6, keys, gts, B, seed):
    """Paired video bootstrap of pooled ROC / PR differences of each arm against r6_bma, per GT set."""
    rng = np.random.default_rng(seed)
    res = {}
    for gname, (ds, Y) in gts.items():
        vids = sorted(v for (d, v) in keys if d == ds and v in Y)
        if not vids:
            continue
        S = {"r6_bma": {v: r6[(ds, v)]["score"] for v in vids}}
        for a, kk in arms.items():
            S[a] = {v: kk[(ds, v)] + r6[(ds, v)]["rank"] for v in vids}
        point = {}
        for a, s in S.items():
            p = pooled(Y, s); w = within_video_macro(Y, s)
            point[a] = (p.get("frame_ROC_AUC"), p.get("frame_PR_AUC"), w["within_video_macro_ROC_AUC"], w["n_videos_defined"])
        diffs = {a: [] for a in arms}
        for _ in range(B):
            idx = rng.integers(0, len(vids), len(vids))
            yb = {f"{vids[i]}#{j}": Y[vids[i]] for j, i in enumerate(idx)}
            base = pooled(yb, {f"{vids[i]}#{j}": S["r6_bma"][vids[i]] for j, i in enumerate(idx)})
            for a in arms:
                q = pooled(yb, {f"{vids[i]}#{j}": S[a][vids[i]] for j, i in enumerate(idx)})
                if base.get("frame_ROC_AUC") is None or q.get("frame_ROC_AUC") is None:
                    diffs[a].append((np.nan, np.nan)); continue
                diffs[a].append((q["frame_ROC_AUC"] - base["frame_ROC_AUC"], q["frame_PR_AUC"] - base["frame_PR_AUC"]))
        res[gname] = {"dataset": ds, "n_videos": len(vids), "point": {a: list(x) for a, x in point.items()}, "delta": {}}
        sub = lambda x, y: None if x is None or y is None else x - y
        for a in arms:
            dd = np.array(diffs[a], float).reshape(-1, 2)
            res[gname]["delta"][a] = {"dROC": sub(point[a][0], point["r6_bma"][0]), "dPR": sub(point[a][1], point["r6_bma"][1]),
                                      "dWithin": sub(point[a][2], point["r6_bma"][2]),
                                      "ROC_ci95": ci(dd[:, 0]), "PR_ci95": ci(dd[:, 1])}
    return res


def fmt(x, nd=4, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def evaluate_values(values, name, out, forms, how, scale, B, seed, log, datasets=DATASETS):
    datasets = tuple(datasets)
    r6_dir, reads_dir = sources(datasets)
    r6, reads, gts = load_r6(datasets=datasets), load_reads(datasets=datasets), gt_sets(datasets)
    keys = sorted(k for k in r6 if k in values)
    cover = {ds: (sum(k[0] == ds for k in keys), sum(k[0] == ds for k in r6)) for ds in datasets}
    log(f"{name}: values for " + ", ".join(f"{ds} {a}/{b}" for ds, (a, b) in cover.items()))
    if any(a == 0 for a, b in cover.values()):
        raise SystemExit(f"no values for some corpus: {cover}")
    cli_ok = all(a >= MIN_COVER * b for a, b in cover.values())
    in01 = all(0.0 <= values[k] <= 1.0 for k in keys)
    forms = [f for f in forms if (in01 or not f.endswith("_prob")) and (cli_ok or f not in CLI_FORMS)]
    skipped = [f for f in ALL_FORMS if f not in forms]
    if not cli_ok:
        log(f"CLI forms SKIPPED: values cover less than {MIN_COVER:.0%} of some corpus")
    need_cli = sorted({f for f in forms if f in CLI_FORMS} | ({"verdict_only"} if "product_zv_prob" in forms else set()),
                      key=ALL_FORMS.index)
    kset = set(keys)
    maps = window_z_maps(values, reads, keys, datasets, how, scale)
    if maps["constant"]:
        log(f"forms replace / add SKIPPED: the scalar is constant on {maps['constant']} (scale {scale} undefined)")
        forms = [f for f in forms if f not in ("replace", "add")]
        skipped = [f for f in ALL_FORMS if f not in forms]
        need_cli = [f for f in need_cli if f not in ("replace", "add")]
    dirs = run_cli(need_cli, maps, datasets, r6_dir, reads_dir, out, log) if need_cli else {}
    arms, checks = {}, {}
    if "share_raw" in forms:
        arms["share_raw"] = {k: maps["share"][k] for k in keys}
    if "calib_scalar" in forms:                    # t(s) alone, r6's two-Gaussian calibration (explore_lib.key_mixture,
        from explore_lib import key_mixture        # checked equal to twolevel_r2.key_calibration in explore_hurdle.py)
        arms["calib_scalar"] = {}
        for ds in datasets:
            ks = [k for k in keys if k[0] == ds]
            mx = key_mixture([maps["share"][k] for k in ks])
            arms["calib_scalar"].update({k: mx["a"] * maps["share"][k] + mx["b"] for k in ks})
    for f in need_cli:
        ck = cli_keys(dirs[f])
        if set(ck) != kset:
            raise SystemExit(f"CLI {f}: video set differs from the covered r6_bma videos")
        arms[f] = {k: ck[k][0] for k in keys}
        checks[f] = {"max_abs_cell_prob_diff_vs_r6": float(max(np.abs(ck[k][1] - r6[k]["cell_prob"]).max() for k in keys))}
        log(f"{f}: CLI M2 cell_prob vs r6_bma max abs diff {checks[f]['max_abs_cell_prob_diff_vs_r6']:.2e} "
            f"(the arm uses r6_bma's rank term in any case)")
    if (out / "cli_inputs").exists():               # synthetic read copies: intermediates, regenerated on demand
        import shutil
        shutil.rmtree(out / "cli_inputs")
        log("removed cli_inputs/ (synthetic copies of the r6 reads)")
    for f in forms:
        if f.endswith("_prob"):
            ref = {k: r6[k]["lo"] for k in keys} if f == "product_K_prob" else arms["verdict_only"]
            lp = {k: float(log_expit(ref[k])) + float(np.log(np.clip(values[k], CLIP, 1.0))) for k in keys}
            arms[f] = {k: x - float(np.log(-np.expm1(min(x, -1e-12)))) for k, x in lp.items()}
    arms = {f: arms[f] for f in ALL_FORMS if f in arms}
    metrics = {"r6_bma": evaluate_arm(f"{name}__r6_bma", {k: r6[k]["score"] for k in keys}, datasets, out)}
    for f in arms:
        metrics[f] = evaluate_arm(f"{name}__{f}", {k: arms[f][k] + r6[k]["rank"] for k in keys}, datasets, out)
    r6m = {p["dataset"]: p for p in json.load(open(r6_dir / "metrics.json"))["per_dataset"]}
    full = all(a == b for a, b in cover.values())
    for ds in datasets:
        if full:
            d6 = max(abs(metrics["r6_bma"][ds][m] - r6m[ds][m]) for m in METRICS if r6m[ds].get(m) is not None)
            if d6 > 1e-9:
                raise SystemExit(f"{ds}: r6_bma re-evaluated differs from its metrics.json by {d6}")
        for f in arms:
            a_, b_ = metrics[f][ds]["within_video_macro_ROC_AUC"], metrics["r6_bma"][ds]["within_video_macro_ROC_AUC"]
            if a_ is not None and b_ is not None and abs(a_ - b_) > 1e-9:
                raise SystemExit(f"{f} {ds}: within changed by {a_ - b_} although the key is constant within a video")
    key_max = {f: {ds: float(max(v for k, v in arms[f].items() if k[0] == ds)) for ds in datasets} for f in arms}
    # twolevel_r2.py clamps the DVD (product) path at logit 27.6 (min(lp, -1e-12)); the calib path used by replace / add /
    # verdict_only has no clamp. Recorded for every form; a maximum above 27 is flagged.
    key_flags = [f"{f}/{ds} max key {x:.2f} > 27" for f, d_ in key_max.items() for ds, x in d_.items() if x > 27]
    for fl in key_flags:
        log(f"WARNING {fl}")
    log(f"bootstrap B={B} seed={seed}")
    boot = bootstrap(arms, r6, keys, gts, B, seed)
    summary = {"name": name, "datasets": list(datasets), "transform": how, "scale": scale, "forms": list(arms),
               "skipped_forms": skipped, "coverage": {ds: list(c) for ds, c in cover.items()}, "full_coverage": full,
               "B": B, "seed": seed, "cli_checks": checks, "key_max": key_max, "key_max_flags": key_flags,
               "r6_metrics_file": {ds: {m: r6m[ds].get(m) for m in METRICS} for ds in datasets},
               "evaluator_metrics": {f: {ds: {m: v[ds].get(m) for m in METRICS + ("n_videos_defined",)} for ds in datasets}
                                     for f, v in metrics.items()},
               "bootstrap": boot}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = [f"key_eval {name}: transform {how}, scale {scale}; frame score = key + r6_bma rank term; "
             f"paired video bootstrap B={B}; HateClipSeg_strict = hateful-only GT (diagnostic); "
             f"coverage " + ", ".join(f"{ds} {a}/{b}" for ds, (a, b) in cover.items())
             + (f"; skipped forms {skipped}" if skipped else "")]
    for gname, r in boot.items():
        b = r["point"]["r6_bma"]
        lines.append(f"\n== {gname} ({r['n_videos']} videos)   r6_bma {fmt(b[0])} / {fmt(b[1])} / within {fmt(b[2])} (n={b[3]})")
        for a, x in r["delta"].items():
            p = r["point"][a]
            lines.append(f"  {a:16s} {fmt(p[0])} / {fmt(p[1])} / {fmt(p[2])}   dROC {fmt(x['dROC'], sign=True)} "
                         f"[{fmt(x['ROC_ci95'][0], sign=True)},{fmt(x['ROC_ci95'][1], sign=True)}]  dPR {fmt(x['dPR'], sign=True)} "
                         f"[{fmt(x['PR_ci95'][0], sign=True)},{fmt(x['PR_ci95'][1], sign=True)}]  dWithin {fmt(x['dWithin'], sign=True)}")
    txt = "\n".join(lines)
    (out / "summary.txt").write_text(txt + "\n")
    print(txt)
    return summary, arms, r6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", help="jsonl: dataset, video_id, value")
    ap.add_argument("--name", default=None)
    ap.add_argument("--forms", nargs="+", default=[f for f in ALL_FORMS if f not in ("share_raw", "calib_scalar")], choices=ALL_FORMS)
    ap.add_argument("--transform", choices=["none", "logit"], default="none")
    ap.add_argument("--scale", choices=["match", "sd", "raw"], default="match")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=sorted(CORPORA))
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="s = window mean, transform none, form replace with scale raw / sd / match: each must equal "
                         "r6_bma; plus verdict_only")
    a = ap.parse_args()
    if not a.selftest and not a.values:
        raise SystemExit("--values or --selftest")
    name = "selftest" if a.selftest else (a.name or Path(a.values).stem)
    out = Path(a.out) if a.out else OUT_ROOT / f"analysis_keyeval_{name}"
    log = start_run(out, "experiments/20261025_m3_coverage/key_eval.py", vars(a))
    if a.selftest:
        reads, r6 = load_reads(datasets=a.datasets), load_r6(datasets=a.datasets)
        vp = out / "selftest_values" / "wmean.jsonl"
        write_values(vp, {k: reads[k]["wm"] for k in r6})
        ok = True
        for scale in ("raw", "sd", "match"):
            forms = ["replace", "verdict_only"] if scale == "raw" else ["replace"]
            summary, arms, r6 = evaluate_values(load_values(vp), f"selftest_{scale}", out / scale, forms, "none", scale,
                                                a.B, a.seed, log, a.datasets)
            dmax = max(float(np.abs(arms["replace"][k] + r6[k]["rank"] - r6[k]["score"]).max()) for k in r6)
            dmet = max(abs(summary["evaluator_metrics"]["replace"][ds][m] - summary["r6_metrics_file"][ds][m])
                       for ds in a.datasets for m in METRICS)
            ok_s = dmax < 1e-6 and dmet < 1e-6; ok &= ok_s
            log(f"SELFTEST replace(window mean), scale {scale}, vs r6_bma: max abs frame-score diff {dmax:.2e}, "
                f"max abs metric diff {dmet:.2e}: {'PASS' if ok_s else 'FAIL'}")
        log(f"SELFTEST {'PASS' if ok else 'FAIL'}")
    else:
        evaluate_values(load_values(a.values), name, out, a.forms, a.transform, a.scale, a.B, a.seed, log, a.datasets)
    log("RUN_DONE")


if __name__ == "__main__":
    main()
