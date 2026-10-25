#!/usr/bin/env python3
"""Candidate 50, stage 1 (HateMM + HateClipSeg) and stage 2 (DeHate) analysis: decode, score, gates.

Rule-10 territory: this script reads the 4 fps test GT, through the evaluator only. GT never enters the reads, the
reconstruction or the decoder.

Steps:
1. Load the arms' reads; check coverage (whole test split of each corpus), identical z_video / windows / branch sets.
2. Stage 1 only: build A1_recon = all_local reads mapped per video and modality to A1's own per-video mean and sd
   (population sd; offset only when a video has < 2 such windows or all_local's sd is 0), z = max over modalities.
3. Raw metrics of every arm (max curve, undecoded) with src/eval/evaluate_four_datasets.py.
4. Decode every arm with the unchanged r6 decoder, experiments/20260926_twolevel/twolevel_r2.py, with the flags of
   the r6 raw-key run (runs/20260926_twolevel/final_rawkey/config.json, checked here key by key). The decoder scores
   its own output with src/eval/evaluate_four_datasets.py.
5. Per-video within ROC-AUC with src/eval/evaluate.within_video_macro (one video at a time); paired video bootstrap.
6. Gates (a)-(f) of review_m1.md §5.3 (stage 1) or the stage-2 floors; PASS / FAIL per gate, kill rule.
Outputs under --out: raw/<arm>/metrics.json, decoded/<arm>/{predictions,metrics,params,config}.json(l),
reads/A1_recon/predictions.jsonl, summary.json, per_video.csv, table.tsv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.evaluate import within_video_macro  # noqa: E402
from src.video_inputs import load_manifest  # noqa: E402

MANIFESTS = {"HateMM": "data/omsl_v6_inputs/manifests/all_test.jsonl",
             "HateClipSeg": "data/omsl_v6_inputs/manifests/all_test.jsonl",
             "DeHate": "data/manifests/DeHate_test.jsonl"}
DECODER = "experiments/20260926_twolevel/twolevel_r2.py"
DECODER_FLAGS = ["--noleak", "--transform", "nscore", "--key", "raw", "--arm", "m2", "--duration", "bma",
                 "--bma-prior", "length", "--min-windows", "2", "--bma-grid", "6"]
# Values the r6 raw-key config must hold (flags above + defaults the decoder leaves untouched).
DECODER_CONFIG = {"noleak": True, "transform": "nscore", "key": "raw", "arm": "m2", "duration": "bma",
                  "bma_prior": "length", "min_windows": 2.0, "bma_grid": 6, "k": 1, "d_gap": 80.0, "d_hate": 80.0,
                  "center": False, "sharedchain": False, "nocoupling": False, "evidence": "em", "fusion": "or",
                  "key_scale": 1.0, "norank": False, "vlevel": "joint", "vtemper": "none", "kappa": 1.0,
                  "bma_fixed": 0.0, "iota_stationary": False, "dvd_reads": [], "dvd_conds": ""}
# Options added to the decoder after the raw-key run; their defaults leave the r6 computation unchanged.
DECODER_NEWER_DEFAULTS = {"kind": "two", "conditions": "", "conditions_mode": "both", "chains": None, "gate": "none",
                          "gate_visual": "", "gate_speech": ""}
R6 = {"HateMM": {"decoded": "runs/20260926_twolevel/final_rawkey", "reads": "runs/20260926_glr/base_gridA"},
      "HateClipSeg": {"decoded": "runs/20260926_twolevel/final_rawkey", "reads": "runs/20260926_glr/base_gridA"},
      "DeHate": {"decoded": "runs/20260927_dehate_external/r6_rawkey", "reads": "runs/20260927_dehate_external/reads_gridA"}}
METRICS = ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC")
MODS = ("z_visual", "z_speech")
STAGE_ARMS = {1: ("base", "all_local", "A1", "A1_other"), 2: ("all_local", "A1")}
STAGE_DATASETS = {1: ("HateMM", "HateClipSeg"), 2: ("DeHate",)}
# Thresholds copied from review_m1.md §5.3 (final; README "Protocol").
G_A_WITHIN = {"HateMM": .7458, "HateClipSeg": .6323}                    # (a) A1 within >= r6 - .005
G_B_POOLED = {"HateMM": (.8905, .6833), "HateClipSeg": (.7087, .6616)}  # (b) A1 pooled ROC / PR
G_CDE_GAIN = .01                                                          # (c) (d) (e) within gains
G_F_RAW = {"HateMM": .670, "HateClipSeg": .600}                          # (f) raw max-curve within of A1
S2_FLOORS = {"DeHate": {"within_video_macro_ROC_AUC": .6331, "frame_ROC_AUC": .6944, "frame_PR_AUC": .1518}}
BOOT = 10000


def jl(path):
    out = {}
    for r in map(json.loads, open(path)):
        key = (r["dataset"], r["video_id"])
        assert key not in out and not r.get("error"), key
        out[key] = r
    return out


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def check_arms(reads, datasets):
    expected = {(ds, r["video_id"]) for ds in datasets for r in load_manifest(ROOT / MANIFESTS[ds], [ds])}
    ref_arm = next(iter(reads))
    for arm, rr in reads.items():
        got = {k for k in rr if k[0] in datasets}
        assert got == expected, (arm, "coverage", len(got), len(expected))
    for key in expected:
        r0 = reads[ref_arm][key]
        for arm, rr in reads.items():
            r = rr[key]
            assert r["extra"]["z_video"] == r0["extra"]["z_video"], (arm, key, "z_video differs")
            assert len(r["score_curve"]) == len(r0["score_curve"]) and r["native_rate"] == 4
            assert np.isfinite(r["score_curve"]).all(), (arm, key)
            w, w0 = r["extra"]["windows"], r0["extra"]["windows"]
            assert [(x["start"], x["end"]) for x in w] == [(x["start"], x["end"]) for x in w0], (arm, key)
            assert [tuple(m for m in MODS if m in x) for x in w] == [tuple(m for m in MODS if m in x) for x in w0], (arm, key)
    return sorted(expected)


def paint(length, values):
    idx = np.clip(((np.arange(length) + .5) / 4 // 8).astype(int), 0, len(values) - 1)
    return np.asarray(values, float)[idx]


def reconstruct(al, a1, keys):
    """all_local reads mapped per video and modality to A1's per-video mean and sd (label-free; diagnostic arm)."""
    out = []
    for key in keys:
        r0, r1 = al[key], a1[key]
        W0, W1 = r0["extra"]["windows"], r1["extra"]["windows"]
        new = {m: [w.get(m) for w in W0] for m in MODS}
        for m in MODS:
            idx = [i for i in range(len(W0)) if m in W0[i] and m in W1[i]]
            if not idx:
                continue
            x0 = np.array([W0[i][m] for i in idx], float)
            x1 = np.array([W1[i][m] for i in idx], float)
            if len(idx) < 2 or x0.std() == 0:
                y = x0 + (x1.mean() - x0.mean())
            else:
                y = (x0 - x0.mean()) / x0.std() * x1.std() + x1.mean()
            for i, v in zip(idx, y):
                new[m][i] = float(v)
        windows = []
        for i, w in enumerate(W0):
            nw = {k: v for k, v in w.items() if k not in MODS + ("z",)}
            for m in MODS:
                if new[m][i] is not None:
                    nw[m] = new[m][i]
            nw["z"] = float(max(nw[m] for m in MODS if m in nw))
            windows.append(nw)
        curve = paint(len(r0["score_curve"]), [w["z"] for w in windows])
        out.append({**r0, "method": "c50_act_state_A1_recon", "score_curve": [float(c) for c in curve],
                    "extra": {**r0["extra"], "windows": windows,
                              "reconstruction": "all_local mapped per video and modality to A1's mean and sd"}})
    return out


def run(cmd, env_threads, log):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": str(env_threads),
           "OPENBLAS_NUM_THREADS": str(env_threads), "MKL_NUM_THREADS": str(env_threads)}
    with open(log, "w") as fh:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=fh, stderr=subprocess.STDOUT, env=env)


def per_dataset(path):
    return {r["dataset"]: r for r in json.load(open(path))["per_dataset"]}


def gold(ds):
    """Test GT exactly as src/eval/evaluate_four_datasets.py loads it."""
    g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
    return {str(v): np.asarray(g["y4"][i], dtype=np.int8) for i, v in enumerate(g["video_ids"]) if str(g["split"][i]) == "test"}


def per_video_within(y, curves):
    out = {}
    for vid, s in curves.items():
        if vid in y:
            out[vid] = within_video_macro({vid: y[vid]}, {vid: np.asarray(s, float)})["within_video_macro_ROC_AUC"]
    return out


def boot(a, b):
    """Paired video bootstrap of mean(a - b) over videos where both are defined."""
    vids = sorted(v for v in a if a[v] is not None and b.get(v) is not None)
    d = np.array([a[v] - b[v] for v in vids], float)
    if not len(d):
        return {"n": 0}
    rng = np.random.default_rng(0)
    means = d[rng.integers(len(d), size=(BOOT, len(d)))].mean(1)
    top = np.argsort(-d)[:3]
    rest = np.delete(d, top)
    return {"n": len(d), "mean": float(d.mean()), "ci95": [float(x) for x in np.quantile(means, [.025, .975])],
            "improved": int((d > 1e-12).sum()), "worse": int((d < -1e-12).sum()),
            "mean_without_top3": float(rest.mean()) if len(rest) else None,
            "top3": [[vids[i], float(d[i])] for i in top]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    ap.add_argument("--reads", type=Path, help="measure.py run directory with one sub-directory per arm")
    ap.add_argument("--arm-map", nargs="*", default=[], metavar="ARM=DIR",
                    help="dry runs only: read arm ARM from DIR/predictions.jsonl instead of <reads>/<ARM>")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--out", type=Path, help="default <reads>_analysis")
    ap.add_argument("--jobs", type=int, default=5, help="parallel CPU decodes")
    ap.add_argument("--threads", type=int, default=2, help="BLAS threads per decode")
    ap.add_argument("--decoder-python", default=sys.executable)
    ap.add_argument("--reuse", action="store_true", help="reuse existing raw / decoded outputs under --out")
    a = ap.parse_args()
    arms = list(STAGE_ARMS[a.stage])
    datasets = list(a.datasets or STAGE_DATASETS[a.stage])
    assert set(datasets) <= set(STAGE_DATASETS[a.stage]), (a.stage, datasets)
    src = {arm: (a.reads / arm if a.reads else None) for arm in arms}
    for item in a.arm_map:
        arm, path = item.split("=", 1)
        assert arm in arms, arm
        src[arm] = Path(path)
    assert all(src.values()), "give --reads or every arm in --arm-map"
    out = a.out or Path(str(a.reads) + "_analysis")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run.log").write_text(socket.gethostname() + "\n")
    (out / "run.pid").write_text(str(os.getpid()))
    (out / "config.json").write_text(json.dumps({**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                                                 "arms": arms, "datasets_used": datasets, "date": time.strftime("%Y-%m-%d"),
                                                 "host": socket.gethostname(), "decoder": DECODER, "decoder_flags": DECODER_FLAGS,
                                                 "code": "experiments/20261025_m1_act_state/stage1.py; implementation tag c50-a1-v1"},
                                                indent=2) + "\n")

    def log(*parts):
        msg = " ".join(str(x) for x in parts)
        print(msg, flush=True)
        with open(out / "run.log", "a") as fh:
            fh.write(msg + "\n")
    log("host", socket.gethostname())
    t0 = time.time()
    # decoder flags = the r6 raw-key run's flags
    rk = json.load(open(ROOT / "runs/20260926_twolevel/final_rawkey/config.json"))
    bad = {k: (rk.get(k), v) for k, v in DECODER_CONFIG.items() if rk.get(k) != v}
    assert not bad, ("decoder flags differ from the r6 raw-key run", bad)
    reads = {arm: jl(p / "predictions.jsonl") for arm, p in src.items()}
    keys = check_arms(reads, datasets)
    arm_dirs = dict(src)
    if a.stage == 1:
        recon = reconstruct(reads["all_local"], reads["A1"], keys)
        write_jsonl(out / "reads/A1_recon/predictions.jsonl", recon)
        (out / "reads/A1_recon/config.json").write_text(json.dumps(
            {"rule": "all_local reads mapped per video and modality to A1's own per-video mean and sd (population sd; "
                     "offset only if < 2 windows or sd 0); z = max over modalities; no labels",
             "sources": {"all_local": str(src["all_local"]), "A1": str(src["A1"])}, "date": time.strftime("%Y-%m-%d"),
             "code": "experiments/20261025_m1_act_state/stage1.py (c50-a1-v1)"}, indent=2) + "\n")
        reads["A1_recon"] = {(r["dataset"], r["video_id"]): r for r in recon}
        arm_dirs["A1_recon"] = out / "reads/A1_recon"
    all_arms = list(arm_dirs)
    # raw (undecoded max curve) metrics of every arm and of the r6 reads
    raw_jobs = {arm: (arm_dirs[arm] / "predictions.jsonl", out / "raw" / arm) for arm in all_arms}
    raw_jobs["r6"] = (ROOT / R6[datasets[0]]["reads"] / "predictions.jsonl", out / "raw" / "r6")
    dec_jobs = {arm: out / "decoded" / arm for arm in all_arms}

    def do_raw(item):
        arm, (pred, d) = item
        if a.reuse and (d / "metrics.json").exists():
            return
        d.mkdir(parents=True, exist_ok=True)
        run([a.decoder_python, str(ROOT / "src/eval/evaluate_four_datasets.py"), "--predictions", str(pred),
             "--gt-dir", str(ROOT / "data/gt_4fps"), "--datasets", *datasets, "--out", str(d / "metrics.json")],
            a.threads, d / "evaluate.log")

    def do_decode(arm):
        d = dec_jobs[arm]
        if a.reuse and (d / "metrics.json").exists():
            return
        run([a.decoder_python, str(ROOT / DECODER), "--run", str(arm_dirs[arm]), "--datasets", *datasets,
             *DECODER_FLAGS, "--out-root", str(out / "decoded"), "--tag", arm], a.threads, out / f"decode_{arm}.log")

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        list(ex.map(do_raw, raw_jobs.items()))
        list(ex.map(do_decode, all_arms))
    log(f"decoded {len(all_arms)} arms in {time.time() - t0:.0f} s")
    raw_m = {arm: per_dataset(d / "metrics.json") for arm, (_, d) in raw_jobs.items()}
    dec_m = {arm: per_dataset(d / "metrics.json") for arm, d in dec_jobs.items()}
    dec_m["r6"] = per_dataset(ROOT / R6[datasets[0]]["decoded"] / "metrics.json")
    for arm in all_arms:
        cfg = json.load(open(dec_jobs[arm] / "config.json"))
        bad = {k: (cfg.get(k), v) for k, v in {**DECODER_CONFIG, **DECODER_NEWER_DEFAULTS}.items()
               if k in cfg and cfg.get(k) != v or (k in DECODER_CONFIG and k not in cfg)}
        assert not bad, (arm, "decoded with other flags", bad)
    dec_curves = {arm: jl(d / "predictions.jsonl") for arm, d in dec_jobs.items()}
    dec_curves["r6"] = jl(ROOT / R6[datasets[0]]["decoded"] / "predictions.jsonl")
    raw_curves = {**{arm: reads[arm] for arm in all_arms}, "r6": jl(ROOT / R6[datasets[0]]["reads"] / "predictions.jsonl")}
    results, rows_csv, lines = {}, [], ["dataset\tarm\tROC\tPR\twithin\traw_ROC\traw_PR\traw_within\n"]
    for ds in datasets:
        y = gold(ds)
        pv = {arm: per_video_within(y, {k[1]: r["score_curve"] for k, r in c.items() if k[0] == ds})
              for arm, c in dec_curves.items()}
        pvr = {arm: per_video_within(y, {k[1]: r["score_curve"] for k, r in c.items() if k[0] == ds})
               for arm, c in raw_curves.items()}
        for arm in pv:   # the per-video means reproduce the evaluator's corpus within exactly
            vals = [v for v in pv[arm].values() if v is not None]
            assert abs(np.mean(vals) - dec_m[arm][ds]["within_video_macro_ROC_AUC"]) < 1e-12, (arm, ds)
        final = {arm: {m: dec_m[arm][ds][m] for m in METRICS} for arm in dec_m}
        raw = {arm: {m: raw_m[arm][ds][m] for m in METRICS} for arm in raw_m}
        comp = {"A1_vs_r6": boot(pv["A1"], pv["r6"]), "A1_vs_all_local": boot(pv["A1"], pv["all_local"]),
                "all_local_vs_r6": boot(pv["all_local"], pv["r6"]),
                "raw_A1_vs_r6": boot(pvr["A1"], pvr["r6"]), "raw_A1_vs_all_local": boot(pvr["A1"], pvr["all_local"])}
        if a.stage == 1:
            comp.update({"A1_vs_A1_other": boot(pv["A1"], pv["A1_other"]), "A1_vs_A1_recon": boot(pv["A1"], pv["A1_recon"]),
                         "base_vs_r6": boot(pv["base"], pv["r6"]), "raw_A1_vs_A1_other": boot(pvr["A1"], pvr["A1_other"])})
        W = "within_video_macro_ROC_AUC"
        gates = {}
        if a.stage == 1:
            gates = {
                "a_A1_within_ge_r6_minus_.005": final["A1"][W] >= G_A_WITHIN[ds],
                "b_A1_pooled_ge_floors": final["A1"]["frame_ROC_AUC"] >= G_B_POOLED[ds][0]
                                         and final["A1"]["frame_PR_AUC"] >= G_B_POOLED[ds][1],
                "c_A1_minus_all_local_ge_.01": final["A1"][W] - final["all_local"][W] >= G_CDE_GAIN,
                "d_A1_minus_A1_other_ge_.01_and_CI_low_gt_0": final["A1"][W] - final["A1_other"][W] >= G_CDE_GAIN
                                                              and comp["A1_vs_A1_other"]["ci95"][0] > 0,
                "e_A1_minus_recon_ge_.01": final["A1"][W] - final["A1_recon"][W] >= G_CDE_GAIN,
                "f_A1_raw_within_ge_floor": raw["A1"][W] >= G_F_RAW[ds]}
            base_dz = {m: abs(final["base"][m] - final["r6"][m]) for m in METRICS}
        else:
            fl = S2_FLOORS[ds]
            gates = {"A1_within_ge_.6331": final["A1"][W] >= fl[W],
                     "A1_pooled_ge_.6944_.1518": final["A1"]["frame_ROC_AUC"] >= fl["frame_ROC_AUC"]
                                                 and final["A1"]["frame_PR_AUC"] >= fl["frame_PR_AUC"]}
            base_dz = None
        results[ds] = {"final": final, "raw": raw, "comparisons": comp, "gates": gates,
                       "base_replica_decoded_abs_diff_vs_r6": base_dz,
                       "thresholds": ({"a": G_A_WITHIN[ds], "b": G_B_POOLED[ds], "cde_gain": G_CDE_GAIN, "f": G_F_RAW[ds]}
                                      if a.stage == 1 else S2_FLOORS[ds])}
        for arm in final:
            rw = raw.get(arm, {m: float("nan") for m in METRICS})
            lines.append(f"{ds}\t{arm}\t" + "\t".join(f"{final[arm][m]:.4f}" for m in METRICS) + "\t"
                         + "\t".join(f"{rw[m]:.4f}" for m in METRICS) + "\n")
        for vid in sorted(pv["A1"]):
            if pv["A1"][vid] is None:
                continue
            rows_csv.append({"dataset": ds, "video_id": vid, **{arm: pv[arm].get(vid) for arm in pv},
                             **{arm + "_raw": pvr[arm].get(vid) for arm in pvr}})
    if a.stage == 1:
        both = {g: all(results[ds]["gates"][g] for ds in datasets) for g in results[datasets[0]]["gates"]}
        kill = not (both["a_A1_within_ge_r6_minus_.005"] and both["d_A1_minus_A1_other_ge_.01_and_CI_low_gt_0"])
        verdict = "PASS" if all(both.values()) else ("FAIL_KILL" if kill else "FAIL")
    else:
        both = {g: all(results[ds]["gates"][g] for ds in datasets) for g in results[datasets[0]]["gates"]}
        kill = None
        verdict = "PASS" if all(both.values()) else "FAIL"
    summary = {"stage": a.stage, "datasets": results, "gates_all_corpora": both, "kill": kill, "verdict": verdict,
               "scope": "development-selected (read on test); unchanged r6 decoder, parameters refit label-free per arm",
               "sources": {arm: str(p) for arm, p in arm_dirs.items()},
               "metric_sources": {arm: str(d / "metrics.json") for arm, d in dec_jobs.items()},
               "r6_reference": {ds: R6[ds] for ds in datasets}, "host": socket.gethostname(),
               "date": time.strftime("%Y-%m-%d"), "code": "experiments/20261025_m1_act_state/stage1.py (c50-a1-v1)",
               "bootstrap": {"resamples": BOOT, "seed": 0, "unit": "video (within-eligible)"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "table.tsv").write_text("".join(lines))
    if rows_csv:
        with open(out / "per_video.csv", "w") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]))
            w.writeheader()
            w.writerows(rows_csv)
    log("".join(lines).rstrip("\n"))
    for ds in datasets:
        c = results[ds]["comparisons"]
        log(ds, "paired within mean [95% CI]: " + "  ".join(
            f"{k} {v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]" for k, v in c.items() if v.get("n")))
        if results[ds]["base_replica_decoded_abs_diff_vs_r6"] is not None:
            log(ds, "base replica decoded |diff| vs r6 raw-key:", results[ds]["base_replica_decoded_abs_diff_vs_r6"])
        for g, ok in results[ds]["gates"].items():
            log(f"{ds:12s} {g:46s} {'PASS' if ok else 'FAIL'}")
    log(f"STAGE{a.stage}_VERDICT {verdict}" + ("" if kill is None else f"  KILL {'yes' if kill else 'no'}") + f"  ({out})")


if __name__ == "__main__":
    main()
