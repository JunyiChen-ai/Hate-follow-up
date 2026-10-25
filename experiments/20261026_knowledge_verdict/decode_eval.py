#!/usr/bin/env python3
"""Candidate 51 analysis (CPU, uoa-lab1): splice each arm's reads into the r6 reads, decode with the r6 settings, evaluate
on the subset with the repo evaluator, and compute the gates of README §4.

Arms read on the GPU (kv_measure.py) plus two CPU arms: A / A0 replace the 8B's z_video by the agent's own judgment
(written with / without the notes), mapped onto the base z_video values of the same subset videos by rank; the window
reads stay those of arm base. Test GT is read only here, for metrics and the false-positive / false-negative counts
(rule 10)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXP, ROOT, load_pack, subset_rows  # noqa: E402

PY = sys.executable
BASE_READS = {"main": ROOT / "runs/20260926_glr/base_gridA", "dehate": ROOT / "runs/20260927_dehate_external/reads_gridA"}
GROUPS = {"main": ["HateMM", "HateClipSeg"], "dehate": ["DeHate"]}
CORPORA = ["HateMM", "HateClipSeg", "DeHate"]
MAIN = ["HateMM", "HateClipSeg"]
GPU_ARMS = ["base", "K", "Kp", "KJ", "J", "J0", "Kother", "Kshift", "Kmark"]
CPU_ARMS = {"A": "judgment", "A0": "judgment0"}
R6 = ["--noleak", "--evidence", "em", "--transform", "nscore", "--fusion", "or", "--arm", "m2", "--key", "raw",
      "--duration", "bma", "--bma-grid", "6", "--bma-prior", "length", "--min-windows", "2", "--k", "1",
      "--d-gap", "80", "--d-hate", "80", "--kind", "two"]
OUT = ROOT / "runs" / EXP
ENV = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def video_gt(ds):
    g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
    return {str(v): bool(np.asarray(y).any()) for v, s, y in zip(g["video_ids"], g["split"], g["y4"]) if s == "test"}


def agent_prob(pack, key):
    j = (pack or {}).get(key)
    if not j:
        return None
    return j["confidence"] if j["violates"] == "yes" else 1 - j["confidence"]


def cpu_arm_reads(base_reads, subset, key):
    """Arm A / A0: z_video := the base z_video value at the agent probability's rank among the subset videos of the
    corpus that have that judgment (ties share their mean rank). Videos without the judgment keep the base read."""
    out = {}
    for ds in CORPORA:
        vids = [v for v in subset[ds] if (ds, v) in base_reads]
        ps = {v: agent_prob(load_pack(ds, v), key) for v in vids}
        have = [v for v in vids if ps[v] is not None]
        if not have:
            continue
        for v in vids:
            if ps[v] is None:
                out[(ds, v)] = base_reads[(ds, v)]
        zs = np.sort([base_reads[(ds, v)]["extra"]["z_video"] for v in have])
        ranks = rankdata([ps[v] for v in have]) - 1
        mapped = np.interp(ranks, np.arange(len(have)), zs)
        for v, z in zip(have, mapped):
            r = json.loads(json.dumps(base_reads[(ds, v)]))
            r["method"] = f"kv_{'A' if key == 'judgment' else 'A0'}"
            r["extra"]["z_video"] = float(z)
            r["extra"]["agent_prob"] = float(ps[v])
            out[(ds, v)] = r
    return out


def base_reproduction(reads_base, subset):
    """README §3: arm base must reproduce the stored r6 reads (bf16 noise)."""
    res = {}
    for grp, dss in GROUPS.items():
        stored = {(r["dataset"], r["video_id"]): r for r in read_jsonl(BASE_READS[grp] / "predictions.jsonl")}
        for ds in dss:
            dz, dw, n = [], [], 0
            for v in subset[ds]:
                a, b = reads_base.get((ds, v)), stored.get((ds, v))
                if a is None or b is None:
                    continue
                n += 1
                dz.append(abs(a["extra"]["z_video"] - b["extra"]["z_video"]))
                wa = [w["z"] for w in a["extra"]["windows"]]
                wb = [w["z"] for w in b["extra"]["windows"]]
                dw.append(max(abs(x - y) for x, y in zip(wa, wb)) if len(wa) == len(wb) else float("inf"))
            res[ds] = {"n": n, "max_abs_dz_video": max(dz) if dz else None, "max_abs_dz_window": max(dw) if dw else None,
                       "median_abs_dz_video": float(np.median(dz)) if dz else None}
    return res


def decode(arm, reads_arm, subset, summary):
    for grp, dss in GROUPS.items():
        if not any((d, v) in reads_arm for d in dss for v in subset[d]):
            continue
        base = read_jsonl(BASE_READS[grp] / "predictions.jsonl")
        sp = OUT / "splice" / f"{arm}_{grp}"
        sp.mkdir(parents=True, exist_ok=True)
        n_rep = 0
        with open(sp / "predictions.jsonl", "w") as f:
            for r in base:
                k = (r["dataset"], r["video_id"])
                if k in reads_arm and k[1] in subset[k[0]]:
                    r = reads_arm[k]
                    n_rep += 1
                f.write(json.dumps(r) + "\n")
        tag = f"{arm}_{grp}"
        subprocess.run([PY, str(ROOT / "experiments/20260926_twolevel/twolevel_r2.py"), "--run", str(sp), *R6,
                        "--tag", tag, "--out-root", str(OUT / "decoded"), "--datasets", *dss], check=True, cwd=ROOT, env=ENV)
        dec = OUT / "decoded" / tag
        with open(dec / "subset_predictions.jsonl", "w") as f:
            for r in read_jsonl(dec / "predictions.jsonl"):
                if r["video_id"] in subset.get(r["dataset"], []):
                    f.write(json.dumps(r) + "\n")
        subprocess.run([PY, str(ROOT / "src/eval/evaluate_four_datasets.py"), "--predictions",
                        str(dec / "subset_predictions.jsonl"), "--gt-dir", str(ROOT / "data/gt_4fps"), "--out",
                        str(dec / "subset_metrics.json"), "--datasets", *dss], check=True, cwd=ROOT, env=ENV)
        for row in json.load(open(dec / "subset_metrics.json"))["per_dataset"]:
            summary["metrics"][f"{arm}/{row['dataset']}"] = {
                "pooled_roc": row["frame_ROC_AUC"], "pooled_pr": row["frame_PR_AUC"],
                "within": row["within_video_macro_ROC_AUC"], "n_within": row["n_videos_defined"],
                "n_videos": row["n_videos_overlap"], "replaced_reads": n_rep}


def main():
    subset = {ds: [r["video_id"] for r in subset_rows(ds)] for ds in CORPORA}
    arms = [a for a in GPU_ARMS if (OUT / f"read_{a}/predictions.jsonl").exists()]
    reads = {a: {(r["dataset"], r["video_id"]): r for r in read_jsonl(OUT / f"read_{a}/predictions.jsonl") if not r.get("error")}
             for a in arms}
    if "base" in arms:
        for a, key in CPU_ARMS.items():
            reads[a] = cpu_arm_reads(reads["base"], subset, key)
            if reads[a]:
                arms.append(a)
    summary = {"arms": arms, "metrics": {}, "gates": {}, "base_reproduction": None}
    if "base" in arms:
        summary["base_reproduction"] = base_reproduction(reads["base"], subset)
    for a in arms:
        decode(a, reads[a], subset, summary)
    M = summary["metrics"]

    # per-corpus verdict counts (GT read here only)
    pooled = {a: {"base_fp": 0, "fp_fixed": 0} for a in arms if a != "base"}
    for ds in CORPORA:
        gt = video_gt(ds)
        packs_all = {v: load_pack(ds, v) for v in subset[ds]}
        g = {"n_subset": len(subset[ds]), "n_not_in_gt": sum(v not in gt for v in subset[ds]),
             # G0 is GT-free: computed over all subset videos
             "share_with_notes": float(np.mean([bool(p and p.get("notes")) for p in packs_all.values()])),
             "mean_notes": float(np.mean([len((p or {}).get("notes") or []) for p in packs_all.values()])),
             "n_no_pack": sum(p is None for p in packs_all.values()),
             "api_errors": {c: sum(c in ((p or {}).get("errors") or {}) for p in packs_all.values())
                            for c in ["clues", "reflect", "judge", "judge0"]},
             "moderation_refusals": sum("DataInspectionFailed" in json.dumps((p or {}).get("errors") or {})
                                        for p in packs_all.values())}
        ids = [v for v in subset[ds] if v in gt and all((ds, v) in reads[a] for a in arms)]
        if not ids or "base" not in arms:
            summary["gates"][ds] = g
            continue
        zb = np.array([reads["base"][(ds, v)]["extra"]["z_video"] for v in ids])
        y = np.array([gt[v] for v in ids])
        g.update({"n": len(ids), "n_pos": int(y.sum()), "base_fp": int(((zb > 0) & ~y).sum()),
                  "base_fn": int(((zb <= 0) & y).sum()),
                  "base_video_auc": float(roc_auc_score(y, zb)) if 0 < y.sum() < len(y) else None})
        for key in ["judgment", "judgment0"]:
            js = np.array([np.nan if agent_prob(packs_all[v], key) is None else agent_prob(packs_all[v], key) for v in ids])
            ok = ~np.isnan(js)
            g[f"agent_{key}"] = {"n": int(ok.sum()),
                                 "video_auc": float(roc_auc_score(y[ok], js[ok])) if 0 < y[ok].sum() < ok.sum() else None,
                                 "fp": int(((js > .5) & ~y & ok).sum()), "fn": int(((js <= .5) & y & ok).sum())}
        fp_b = (zb > 0) & ~y
        for a in arms:
            if a == "base":
                continue
            za = np.array([reads[a][(ds, v)]["extra"]["z_video"] for v in ids])
            g[a] = {"spearman_z_vs_base": float(spearmanr(za, zb)[0]),
                    "fp_fixed": int(((za <= 0) & fp_b).sum()),
                    "fp_fixed_share": float(((za <= 0) & fp_b).sum() / max(1, fp_b.sum())),
                    "new_fn": int(((za <= 0) & (zb > 0) & y).sum()),
                    "new_fp": int(((za > 0) & (zb <= 0) & ~y).sum()),
                    "fn_fixed": int(((za > 0) & (zb <= 0) & y).sum()),
                    "video_auc": float(roc_auc_score(y, za)) if 0 < y.sum() < len(y) else None,
                    "mean_windows_with_notes": float(np.mean([reads[a][(ds, v)]["extra"].get("windows_with_notes", 0)
                                                              for v in ids]))}
            pooled[a]["base_fp"] += int(fp_b.sum())
            pooled[a]["fp_fixed"] += g[a]["fp_fixed"]
        summary["gates"][ds] = g

    # gates of README §4 (fixed before any run)
    G = summary["gates"]

    def delta(a, ds, m):
        return M[f"{a}/{ds}"][m] - M[f"base/{ds}"][m] if f"{a}/{ds}" in M and f"base/{ds}" in M else None

    def gain(a):
        d = [delta(a, ds, m) for ds in MAIN for m in ("pooled_roc", "pooled_pr")]
        return float(np.mean(d)) if all(x is not None for x in d) else None

    verdict = {"G0": None, "G1": {}, "G2": None,
               "gain_mean_main": {a: gain(a) for a in arms if a != "base"},
               "pooled_fp_fixed_share": {a: v["fp_fixed"] / max(1, v["base_fp"]) for a, v in pooled.items()}}
    if "K" in arms:
        n_sp = sum(G[ds].get("K", {}).get("spearman_z_vs_base", 1.0) < .9 for ds in CORPORA if ds in G)
        verdict["G0"] = {"notes_share_ok": all(G[ds]["share_with_notes"] >= .6 for ds in CORPORA if ds in G),
                         "n_corpora_spearman_below_.9": int(n_sp)}
        verdict["G0"]["pass"] = verdict["G0"]["notes_share_ok"] and n_sp >= 2
    for a in ["K", "KJ"]:
        if a not in arms or gain(a) is None:
            continue
        c = {"a_pooled_up_HateMM_HCS": all(delta(a, ds, m) > 0 for ds in MAIN for m in ("pooled_roc", "pooled_pr")),
             "b_fp_fixed_share_pooled_ge_.40": verdict["pooled_fp_fixed_share"][a] >= .40,
             "c_new_fn_le_1_each": all(G[ds].get(a, {}).get("new_fn", 0) <= 1 for ds in CORPORA if ds in G)}
        if "Kother" in arms and gain("Kother") is not None:
            c["d_gain_ge_2x_Kother"] = gain(a) >= 2 * max(gain("Kother"), 0.0)
        else:
            c["d_gain_ge_2x_Kother"] = None
        if a == "KJ":
            c["e_gain_above_J0"] = gain("KJ") > gain("J0") if "J0" in arms and gain("J0") is not None else None
        c["pass"] = all(v is True for v in c.values())
        verdict["G1"][a] = c
    if all(x in arms for x in ["K", "Kp", "Kshift", "Kmark"]):
        w = {f"{x}/{ds}": delta("K", ds, "within") - delta(x, ds, "within") for x in ["Kp", "Kshift", "Kmark"] for ds in MAIN}
        verdict["G2"] = {"K_within_minus": w, "pass": all(v > 0 for v in w.values()),
                         "n_within": {ds: M[f"K/{ds}"]["n_within"] for ds in MAIN}}
    summary["verdict"] = verdict
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(OUT / "analysis/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
