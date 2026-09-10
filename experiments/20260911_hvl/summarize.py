#!/usr/bin/env python3
"""Table of HVL runs: runs/20260911_hvl/<run>/metrics_<tag>.json plus state-chain diagnostics from predictions.

Columns per corpus: pooled ROC / PR / within for the requested composition; diagnostics: fraction of chain
answers per state, mean run length of consecutive "present" windows, Spearman(z_rev, z_video),
Spearman(z_rev, number of present windows). Writes runs/20260911_hvl/table.md.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "runs/20260911_hvl"
DS = ["HateMM", "HateClipSeg"]
COMPS = sys.argv[1:] or ["izv_rev_rrank", "ispvl_rrank", "izv_plus_mean_rrank"]
REF = {"SPVL-r2 full set (cache path)": ROOT / "runs/20260910_spvl/mllm/q3vl-8b/full/metrics_izv_plus_mean_rrank.json",
       "per-window alone (SPVL winonly)": ROOT / "runs/20260910_spvl/mllm/q3vl-8b/winonly/metrics_ispvl_rrank.json"}


def cell(f):
    if not f.exists():
        return {d: "—" for d in DS}
    per = {p["dataset"]: p for p in json.load(open(f))["per_dataset"]}
    return {d: (f"{per[d]['frame_ROC_AUC']:.4f} / {per[d]['frame_PR_AUC']:.4f} / {per[d]['within_video_macro_ROC_AUC']:.4f} ({per[d]['n_videos_predicted']})"
                if d in per else "—") for d in DS}


def diag(run):
    f = run / "predictions.jsonl"
    if not f.exists():
        return "—"
    counts = {}; runs_len = []; zr = []; zv = []; npres = []
    for line in open(f):
        r = json.loads(line)
        if r.get("error"):
            continue
        e = r["extra"]
        ws = e["windows"]
        pres = [1 if w["z"] > 0 else 0 for w in ws]
        k = 0
        for p in pres + [0]:
            if p: k += 1
            elif k: runs_len.append(k); k = 0
        for w in ws:
            for key, lp in w.items():
                if key.startswith("lp_"):
                    st = int(np.argmax(lp)); counts[st] = counts.get(st, 0) + 1
        if e.get("z_rev") is not None:
            zr.append(e["z_rev"]); zv.append(e["z_video"]); npres.append(sum(pres))
    tot = sum(counts.values())
    parts = []
    if tot:
        parts.append("states s/c/x/n " + "/".join(f"{counts.get(i, 0) / tot:.2f}" for i in range(4)))
    if runs_len:
        parts.append(f"present-run {np.mean(runs_len):.1f}")
    if len(zr) > 3:
        parts.append(f"rho(zrev,zv) {spearmanr(zr, zv).correlation:.2f} rho(zrev,#pres) {spearmanr(zr, npres).correlation:.2f}")
    return "; ".join(parts) or "—"


out = ["| run | composition | " + " | ".join(f"{d} ROC / PR / within (n)" for d in DS) + " | diagnostics |", "|---|---|" + "---|" * (len(DS) + 1)]
for name, f in REF.items():
    c = cell(f); out.append(f"| {name} | — | {c[DS[0]]} | {c[DS[1]]} | — |")
for run in sorted(p for p in R.iterdir() if p.is_dir()):
    d = diag(run)
    for comp in COMPS:
        f = run / f"metrics_{comp}.json"
        if not f.exists():
            continue
        c = cell(f)
        out.append(f"| {run.name} | {comp} | {c[DS[0]]} | {c[DS[1]]} | {d if comp == COMPS[0] else ''} |")
text = "\n".join(out)
(R / "table.md").write_text(text + "\n")
print(text)
