#!/usr/bin/env python3
"""Matched evaluation for cross-fitted coarse-to-dense query adaptation."""
from pathlib import Path
import json, sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from prequential_write_gate_pilot import ecdf, interp, metrics, text_curve

ROOT = Path(__file__).resolve().parents[2]
GT = Path("/home/jehc223/Retrieval-hate/data/gt/frame_gt_4fps")
BASE = ROOT / "results/idea_discovery/crossfit_topology"
DATASETS = ["HateMM", "HateClipSeg", "MHC", "MHC_zh"]
METHODS = {name: BASE / name for name in ["main", "shuffle", "reverse"]}
ASR = {
    "HateMM": ROOT / "results/hatemm_localization/per_chunk.jsonl",
    "HateClipSeg": ROOT / "results/reproduction/ours/hateclipseg/per_chunk.jsonl",
    "MHC": ROOT / "results/reproduction/ours/mhclip_en/per_chunk.jsonl",
    "MHC_zh": ROOT / "results/reproduction/ours/mhclip_zh/per_chunk.jsonl",
}


def bootstrap(values, n=20000):
    values = np.asarray(values)
    rng = np.random.default_rng(20260826)
    draws = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(n)])
    return {"delta": float(values.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(), "n": len(values)}


items = {m: {d: [] for d in DATASETS} for m in METHODS}
fusion = {m: {d: [] for d in DATASETS} for m in METHODS}
aucs = {m: {d: {} for d in DATASETS} for m in METHODS}
faucs = {m: {d: {} for d in DATASETS} for m in METHODS}
teacher = {d: [] for d in DATASETS}
tauc = {d: {} for d in DATASETS}

for dataset in DATASETS:
    chunks = {}
    for line in ASR[dataset].read_text().splitlines():
        row = json.loads(line)
        chunks.setdefault(row["video_id"], []).append(row)
    gt = np.load(GT / f"{dataset}.npz", allow_pickle=True)
    gt_map = {str(v): (float(d), np.asarray(y, int), str(s)) for v, d, y, s in zip(gt["video_ids"], gt["duration"], gt["y4"], gt["split"])}
    common = {v for v, (_, _, split) in gt_map.items() if split == "test" and v in chunks}
    for path in METHODS.values():
        common &= {p.stem for p in (path / dataset).glob("*.npy")}
    for vid in sorted(common):
        duration, y, _ = gt_map[vid]
        rows = sorted(chunks[vid], key=lambda r: (float(r["span"][0]), float(r["span"][1])))
        tc = ecdf(text_curve(rows, duration, len(y), 4)[0])
        teacher[dataset].append((y, tc))
        if len(np.unique(y)) == 2:
            tauc[dataset][vid] = float(roc_auc_score(y, tc))
        for method, path in METHODS.items():
            score = interp(ecdf(np.load(path / dataset / f"{vid}.npy")), len(y))
            fused = .5 * score + .5 * tc
            items[method][dataset].append((y, score))
            fusion[method][dataset].append((y, fused))
            if len(np.unique(y)) == 2:
                aucs[method][dataset][vid] = float(roc_auc_score(y, score))
                faucs[method][dataset][vid] = float(roc_auc_score(y, fused))

result = {"visual": {}, "fusion50": {}, "paired": {}, "coverage": {}}
for method in METHODS:
    result["visual"][method] = {**{d: metrics(items[method][d]) for d in DATASETS}, "all": metrics(sum((items[method][d] for d in DATASETS), []))}
    result["fusion50"][method] = {**{d: metrics(fusion[method][d]) for d in DATASETS}, "all": metrics(sum((fusion[method][d] for d in DATASETS), []))}
result["teacher"] = {**{d: metrics(teacher[d]) for d in DATASETS}, "all": metrics(sum((teacher[d] for d in DATASETS), []))}
for score_type, store in [("visual", aucs), ("fusion50", faucs)]:
    result["paired"][score_type] = {}
    for baseline in ["shuffle", "reverse"]:
        result["paired"][score_type][f"main_minus_{baseline}"] = bootstrap([
            store["main"][d][v] - store[baseline][d][v]
            for d in DATASETS for v in store["main"][d]
        ])
result["paired"]["fusion50"]["main_minus_teacher"] = bootstrap([
    faucs["main"][d][v] - tauc[d][v] for d in DATASETS for v in faucs["main"][d]
])
run = json.loads((METHODS["main"] / "run.json").read_text())
for d in DATASETS:
    rows = [r for r in run["videos"] if r["dataset"] == d]
    result["coverage"][d] = {"adapted": sum(r["pairs_even"] + r["pairs_odd"] > 0 for r in rows), "both_folds": sum(r["pairs_even"] > 0 and r["pairs_odd"] > 0 for r in rows), "total": len(rows)}
(BASE / "eval.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
