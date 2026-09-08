#!/usr/bin/env python3
"""Evaluate role-separated coarse-to-dense multimodal localization.

Transcript evidence orders coarse chunks.  A visual query only resolves ranks
inside each chunk, while frozen visual change points refine proposal edges.
The label-blind32 cohort is treated as development data for boundary settings;
all remaining common videos are reported separately as confirmation data.
"""
from pathlib import Path
import json, math, sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from prequential_write_gate_pilot import ecdf, interp, interval_counts, metrics, spans, text_curve

ROOT = Path(__file__).resolve().parents[2]
GT = Path("/home/jehc223/Retrieval-hate/data/gt/frame_gt_4fps")
CURVES = ROOT / "results/idea_discovery/topology_query_v2"
OUT = ROOT / "results/idea_discovery/role_separated_c2d/eval.json"
DATASETS = ["HateMM", "HateClipSeg", "MHC", "MHC_zh"]
METHODS = ["fixed", "ordinal", "topology", "shuffle", "reverse"]
ASR = {
    "HateMM": ROOT / "results/hatemm_localization/per_chunk.jsonl",
    "HateClipSeg": ROOT / "results/reproduction/ours/hateclipseg/per_chunk.jsonl",
    "MHC": ROOT / "results/reproduction/ours/mhclip_en/per_chunk.jsonl",
    "MHC_zh": ROOT / "results/reproduction/ours/mhclip_zh/per_chunk.jsonl",
}
DEV = {(r["dataset"], r["video_id"]) for r in map(json.loads, (ROOT / "results/idea_discovery/causal_credit/label_blind32.jsonl").read_text().splitlines())}
# The ordinal confidence margin is .4.  The unprojected visual residual gets a
# .1 budget and the order-projected residual gets the remaining .3; hence even
# an extreme visual disagreement cannot overturn a high-confidence language
# pair, while uncertain pairs remain visually rankable.
VISUAL_BUDGET = 0.1
ORDER_BUDGET = 0.3
BOUNDARY_RADIUS = 8
BOUNDARY_SIGMA = 2


def add_counts(total, y, score):
    tp, fp, fn = interval_counts(y, score)
    total[0] += tp
    total[1] += fp
    total[2] += fn


def f1(counts):
    tp, fp, fn = counts
    return 2 * tp / max(1, 2 * tp + fp + fn)


def bootstrap(values, n=50000):
    values = np.asarray(values)
    rng = np.random.default_rng(20260826)
    draws = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(n)])
    return {"delta": float(values.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(), "n": len(values)}


rank_items = {m: {d: [] for d in DATASETS} for m in METHODS}
rank_auc = {m: {} for m in METHODS}
teacher_items = {d: [] for d in DATASETS}
teacher_auc = {}
boundary = {split: {"teacher": [0, 0, 0], "fixed_snap": [0, 0, 0]} for split in ["dev32", "held256"]}

for dataset in DATASETS:
    chunks = {}
    for line in ASR[dataset].read_text().splitlines():
        row = json.loads(line)
        chunks.setdefault(row["video_id"], []).append(row)
    gt = np.load(GT / f"{dataset}.npz", allow_pickle=True)
    gt_map = {str(v): (float(d), np.asarray(y, int), str(s)) for v, d, y, s in zip(gt["video_ids"], gt["duration"], gt["y4"], gt["split"])}
    common = {v for v, (_, _, split) in gt_map.items() if split == "test" and v in chunks}
    for method in METHODS:
        common &= {p.stem for p in (CURVES / method / dataset).glob("*.npy")}
    for vid in sorted(common):
        duration, y, _ = gt_map[vid]
        rows = sorted(chunks[vid], key=lambda r: (float(r["span"][0]), float(r["span"][1])))
        teacher = ecdf(text_curve(rows, duration, len(y), 4)[0])
        teacher_items[dataset].append((y, teacher))
        if len(np.unique(y)) == 2:
            teacher_auc[(dataset, vid)] = float(roc_auc_score(y, teacher))

        # Boundary branch: use the frozen query so semantic adaptation cannot
        # manufacture change points.  Settings were selected on DEV only.
        split = "dev32" if (dataset, vid) in DEV else "held256"
        coarse = teacher >= np.quantile(teacher, .8)
        add_counts(boundary[split]["teacher"], y, coarse.astype(float))
        frozen = interp(ecdf(np.load(CURVES / "fixed" / dataset / f"{vid}.npy")), len(y))
        smooth = gaussian_filter1d(frozen, BOUNDARY_SIGMA)
        change = np.abs(np.diff(smooth, prepend=smooth[0]))
        snapped = np.zeros(len(y), dtype=bool)
        for lo, hi in spans(coarse):
            left = np.arange(max(0, lo - BOUNDARY_RADIUS), min(len(y), lo + BOUNDARY_RADIUS + 1))
            right = np.arange(max(0, hi - BOUNDARY_RADIUS), min(len(y), hi + BOUNDARY_RADIUS + 1))
            new_lo = int(left[np.argmax(change[left])])
            new_hi = int(right[np.argmax(change[right])])
            if new_hi <= new_lo:
                new_lo, new_hi = lo, hi
            snapped[new_lo:new_hi] = True
        add_counts(boundary[split]["fixed_snap"], y, snapped.astype(float))

        # Ranking branch: minimally project visual chunk means onto transcript
        # order, then combine bounded raw and projected visual residuals with
        # the coarse evidence.  This is a partial order, not hard text control.
        for method in METHODS:
            visual = interp(ecdf(np.load(CURVES / method / dataset / f"{vid}.npy")), len(y))
            chunk_means = []
            confidences = []
            chunk_spans = []
            for row in rows:
                lo = max(0, min(len(y) - 1, int(float(row["span"][0]) / duration * len(y))))
                hi = min(len(y), max(lo + 1, int(math.ceil(float(row["span"][1]) / duration * len(y)))))
                chunk_spans.append((lo, hi))
                chunk_means.append(float(visual[lo:hi].mean()))
                z = float(row.get("z_masked", row.get("z_isolated", -20)))
                confidences.append(1 / (1 + math.exp(-np.clip(z / 4, -30, 30))))
            chunk_means = np.asarray(chunk_means)
            confidences = np.asarray(confidences)
            projected = chunk_means.copy()
            if len(chunk_means) > 1 and np.ptp(confidences) > 0:
                order = np.argsort(confidences)
                projected[order] = IsotonicRegression().fit_transform(confidences[order], chunk_means[order])
            correction = np.zeros(len(y))
            count = np.zeros(len(y))
            for (lo, hi), delta in zip(chunk_spans, projected - chunk_means):
                correction[lo:hi] += delta
                count[lo:hi] += 1
            projected_visual = ecdf(visual + np.where(count > 0, correction / np.maximum(count, 1), 0))
            score = teacher + VISUAL_BUDGET * (visual - .5) + ORDER_BUDGET * (projected_visual - .5)
            rank_items[method][dataset].append((y, score))
            if len(np.unique(y)) == 2:
                rank_auc[method][(dataset, vid)] = float(roc_auc_score(y, score))

result = {
    "config": {"visual_budget": VISUAL_BUDGET, "order_budget": ORDER_BUDGET, "ordinal_margin": 0.4, "boundary_radius_frames": BOUNDARY_RADIUS, "boundary_sigma_frames": BOUNDARY_SIGMA, "development_cohort": "label_blind32"},
    "ranking": {},
    "teacher": {**{d: metrics(teacher_items[d]) for d in DATASETS}, "all": metrics(sum((teacher_items[d] for d in DATASETS), []))},
    "boundary": {split: {name: {"interval_f1_05": f1(counts), "counts": counts} for name, counts in arms.items()} for split, arms in boundary.items()},
    "paired": {},
}
for method in METHODS:
    result["ranking"][method] = {**{d: metrics(rank_items[method][d]) for d in DATASETS}, "all": metrics(sum((rank_items[method][d] for d in DATASETS), []))}
for baseline, source in [("teacher", teacher_auc)] + [(m, rank_auc[m]) for m in ["fixed", "ordinal", "shuffle", "reverse"]]:
    result["paired"][f"topology_minus_{baseline}"] = bootstrap([rank_auc["topology"][k] - source[k] for k in rank_auc["topology"]])

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
