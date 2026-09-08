#!/usr/bin/env python3
"""Evaluate a margin-qualified poset successor to T3AL projection-TTA.

Only transcript pairs whose confidence gap is at least MARGIN constrain the
dense visual score.  The decoder finds the minimum-L2 chunk-offset correction
that satisfies those graph constraints; all other ordering remains visual.
Shuffle/reverse controls transform both query preconditioning and decoding.
"""
from pathlib import Path
import hashlib, json, math, os, sys

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from prequential_write_gate_pilot import ecdf, interp, metrics, text_curve

ROOT = Path(__file__).resolve().parents[2]
GT = Path("/home/jehc223/Retrieval-hate/data/gt/frame_gt_4fps")
CURVES = ROOT / "results/idea_discovery/topology_query_v2"
OUT = ROOT / "results/idea_discovery/poset_t3al_successor/eval.json"
DATASETS = ["HateMM", "HateClipSeg", "MHC", "MHC_zh"]
METHODS = ["fixed", "ordinal", "topology", "shuffle", "reverse"]
ASR = {
    "HateMM": ROOT / "results/hatemm_localization/per_chunk.jsonl",
    "HateClipSeg": ROOT / "results/reproduction/ours/hateclipseg/per_chunk.jsonl",
    "MHC": ROOT / "results/reproduction/ours/mhclip_en/per_chunk.jsonl",
    "MHC_zh": ROOT / "results/reproduction/ours/mhclip_zh/per_chunk.jsonl",
}
DEV = {(r["dataset"], r["video_id"]) for r in map(json.loads, (ROOT / "results/idea_discovery/causal_credit/label_blind32.jsonl").read_text().splitlines())}
MARGIN = 0.4
VISUAL_BUDGET = float(os.environ.get("POSET_VISUAL_BUDGET", "0.05"))


def controlled_confidence(confidence, method, dataset, video_id):
    confidence = confidence.copy()
    if method == "shuffle" and len(confidence) > 2:
        seed = int.from_bytes(hashlib.sha256(f"parity/{dataset}/{video_id}".encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        perm = np.arange(len(confidence))
        for parity in (0, 1):
            idx = np.arange(parity, len(confidence), 2)
            perm[idx] = rng.permutation(idx)
        confidence = confidence[perm]
    elif method == "reverse":
        confidence = 1 - confidence
    return confidence


def rasterize_confidence(confidence, chunk_spans, n):
    total = np.zeros(n)
    count = np.zeros(n)
    for value, (lo, hi) in zip(confidence, chunk_spans):
        total[lo:hi] += value
        count[lo:hi] += 1
    return np.where(count > 0, total / np.maximum(count, 1), .5)


def poset_project(base, chunk_spans, confidence):
    """Minimum chunk-offset correction under margin-qualified pair constraints."""
    n_chunks = len(chunk_spans)
    edges = [(i, j) for i in range(n_chunks) for j in range(n_chunks) if confidence[i] - confidence[j] >= MARGIN]
    if not edges:
        return base.copy(), edges, True
    membership = np.zeros((len(base), n_chunks))
    overlap_count = np.zeros(len(base))
    for k, (lo, hi) in enumerate(chunk_spans):
        membership[lo:hi, k] = 1
        overlap_count[lo:hi] += 1
    membership /= np.maximum(overlap_count[:, None], 1)
    chunk_operator = np.stack([membership[lo:hi].mean(0) for lo, hi in chunk_spans])
    base_means = np.asarray([base[lo:hi].mean() for lo, hi in chunk_spans])
    edge_operator = np.stack([chunk_operator[i] - chunk_operator[j] for i, j in edges])
    edge_base = np.asarray([base_means[i] - base_means[j] for i, j in edges])
    result = minimize(
        lambda delta: .5 * float(delta @ delta),
        np.zeros(n_chunks),
        jac=lambda delta: delta,
        constraints={"type": "ineq", "fun": lambda delta: edge_base + edge_operator @ delta, "jac": lambda delta: edge_operator},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    score = base + membership @ result.x
    means = np.asarray([score[lo:hi].mean() for lo, hi in chunk_spans])
    feasible = bool(result.success and all(means[i] >= means[j] - 1e-7 for i, j in edges))
    return score, edges, feasible


items = {m: {d: [] for d in DATASETS} for m in METHODS}
held_items = {m: {d: [] for d in DATASETS} for m in METHODS}
aucs = {m: {} for m in METHODS}
teacher = {d: [] for d in DATASETS}
held_teacher = {d: [] for d in DATASETS}
teacher_auc = {}
audit = {m: {"edges": 0, "violations": 0, "solver_failures": 0} for m in METHODS}

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
    for video_id in sorted(common):
        duration, y, _ = gt_map[video_id]
        rows = sorted(chunks[video_id], key=lambda r: (float(r["span"][0]), float(r["span"][1])))
        chunk_spans = []
        confidence = []
        for row in rows:
            lo = max(0, min(len(y) - 1, int(float(row["span"][0]) / duration * len(y))))
            hi = min(len(y), max(lo + 1, int(math.ceil(float(row["span"][1]) / duration * len(y)))))
            chunk_spans.append((lo, hi))
            z = float(row.get("z_masked", row.get("z_isolated", -20)))
            confidence.append(1 / (1 + math.exp(-np.clip(z / 4, -30, 30))))
        confidence = np.asarray(confidence)
        teacher_score = text_curve(rows, duration, len(y), 4)[0]
        teacher[dataset].append((y, teacher_score))
        if (dataset, video_id) not in DEV:
            held_teacher[dataset].append((y, teacher_score))
        if len(np.unique(y)) == 2:
            teacher_auc[(dataset, video_id)] = float(roc_auc_score(y, teacher_score))
        for method in METHODS:
            controlled = controlled_confidence(confidence, method, dataset, video_id)
            coarse = rasterize_confidence(controlled, chunk_spans, len(y))
            visual = interp(ecdf(np.load(CURVES / method / dataset / f"{video_id}.npy")), len(y))
            score, edges, feasible = poset_project(coarse + VISUAL_BUDGET * (visual - .5), chunk_spans, controlled)
            means = np.asarray([score[lo:hi].mean() for lo, hi in chunk_spans])
            violations = int(sum(means[i] < means[j] - 1e-7 for i, j in edges))
            audit[method]["edges"] += len(edges)
            audit[method]["violations"] += violations
            audit[method]["solver_failures"] += int(not feasible)
            items[method][dataset].append((y, score))
            if (dataset, video_id) not in DEV:
                held_items[method][dataset].append((y, score))
            if len(np.unique(y)) == 2:
                aucs[method][(dataset, video_id)] = float(roc_auc_score(y, score))


def bootstrap(values, n=50000):
    values = np.asarray(values)
    rng = np.random.default_rng(20260826)
    draws = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(n)])
    return {"delta": float(values.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(), "n": len(values)}


result = {
    "config": {"margin": MARGIN, "visual_budget": VISUAL_BUDGET, "projection": "minimum-L2 chunk offsets", "development_cohort": "label_blind32"},
    "all": {}, "held256": {}, "teacher": {}, "paired": {"all": {}, "held256": {}}, "constraint_audit": audit,
}
for method in METHODS:
    result["all"][method] = {**{d: metrics(items[method][d]) for d in DATASETS}, "all": metrics(sum((items[method][d] for d in DATASETS), []))}
    result["held256"][method] = {**{d: metrics(held_items[method][d]) for d in DATASETS}, "all": metrics(sum((held_items[method][d] for d in DATASETS), []))}
result["teacher"] = {
    "all": {**{d: metrics(teacher[d]) for d in DATASETS}, "all": metrics(sum((teacher[d] for d in DATASETS), []))},
    "held256": {**{d: metrics(held_teacher[d]) for d in DATASETS}, "all": metrics(sum((held_teacher[d] for d in DATASETS), []))},
}
for split, keys in [("all", set(aucs["topology"])), ("held256", set(aucs["topology"]) - DEV)]:
    for baseline, source in [("teacher", teacher_auc)] + [(m, aucs[m]) for m in ["fixed", "ordinal", "shuffle", "reverse"]]:
        result["paired"][split][f"topology_minus_{baseline}"] = bootstrap([aucs["topology"][k] - source[k] for k in keys])

assert all(v["violations"] == 0 and v["solver_failures"] == 0 for v in audit.values()), audit
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
