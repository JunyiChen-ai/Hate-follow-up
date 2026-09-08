#!/usr/bin/env python3
"""PoS-LESS: minimum-change projection of LESS under sparse text posets.

LESS is the dense evidence field. Timestamped transcript chunks may only impose
pairwise ordering constraints whose confidence gap exceeds a fixed margin.
Missing transcripts and infeasible projections fall back exactly to LESS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
ASR = {
    "HateMM": ROOT / "results/hatemm_localization/per_chunk.jsonl",
    "HateClipSeg": ROOT / "results/reproduction/ours/hateclipseg/per_chunk.jsonl",
    "MHC": ROOT / "results/reproduction/ours/mhclip_en/per_chunk.jsonl",
    "MHC_zh": ROOT / "results/reproduction/ours/mhclip_zh/per_chunk.jsonl",
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(np.clip(x / 4.0, -30, 30))))


def load_chunks() -> dict[tuple[str, str], list[dict]]:
    out = {}
    for dataset, path in ASR.items():
        grouped = {}
        for row in map(json.loads, path.open()):
            grouped.setdefault(str(row["video_id"]), []).append(row)
        for video_id, rows in grouped.items():
            out[(dataset, video_id)] = sorted(rows, key=lambda r: tuple(map(float, r["span"])))
    return out


def shifted(values: np.ndarray, dataset: str, video_id: str) -> np.ndarray:
    if len(values) < 2:
        return values.copy()
    digest = hashlib.sha256(f"pos-less/{dataset}/{video_id}".encode()).digest()
    offset = 1 + int.from_bytes(digest[:8], "little") % (len(values) - 1)
    return np.roll(values, offset)


def project(base: np.ndarray, spans: list[tuple[int, int]], conf: np.ndarray, margin: float,
            certify_orbit: bool = False, orbit_mode: str = "absence", relation_alpha: float = 0.25):
    edges = [(i, j) for i in range(len(spans)) for j in range(len(spans))
             if conf[i] - conf[j] >= margin]
    if certify_orbit and len(conf) > 2 and orbit_mode == "absence":
        # A relation is alignment-certified only when the same temporal pair is
        # not reproduced by any non-trivial circular reassignment of language.
        null_edges = set()
        for offset in range(1, len(conf)):
            null = np.roll(conf, offset)
            null_edges.update((i, j) for i in range(len(spans)) for j in range(len(spans))
                              if null[i] - null[j] >= margin)
        edges = [edge for edge in edges if edge not in null_edges]
    elif certify_orbit and len(conf) > 2 and orbit_mode == "empirical":
        rotations = [np.roll(conf, offset) for offset in range(1, len(conf))]
        qualified = []
        for i, j in edges:
            observed = conf[i] - conf[j]
            null = np.asarray([x[i] - x[j] for x in rotations])
            pvalue = (1 + int(np.sum(null >= observed))) / (1 + len(null))
            if pvalue <= relation_alpha:
                qualified.append((i, j))
        edges = qualified
    if not edges:
        return base.copy(), 0, True, 0.0
    membership = np.zeros((len(base), len(spans)), dtype=float)
    overlap = np.zeros(len(base), dtype=float)
    for k, (lo, hi) in enumerate(spans):
        membership[lo:hi, k] = 1.0
        overlap[lo:hi] += 1.0
    membership /= np.maximum(overlap[:, None], 1.0)
    chunk_op = np.stack([membership[lo:hi].mean(0) for lo, hi in spans])
    means = np.asarray([base[lo:hi].mean() for lo, hi in spans])
    edge_op = np.stack([chunk_op[i] - chunk_op[j] for i, j in edges])
    edge_base = np.asarray([means[i] - means[j] for i, j in edges])
    result = minimize(lambda x: 0.5 * float(x @ x), np.zeros(len(spans)), jac=lambda x: x,
                      constraints={"type": "ineq",
                                   "fun": lambda x: edge_base + edge_op @ x,
                                   "jac": lambda x: edge_op},
                      method="SLSQP", options={"ftol": 1e-10, "maxiter": 500})
    score = base + membership @ result.x
    projected_means = np.asarray([score[lo:hi].mean() for lo, hi in spans])
    feasible = bool(result.success and all(projected_means[i] >= projected_means[j] - 1e-6
                                           for i, j in edges))
    return (score if feasible else base.copy()), len(edges), feasible, float(np.linalg.norm(result.x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.4)
    parser.add_argument("--orbit-certified", action="store_true")
    parser.add_argument("--orbit-mode", choices=("absence", "empirical"), default="absence")
    parser.add_argument("--relation-alpha", type=float, default=0.25)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    chunks = load_chunks()
    audit = {"rows": 0, "with_text": 0, "changed": 0, "edges": 0, "failures": 0}
    with args.base.open() as source, args.out.open("w") as sink:
        for row in map(json.loads, source):
            # The dual-geometry file stores six methods. Keep the frozen midpoint only.
            if row["method"] != "fact_less_t3al_dualgeo_midpoint_v5":
                continue
            audit["rows"] += 1
            key = (row["dataset"], row["video_id"])
            rows = chunks.get(key, [])
            p = np.clip(np.asarray(row["score_curve"], float), 1e-5, 1 - 1e-5)
            base = np.log(p / (1 - p))
            spans, confidence = [], []
            for chunk in rows:
                lo = max(0, min(len(p) - 1, int(float(chunk["span"][0]) * 4)))
                hi = min(len(p), max(lo + 1, int(math.ceil(float(chunk["span"][1]) * 4))))
                spans.append((lo, hi))
                confidence.append(sigmoid(float(chunk.get("z_masked", chunk.get("z_isolated", -20)))))
            conf = np.asarray(confidence)
            if spans:
                audit["with_text"] += 1
            for control, controlled in (("aligned", conf), ("circular_shift", shifted(conf, *key))):
                score, n_edges, feasible, norm = project(base, spans, controlled, args.margin, args.orbit_certified, args.orbit_mode, args.relation_alpha) if spans else (base, 0, True, 0.0)
                posterior = 1 / (1 + np.exp(-np.clip(score, -30, 30)))
                changed = bool(np.max(np.abs(posterior - p)) > 1e-8)
                audit["changed"] += int(control == "aligned" and changed)
                audit["edges"] += n_edges if control == "aligned" else 0
                audit["failures"] += int(not feasible)
                output = dict(row)
                output["method"] = f"pos_less_{control}_v1"
                output["score_curve"] = posterior.tolist()
                output["raw"] = {**row.get("raw", {}), "gt_access": False,
                                 "dense_authority": "LESS", "text_role": "sparse_poset_only",
                                 "poset_margin": args.margin, "n_edges": n_edges,
                                 "orbit_certified": args.orbit_certified,
                                 "orbit_mode": args.orbit_mode, "relation_alpha": args.relation_alpha,
                                 "projection_norm": norm, "feasible": feasible,
                                 "exact_fallback": not changed}
                sink.write(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
