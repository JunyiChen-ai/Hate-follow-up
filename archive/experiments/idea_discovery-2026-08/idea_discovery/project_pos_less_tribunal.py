#!/usr/bin/env python3
"""Proposal-level temporal-randomization tribunal for PoS-LESS.

The decoder never reads labels. Each frozen geometry proposal is evaluated by
the same dense LESS field and by timestamped language against a within-video
circular-shift null. Missing language defers exactly to midpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASR = {
    "HateMM": ROOT / "results/hatemm_localization/per_chunk.jsonl",
    "HateClipSeg": ROOT / "results/reproduction/ours/hateclipseg/per_chunk.jsonl",
    "MHC": ROOT / "results/reproduction/ours/mhclip_en/per_chunk.jsonl",
    "MHC_zh": ROOT / "results/reproduction/ours/mhclip_zh/per_chunk.jsonl",
}
RULES = ("cca", "t3al", "midpoint", "union", "casa")


def chunks_by_video():
    result = {}
    for dataset, path in ASR.items():
        grouped = {}
        for row in map(json.loads, path.open()):
            grouped.setdefault(str(row["video_id"]), []).append(row)
        for video_id, rows in grouped.items():
            result[(dataset, video_id)] = sorted(rows, key=lambda r: tuple(map(float, r["span"])))
    return result


def text_curve(rows, n):
    total, count = np.zeros(n), np.zeros(n)
    for row in rows:
        lo = max(0, min(n - 1, int(float(row["span"][0]) * 4)))
        hi = min(n, max(lo + 1, int(math.ceil(float(row["span"][1]) * 4))))
        z = float(row.get("z_masked", row.get("z_isolated", -20)))
        value = 1 / (1 + math.exp(-float(np.clip(z / 4, -30, 30))))
        total[lo:hi] += value
        count[lo:hi] += 1
    return np.where(count > 0, total / np.maximum(count, 1), 0.5)


def mask(intervals, duration, n):
    out = np.zeros(n, dtype=bool)
    for interval in intervals:
        lo = max(0, min(n - 1, int(float(interval[0]) / duration * n)))
        hi = min(n, max(lo + 1, int(math.ceil(float(interval[1]) / duration * n))))
        out[lo:hi] = True
    return out


def contrast(curve, region):
    if not region.any():
        return -float("inf")
    inside = float(np.mean(curve[region]))
    outside = float(np.mean(curve[~region])) if (~region).any() else 0.0
    return inside - outside


def certificate(language, region, offsets):
    if not region.any():
        return 0.0, 1.0, 0.0
    aligned = contrast(language, region)
    null = np.asarray([contrast(np.roll(language, offset), region) for offset in offsets])
    p = (1 + int(np.sum(null >= aligned))) / (1 + len(null))
    median = float(np.median(null))
    mad = float(np.median(np.abs(null - median))) + 1e-8
    return aligned, float(p), float((aligned - median) / mad)


def rank(values, reverse=False):
    order = np.argsort(-np.asarray(values) if reverse else np.asarray(values), kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(len(values))
    return ranks


def choose(candidates, policy):
    # Candidate tuple: rule, row, visual contrast, p-value, alignment surplus.
    visual_rank = rank([x[2] for x in candidates], reverse=True)
    p_rank = rank([x[3] for x in candidates])
    surplus_rank = rank([x[4] for x in candidates], reverse=True)
    midpoint = next((i for i, x in enumerate(candidates) if x[0] == "midpoint"), 0)
    policy = policy.removesuffix("_set")
    if policy == "minimax":
        objective = np.maximum(visual_rank, np.maximum(p_rank, surplus_rank))
    elif policy == "rank_sum":
        objective = visual_rank + p_rank + surplus_rank
    elif policy == "closed":
        certified = [i for i, x in enumerate(candidates) if x[3] <= 0.25 and x[4] > 0]
        if not certified:
            return midpoint
        objective = np.full(len(candidates), 10_000)
        for i in certified:
            objective[i] = visual_rank[i]
    elif policy == "pareto":
        m = candidates[midpoint]
        admissible = [i for i, x in enumerate(candidates)
                      if x[2] >= m[2] and x[3] <= m[3] and x[4] >= m[4]
                      and (x[2] > m[2] or x[3] < m[3] or x[4] > m[4])]
        if not admissible:
            return midpoint
        objective = np.full(len(candidates), 10_000)
        for i in admissible:
            objective[i] = visual_rank[i] + p_rank[i] + surplus_rank[i]
    else:
        raise ValueError(policy)
    best = np.flatnonzero(objective == objective.min()).tolist()
    return midpoint if midpoint in best else best[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--geometry-bank", type=Path, required=True)
    parser.add_argument("--broad-bank", type=Path)
    parser.add_argument("--rules", nargs="+", choices=RULES, default=list(RULES))
    parser.add_argument("--hull-appeal", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--posterior-method", default="pos_less_aligned_v1")
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    post = {(r["method"], r["dataset"], r["video_id"]): r
            for r in map(json.loads, args.posterior.open()) if r["method"] == args.posterior_method}
    bank = {(r["method"], r["dataset"], r["video_id"]): r
            for r in map(json.loads, args.geometry_bank.open())}
    broad = {(r["dataset"], r["video_id"]): r for r in map(json.loads, args.broad_bank.open())} if args.broad_bank else {}
    chunks = chunks_by_video()
    counts = {}
    policies = ("minimax", "rank_sum", "closed", "pareto", "minimax_set", "rank_sum_set")
    with args.out.open("w") as sink:
        for (post_method, dataset, video_id), row in sorted(post.items()):
            curve = np.asarray(row["score_curve"], float)
            duration, n = float(row["duration"]), len(curve)
            language = text_curve(chunks.get((dataset, video_id), []), n)
            has_language = (dataset, video_id) in chunks
            seed = int.from_bytes(hashlib.sha256(f"tribunal/{dataset}/{video_id}".encode()).digest()[:8], "little")
            control_offset = max(1, seed % max(2, n - 1))
            offsets = sorted({max(1, int(round(n * fraction / 16))) for fraction in range(1, 16)})
            for control, evidence in (("aligned", language), ("shift", np.roll(language, control_offset))):
                candidates, seen = [], set()
                for rule in args.rules:
                    if rule == "casa":
                        candidate = broad.get((dataset, video_id))
                        if candidate is None:
                            continue
                    else:
                        candidate = bank[(f"fact_less_t3al_dualgeo_{rule}_v5", dataset, video_id)]
                    identity = tuple(tuple(map(float, x[:2])) for x in candidate["intervals"])
                    if identity in seen:
                        continue
                    seen.add(identity)
                    region = mask(candidate["intervals"], duration, n)
                    v = contrast(curve, region)
                    aligned, pvalue, surplus = certificate(evidence, region, offsets) if has_language else (0.0, 1.0, 0.0)
                    candidates.append((rule, candidate, v, pvalue, surplus, aligned))
                for policy in policies:
                    index = choose(candidates, policy) if has_language else next((i for i, x in enumerate(candidates) if x[0] == "midpoint"), 0)
                    selected = candidates[index]
                    counts[(control, policy, selected[0])] = counts.get((control, policy, selected[0]), 0) + 1
                    output = dict(row)
                    output["method"] = f"pos_less_tribunal_{control}_{policy}_v1"
                    selected_intervals = selected[1]["intervals"]
                    if args.hull_appeal and selected[0] != "midpoint" and selected_intervals:
                        midpoint_row = bank[("fact_less_t3al_dualgeo_midpoint_v5", dataset, video_id)]
                        if midpoint_row["intervals"]:
                            a0, a1 = midpoint_row["intervals"][0][:2]
                            b0, b1 = selected_intervals[0][:2]
                            selected_intervals = [[min(float(a0), float(b0)), max(float(a1), float(b1)), 1.0]]
                    if policy.endswith("_set"):
                        midpoint_row = bank[("fact_less_t3al_dualgeo_midpoint_v5", dataset, video_id)]
                        selected_intervals = midpoint_row["intervals"] + [x for x in selected_intervals
                                                                          if x not in midpoint_row["intervals"]]
                    output["intervals"] = selected_intervals
                    output["raw"] = {**row.get("raw", {}), "tribunal": policy,
                                     "temporal_null": "15 within-video circular shifts",
                                     "candidate_rules": list(args.rules), "hull_appeal": args.hull_appeal,
                                     "selected_rule": selected[0], "visual_contrast": selected[2],
                                     "alignment_p": selected[3], "alignment_surplus": selected[4],
                                     "has_timestamp_language": has_language, "gt_access": False}
                    sink.write(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps({"/".join(k): v for k, v in sorted(counts.items())}, indent=2))


if __name__ == "__main__":
    main()
