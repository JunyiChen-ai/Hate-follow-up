#!/usr/bin/env python3
"""LESS proposal-survival semi-Markov decoder.

The latent multimodal posterior supplies semantic emissions.  Proposal rank
survival supplies an independent geometric emission.  Their odds multiply,
and exact decoding ranges over every pair of native proposal endpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path: Path, method=None):
    output = {}
    for row in map(json.loads, path.open()):
        if method is None or row.get("method") == method:
            output[(row["dataset"], row["video_id"])] = row
    return output


def fields(row: dict, posterior: np.ndarray):
    proposals = row["proposals"][:8]
    length = len(posterior)
    support = np.zeros(length, dtype=float)
    endpoints = set()
    for proposal in proposals:
        lo = max(0, min(length - 1, int(math.floor(float(proposal["start"]) * 4))))
        hi = max(lo + 1, min(length, int(math.ceil(float(proposal["end"]) * 4))))
        support[lo:hi] += 1
        endpoints.update((lo, hi))
    persistence_probability = (support + 1) / (len(proposals) + 2)
    semantic = np.log(np.clip(posterior, 1e-6, 1 - 1e-6) /
                      np.clip(1 - posterior, 1e-6, 1 - 1e-6))
    geometric = np.log(persistence_probability / (1 - persistence_probability))
    return {"semantic": semantic, "geometric": geometric,
            "product": semantic + geometric}, sorted(endpoints)


def decode(values: np.ndarray, endpoints: list[int], objective: str):
    cumulative = np.r_[0.0, np.cumsum(values)]
    options = []
    for left_index, left in enumerate(endpoints):
        for right in endpoints[left_index + 1:]:
            if right <= left:
                continue
            total = float(cumulative[right] - cumulative[left])
            width = right - left
            if objective == "sum":
                score = total
            elif objective == "standardized":
                score = total / math.sqrt(width)
            else:
                raise ValueError(objective)
            options.append({"left": left, "right": right, "objective": score,
                            "sum_log_odds": total, "width_frames": width})
    return max(options, key=lambda row: row["objective"], default=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--cca", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    posteriors = load(args.posterior, "less_3v_full_v1")
    proposals = load(args.proposals)
    cca = load(args.cca)
    keys = sorted(set(posteriors) & set(proposals) & set(cca))
    summary = {}
    configurations = [
        (field, objective, gate)
        for field in ("semantic", "geometric", "product")
        for objective in ("sum", "standardized")
        for gate in ("cca", "bayes")]
    for field_name, objective, gate in configurations:
        method = f"less_3v_sm_{field_name}_{objective}_{gate}_v1"
        nonempty = 0
        for key in keys:
            posterior_row = posteriors[key]
            posterior = np.asarray(posterior_row["score_curve"], dtype=float)
            evidence_fields, endpoints = fields(proposals[key], posterior)
            chosen = decode(evidence_fields[field_name], endpoints, objective)
            keep = bool(cca[key]["intervals"]) if gate == "cca" else (
                chosen is not None and chosen["sum_log_odds"] > 0)
            intervals = []
            if keep and chosen is not None:
                nonempty += 1
                left, right = chosen["left"], chosen["right"]
                intervals = [Interval(left / 4, min(float(posterior_row["duration"]), right / 4),
                                      float(posterior[left:right].mean()))]
            append_jsonl(args.out, Prediction(
                method, key[0], key[1], float(posterior_row["duration"]),
                score_curve=posterior.tolist(), intervals=intervals, calls=0,
                modality_evidence={**posterior_row.get("modality_evidence", {}),
                                   "field": field_name, "objective": objective,
                                   "existence_gate": gate, "chosen": chosen,
                                   "proposal_endpoint_count": len(endpoints)},
                raw={"gt_access": False, "decoder": "exact_one_event_semi_markov",
                     "candidate_boundaries": "native_top8_proposal_endpoints",
                     "persistence_probability": "laplace_rank_survival",
                     "combination": "product_of_odds_no_weight"}))
        summary[method] = nonempty
    print(json.dumps({"n": len(keys), "nonempty": summary}, indent=2))


if __name__ == "__main__":
    main()
