#!/usr/bin/env python3
"""Label-free structured projection of dense LESS evidence onto event hypotheses."""
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
    rows = {}
    for row in map(json.loads, path.open()):
        if method is None or row.get("method") == method:
            rows[(row["dataset"], row["video_id"])] = row
    return rows


def candidates(row, length, duration):
    endpoints = {0, length}
    for proposal in row.get("proposals", []):
        endpoints.add(max(0, min(length, round(float(proposal["start"]) * length / duration))))
        endpoints.add(max(0, min(length, round(float(proposal["end"]) * length / duration))))
    endpoints = sorted(endpoints)
    return [(left, right) for i, left in enumerate(endpoints)
            for right in endpoints[i + 1:] if right > left]


def decode(posterior, intervals, objective):
    p = np.clip(np.asarray(posterior, dtype=float), 1e-6, 1 - 1e-6)
    log_yes, log_no = np.log(p), np.log(1 - p)
    empty_length = float(-log_no.sum())
    global_length = float(-log_yes.sum())
    prefix_delta = np.r_[0.0, np.cumsum(log_no - log_yes)]
    options = []
    for left, right in intervals:
        # Replacing background by foreground inside I changes code length by
        # sum(log(1-p)-log(p)).
        local_length = empty_length + float(prefix_delta[right] - prefix_delta[left])
        if objective == "mdl":
            local_length += math.log(max(1, len(intervals)))
        options.append((local_length, left, right))
    best_local = min(options, default=(math.inf, 0, 0))
    if objective == "mdl":
        hypotheses = [(empty_length, "EMPTY", 0, 0),
                      (global_length, "GLOBAL", 0, len(p)),
                      (best_local[0], "LOCAL", best_local[1], best_local[2])]
        return min(hypotheses)
    # Bayes-risk projection: EMPTY is the zero-gain hypothesis. GLOBAL is
    # already present in the endpoint lattice through {0,T}.
    gain = empty_length - best_local[0]
    return ((empty_length, "EMPTY", 0, 0) if gain <= 0 else
            (best_local[0], "LOCAL", best_local[1], best_local[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--posterior-method", required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    post = load(args.posterior, args.posterior_method)
    proposals = load(args.proposals)
    keys = sorted(set(post) & set(proposals))
    counts = {}
    for objective in ("mdl", "bayes-risk"):
        for readout in ("dense", "projected"):
            method = f"less_{objective}_{readout}_{args.tag}_v1"
            states = {"EMPTY": 0, "GLOBAL": 0, "LOCAL": 0}
            for key in keys:
                row = post[key]
                posterior = np.asarray(row["score_curve"], dtype=float)
                pool = candidates(proposals[key], len(posterior), float(row["duration"]))
                code_length, state, left, right = decode(posterior, pool, objective)
                states[state] += 1
                intervals = [] if state == "EMPTY" else [Interval(
                    left * float(row["duration"]) / len(posterior),
                    right * float(row["duration"]) / len(posterior),
                    float(posterior[left:right].mean()))]
                curve = posterior.copy()
                if readout == "projected":
                    mask = np.zeros(len(curve), dtype=bool)
                    mask[left:right] = state != "EMPTY"
                    curve[~mask] = 0.0
                append_jsonl(args.out, Prediction(
                    method, key[0], key[1], float(row["duration"]),
                    score_curve=curve.tolist(), intervals=intervals, calls=0,
                    modality_evidence={"posterior_source": args.posterior_method,
                                       "hypothesis": state,
                                       "code_length": code_length,
                                       "candidate_count": len(pool)},
                    raw={"gt_access": False,
                         "decoder": "structured_hypothesis_projection",
                         "objective": objective,
                         "candidate_source": "top8_proposal_endpoint_lattice",
                         "fitted_hyperparameters": 0}))
            counts[method] = states
    print(json.dumps({"n": len(keys), "states": counts}, indent=2))


if __name__ == "__main__":
    main()
