#!/usr/bin/env python3
"""Proposal-lattice semi-Markov decoder for frozen LESS posteriors.

Candidate intervals are native T3AL proposals and their rank-prefix hulls.
The decoder compares each one-event state against EMPTY using posterior log
Bayes factors.  No boundary, duration, or evaluation label is fitted here.
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


def load(path: Path, method: str | None = None) -> dict[tuple[str, str], dict]:
    output = {}
    for row in map(json.loads, path.open()):
        if method is None or row.get("method") == method:
            output[(row["dataset"], row["video_id"])] = row
    return output


def candidate_lattice(row: dict, duration: float) -> list[dict]:
    proposals = sorted(row.get("proposals", []), key=lambda item: int(item.get("rank", 999)))
    output = []
    seen = set()
    for proposal in proposals:
        interval = (max(0.0, float(proposal["start"])),
                    min(duration, float(proposal["end"])))
        rounded = tuple(round(value, 6) for value in interval)
        if interval[1] > interval[0] and rounded not in seen:
            seen.add(rounded)
            output.append({"start": interval[0], "end": interval[1],
                           "kind": "native", "rank": int(proposal.get("rank", 0))})
    for count in range(1, len(proposals) + 1):
        interval = (max(0.0, min(float(item["start"]) for item in proposals[:count])),
                    min(duration, max(float(item["end"]) for item in proposals[:count])))
        rounded = tuple(round(value, 6) for value in interval)
        if interval[1] > interval[0] and rounded not in seen:
            seen.add(rounded)
            output.append({"start": interval[0], "end": interval[1],
                           "kind": "prefix_hull", "rank": count})
    return output


def select(candidates: list[dict], posterior: np.ndarray, rule: str) -> tuple[dict | None, list[dict]]:
    log_odds = np.log(np.clip(posterior, 1e-6, 1 - 1e-6) /
                      np.clip(1 - posterior, 1e-6, 1 - 1e-6))
    scored = []
    total_frames = len(posterior)
    for candidate in candidates:
        start = max(0, int(math.floor(candidate["start"] * 4)))
        end = min(total_frames, max(start + 1, int(math.ceil(candidate["end"] * 4))))
        values = log_odds[start:end]
        raw_bayes_factor = float(values.sum())
        if rule == "bayes":
            objective = raw_bayes_factor
        elif rule == "mdl":
            objective = raw_bayes_factor - 2 * math.log(max(total_frames, 2))
        elif rule == "density":
            objective = float(values.mean())
        else:
            raise ValueError(rule)
        scored.append({**candidate, "start_frame": start, "end_frame": end,
                       "log_bayes_factor": raw_bayes_factor,
                       "objective": objective,
                       "posterior_mean": float(posterior[start:end].mean())})
    best = max(scored, key=lambda item: item["objective"], default=None)
    return (best if best is not None and best["objective"] > 0 else None), scored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    posterior_rows = load(args.posterior, "less_3v_full_v1")
    proposals = load(args.proposals)
    keys = sorted(set(posterior_rows) & set(proposals))
    summary = {}
    for rule in ("bayes", "mdl", "density"):
        method = f"less_3v_lattice_{rule}_v1"
        counts = {"empty": 0, "native": 0, "prefix_hull": 0}
        for key in keys:
            row = posterior_rows[key]
            posterior = np.asarray(row["score_curve"], dtype=float)
            lattice = candidate_lattice(proposals[key], float(row["duration"]))
            chosen, scored = select(lattice, posterior, rule)
            intervals = []
            if chosen is None:
                counts["empty"] += 1
            else:
                counts[chosen["kind"]] += 1
                intervals = [Interval(chosen["start"], chosen["end"],
                                      chosen["posterior_mean"])]
            append_jsonl(args.out, Prediction(
                method, key[0], key[1], float(row["duration"]),
                score_curve=posterior.tolist(), intervals=intervals, calls=0,
                modality_evidence={**row.get("modality_evidence", {}),
                                   "decoder_rule": rule,
                                   "chosen_state": chosen,
                                   "candidate_count": len(scored)},
                raw={"gt_access": False, "candidate_states": "T3AL_native_plus_prefix_hulls",
                     "empty_objective": 0.0,
                     "event_objective": {"bayes": "sum_log_posterior_odds",
                                         "mdl": "sum_log_odds_minus_universal_boundary_code",
                                         "density": "mean_log_posterior_odds"}[rule]}))
        summary[method] = counts
    print(json.dumps({"n": len(keys), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
