#!/usr/bin/env python3
"""Label-blind structured re-decoding of saved MELT phase fields.

This program never reads annotations.  It is used to test whether the semantic
ENTER/SUPPORT/EXIT roles contribute beyond merging three positive tokens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.run_counterfactual_evidence import ecdf
from scripts.idea_discovery.run_melt import PHASES, dense_bins
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


STATES = ("OUT", "ENTER", "SUPPORT", "EXIT")


def phase_viterbi(phase: np.ndarray) -> tuple[np.ndarray, float]:
    """Decode legal OUT→ENTER→SUPPORT*→EXIT→OUT lifecycle paths."""
    phase = np.asarray(phase, dtype=float)
    if phase.ndim != 2 or phase.shape[1] != len(PHASES):
        raise ValueError("expected an N x 5 MELT phase field")
    eps = 1e-12
    emit = np.column_stack([
        np.maximum(phase[:, PHASES.index("OUT")],
                   phase[:, PHASES.index("UNKNOWN")]),
        phase[:, PHASES.index("ENTER")],
        phase[:, PHASES.index("SUPPORT")],
        phase[:, PHASES.index("EXIT")],
    ])
    emit = np.log(np.clip(emit, eps, 1.0))
    legal = {
        0: (0, 1),       # background can persist or start an event
        1: (2, 3),       # ENTER must become support or directly exit
        2: (2, 3),       # SUPPORT persists until EXIT
        3: (0, 1),       # a completed event returns to OUT or restarts
    }
    n, k = emit.shape
    score = np.full((n, k), -np.inf)
    back = np.full((n, k), -1, dtype=int)
    score[0, 0] = emit[0, 0]
    score[0, 1] = emit[0, 1]
    for t in range(1, n):
        for prev, next_states in legal.items():
            for state in next_states:
                candidate = score[t - 1, prev] + emit[t, state]
                if candidate > score[t, state]:
                    score[t, state] = candidate
                    back[t, state] = prev
    # An unfinished ENTER/SUPPORT trajectory is not a complete lifecycle.
    state = max((0, 3), key=lambda s: score[-1, s])
    path = np.empty(n, dtype=int)
    path[-1] = state
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path, float(score[-1, state])


def structured_intervals(phase: np.ndarray, duration: float, citations: list[int]):
    path, path_score = phase_viterbi(phase)
    active = path != STATES.index("OUT")
    bounds = np.flatnonzero(np.diff(np.r_[False, active, False])).reshape(-1, 2)
    event_prob = np.asarray(phase)[:, 1:4].sum(1)
    intervals = []
    kept = []
    for start, end in bounds:
        if citations and not any(start - 1 <= c <= end for c in citations):
            continue
        intervals.append(Interval(start / len(path) * duration,
                                  end / len(path) * duration,
                                  float(event_prob[start:end].mean())))
        kept.append([int(start), int(end)])
    return intervals, event_prob, {
        "state_path": [STATES[x] for x in path],
        "kept_paths": kept,
        "path_log_likelihood": path_score,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-method", default="melt_adaptive")
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing to append to existing output: {args.out}")
    with args.input.open(encoding="utf-8") as fh:
        source = [json.loads(line) for line in fh]
    for row in source:
        if row.get("method") != args.source_method:
            continue
        if row.get("error"):
            pred = Prediction("melt_phase_viterbi", row["dataset"], row["video_id"],
                              float(row["duration"]), error=row["error"],
                              raw={"source": str(args.input)})
            append_jsonl(args.out, pred)
            continue
        evidence = row["modality_evidence"]
        phase = np.asarray(evidence["phase_scores"], dtype=float)
        citations = [int(x) for x in evidence.get("evidence_bins", [])]
        intervals, event_prob, meta = structured_intervals(
            phase, float(row["duration"]), citations)
        source_curve = np.asarray(row["score_curve"], dtype=float)
        ranking_source = evidence.get("ranking_source")
        curve = (dense_bins(event_prob, len(source_curve)) if ranking_source == "phase_field"
                 else source_curve)
        pred = Prediction(
            "melt_phase_viterbi", row["dataset"], row["video_id"],
            float(row["duration"]), score_curve=curve.tolist(), intervals=intervals,
            calls=0, modality_evidence={"decoder": meta, "evidence_bins": citations,
                                       "ranking_source": ranking_source},
            raw={"source": str(args.input), "decoder": "fixed_legal_phase_viterbi_v1"})
        append_jsonl(args.out, pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
