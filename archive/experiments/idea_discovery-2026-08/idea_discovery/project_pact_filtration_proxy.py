#!/usr/bin/env python3
"""Label-blind cache proxy for PACT's proposal-evidence filtration.

This script is a mechanism kill-test, not the final interval-level language
scorer.  It uses already frozen per-chunk language logits and never reads a
target label or temporal annotation.  The output contains the aligned method
and the controls needed to decide whether fresh MLLM inference is warranted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from scripts.label_free_adapt.schema import Interval, Prediction, intervals_to_curve


ROOT = Path(__file__).resolve().parents[2]
CHUNK_SOURCES = {
    "HateMM": (ROOT / "results/hatemm_localization/per_chunk.jsonl", "z_isolated"),
    "HateClipSeg": (ROOT / "results/reproduction/ours/hateclipseg/per_chunk.jsonl", "z_sequential"),
    "MHC": (ROOT / "results/reproduction/ours/mhclip_en/per_chunk.jsonl", "z_sequential"),
    "MHC_zh": (ROOT / "results/reproduction/ours/mhclip_zh/per_chunk.jsonl", "z_sequential"),
}
NULL_LOG_ODDS = -12.0


def load_map(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row
            for row in map(json.loads, path.open())}


def load_chunks() -> dict[tuple[str, str], list[dict]]:
    """Load inference-only fields; intentionally discard label/gold columns."""
    output: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for dataset, (path, score_field) in CHUNK_SOURCES.items():
        for source in map(json.loads, path.open()):
            span, value = source.get("span"), source.get(score_field)
            if not span or len(span) != 2 or value is None:
                continue
            output[(dataset, str(source["video_id"]))].append({
                "start": float(span[0]), "end": float(span[1]), "z": float(value),
            })
    for rows in output.values():
        rows.sort(key=lambda row: (row["start"], row["end"]))
    return dict(output)


def topk_hulls(proposals: list[dict]) -> list[tuple[float, float]]:
    output = []
    for k in range(1, len(proposals) + 1):
        current = proposals[:k]
        pair = (min(float(row["start"]) for row in current),
                max(float(row["end"]) for row in current))
        if not output or pair != output[-1]:
            output.append(pair)
    return output


def filtration_states(proposals: list[dict]) -> list[dict]:
    """Collapse identical consecutive hulls while retaining rank persistence."""
    states = []
    for rank in range(1, len(proposals) + 1):
        current = proposals[:rank]
        interval = (min(float(row["start"]) for row in current),
                    max(float(row["end"]) for row in current))
        if states and states[-1]["interval"] == interval:
            states[-1]["persistence"] += 1
            states[-1]["last_rank"] = rank
        else:
            states.append({"interval": interval, "persistence": 1,
                           "first_rank": rank, "last_rank": rank})
    return states


def centered_indices(chunks: list[dict], start: float, end: float) -> list[int]:
    return [index for index, row in enumerate(chunks)
            if start <= (row["start"] + row["end"]) / 2.0 < end]


def shifted(indices: list[int], offset: int, n: int) -> list[int]:
    return sorted({min(max(index + offset, 0), n - 1) for index in indices}) if n else []


def max_score(chunks: list[dict], indices: list[int]) -> float:
    return max((chunks[index]["z"] for index in indices), default=NULL_LOG_ODDS)


def stable_warrant(chunks: list[dict], candidate: tuple[float, float],
                   closure: tuple[float, float], values: list[float]) -> tuple[bool, list[dict]]:
    inside_native = centered_indices(chunks, *candidate)
    closure_native = centered_indices(chunks, *closure)
    shell_native = sorted(set(closure_native) - set(inside_native))
    audit = []
    for offset in (-1, 0, 1):
        inside = shifted(inside_native, offset, len(chunks))
        shell = shifted(shell_native, offset, len(chunks))
        z_inside, z_shell = max_score(chunks, inside), max_score(chunks, shell)
        passed = bool(inside and shell and z_inside > NULL_LOG_ODDS and z_inside > z_shell)
        audit.append({"offset": offset, "z_inside": z_inside, "z_shell": z_shell,
                      "inside_chunks": len(inside), "shell_chunks": len(shell), "passed": passed})
        values.append(z_inside - z_shell)
    return all(row["passed"] for row in audit), audit


def shuffled_chunks(key: tuple[str, str], chunks: list[dict]) -> list[dict]:
    if len(chunks) < 2:
        return chunks
    digest = hashlib.sha256((key[0] + "\0" + key[1]).encode()).digest()
    shift = 1 + int.from_bytes(digest[:4], "big") % (len(chunks) - 1)
    scores = [row["z"] for row in chunks]
    scores = scores[shift:] + scores[:shift]
    return [{**row, "z": score} for row, score in zip(chunks, scores)]


def select(hulls: list[tuple[float, float]], chunks: list[dict],
           stable: bool = True) -> tuple[tuple[float, float] | None, dict]:
    if len(hulls) < 2 or not chunks:
        return None, {"reason": "no_filtration_or_chunks", "feasible": []}
    closure = hulls[-1]
    feasible, margins = [], []
    for index, candidate in enumerate(hulls[:-1]):
        passed, audit = stable_warrant(chunks, candidate, closure, margins)
        if not stable:
            passed = audit[1]["passed"]
        if passed:
            feasible.append({"filtration_index": index + 1, "interval": list(candidate),
                             "audit": audit})
    chosen = tuple(feasible[0]["interval"]) if feasible else None
    return chosen, {"reason": "warrant" if chosen else "abstain", "feasible": feasible,
                    "mean_margin": sum(margins) / len(margins) if margins else None}


def select_persistent(states: list[dict], chunks: list[dict],
                      require_stronger_than_closure: bool = False) -> tuple[tuple[float, float] | None, dict]:
    """Select the most visually persistent semantically warranted state.

    Persistence is an ordinal visual statistic: how many successive proposal
    ranks leave the same endpoint pair unchanged.  Language is a binary
    eligibility obligation and never ranks two eligible boundary states.
    """
    if len(states) < 2 or not chunks:
        return None, {"reason": "no_filtration_or_chunks", "feasible": []}
    closure = states[-1]["interval"]
    closure_persistence = states[-1]["persistence"]
    feasible, margins = [], []
    for state in states[:-1]:
        if require_stronger_than_closure and state["persistence"] <= closure_persistence:
            continue
        passed, audit = stable_warrant(chunks, state["interval"], closure, margins)
        if passed:
            feasible.append({**state, "interval": list(state["interval"]), "audit": audit})
    if not feasible:
        return None, {"reason": "abstain", "feasible": [],
                      "mean_margin": sum(margins) / len(margins) if margins else None}
    # No learned score and no cross-modal sum: visual persistence decides;
    # the larger extent is the deterministic conservative tie-break.
    chosen = max(feasible, key=lambda row: (
        row["persistence"], row["interval"][1] - row["interval"][0]))
    return tuple(chosen["interval"]), {"reason": "persistent_warrant",
                                      "feasible": feasible, "chosen": chosen,
                                      "mean_margin": sum(margins) / len(margins) if margins else None}


def text_only(chunks: list[dict], duration: float) -> tuple[float, float] | None:
    positive = [row for row in chunks if row["z"] > NULL_LOG_ODDS]
    if not positive:
        return None
    return max(0.0, min(row["start"] for row in positive)), min(duration, max(row["end"] for row in positive))


def emit(method: str, key: tuple[str, str], base: dict, interval: tuple[float, float] | None,
         evidence: dict) -> dict:
    duration = float(base["duration"])
    base_nonempty = bool(base.get("intervals"))
    score = max((float(row[2]) for row in base.get("intervals", []) if len(row) > 2), default=1.0)
    intervals = [Interval(interval[0], interval[1], score)] if base_nonempty and interval else []
    return Prediction(method, key[0], key[1], duration,
                      score_curve=intervals_to_curve(intervals, duration), intervals=intervals,
                      calls=0, modality_evidence=evidence,
                      raw={"gt_access": False, "proxy": "frozen_per_chunk_logit_max",
                           "null_log_odds": NULL_LOG_ODDS}).to_dict()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    proposals, base, chunks = load_map(args.proposals), load_map(args.base), load_chunks()
    keys = sorted(set(proposals) & set(base))
    counts = defaultdict(int)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        for key in keys:
            states = filtration_states(proposals[key]["proposals"])
            hulls = [row["interval"] for row in states]
            closure = hulls[-1]
            native = chunks.get(key, [])
            aligned, audit = select(hulls, native, stable=True)
            persistent, persistent_audit = select_persistent(states, native)
            strict_persistent, strict_persistent_audit = select_persistent(
                states, native, require_stronger_than_closure=True)
            persistent_shuffle, persistent_shuffle_audit = select_persistent(
                states, shuffled_chunks(key, native))
            no_stability, no_stability_audit = select(hulls, native, stable=False)
            shuffled, shuffled_audit = select(hulls, shuffled_chunks(key, native), stable=True)
            visual = hulls[0] if len(hulls) > 1 else closure
            text = text_only(native, float(base[key]["duration"]))
            variants = {
                "pact_proxy_aligned": (aligned or closure, {**audit, "edited": aligned is not None}),
                "pact_proxy_persistent": (persistent or closure,
                                           {**persistent_audit, "edited": persistent is not None}),
                "pact_proxy_strict_persistent": (
                    strict_persistent or closure,
                    {**strict_persistent_audit, "edited": strict_persistent is not None}),
                "pact_proxy_persistent_shuffle": (
                    persistent_shuffle or closure,
                    {**persistent_shuffle_audit, "edited": persistent_shuffle is not None}),
                "pact_proxy_shuffle": (shuffled or closure, {**shuffled_audit, "edited": shuffled is not None}),
                "pact_proxy_no_stability": (no_stability or closure,
                                             {**no_stability_audit, "edited": no_stability is not None}),
                "pact_proxy_visual_only": (visual, {"edited": visual != closure}),
                "pact_proxy_text_only": (text, {"edited": text is not None}),
                "pact_proxy_closure": (closure, {"edited": False}),
            }
            for method, (interval, evidence) in variants.items():
                row = emit(method, key, base[key], interval, evidence)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[method + "_edited"] += int(evidence["edited"])
    print(json.dumps({"videos": len(keys), "rows": len(keys) * 9, "counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
