#!/usr/bin/env python3
"""PACT-v3: persistent visual filtration with native semantic shell warrants."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_pact_filtration_proxy import (
    CHUNK_SOURCES, NULL_LOG_ODDS, load_map,
)
from scripts.label_free_adapt.schema import Interval, Prediction, intervals_to_curve

FPS = 4.0


def load_chunks(unified: Path | None = None) -> tuple[dict[tuple[str, str], list[dict]], float]:
    output: dict[tuple[str, str], list[dict]] = defaultdict(list)
    if unified is not None:
        anchors = set()
        for source in map(json.loads, unified.open()):
            anchors.add(float(source["config"]["null_log_odds"]))
            output[(str(source["dataset"]), str(source["video_id"]))].append({
                "start": float(source["start"]), "end": float(source["end"]),
                "z": float(source["log_odds"])})
        if len(anchors) != 1:
            raise RuntimeError(f"expected one unified null anchor, got {anchors}")
        for rows in output.values():
            rows.sort(key=lambda row: (row["start"], row["end"]))
        return dict(output), anchors.pop()
    for dataset, (path, score_field) in CHUNK_SOURCES.items():
        for source in map(json.loads, path.open()):
            span, value = source.get("span"), source.get(score_field)
            if span and len(span) == 2 and value is not None:
                output[(dataset, str(source["video_id"]))].append({
                    "start": float(span[0]), "end": float(span[1]), "z": float(value)})
    for rows in output.values():
        rows.sort(key=lambda row: (row["start"], row["end"]))
    return dict(output), NULL_LOG_ODDS


def component(proposals: list[dict], duration: float) -> tuple[int, int]:
    n = max(1, int(math.floor(duration * FPS)))
    centers = (np.arange(n) + 0.5) / FPS
    active = np.zeros(n, dtype=bool)
    for proposal in proposals:
        active |= ((centers >= float(proposal["start"])) &
                   (centers < float(proposal["end"])))
    bounds = np.flatnonzero(np.diff(np.r_[False, active, False])).reshape(-1, 2)
    anchor = 0.5 * (float(proposals[0]["start"]) + float(proposals[0]["end"]))
    containing = [pair for pair in bounds if pair[0] / FPS <= anchor < pair[1] / FPS]
    chosen = max(containing if containing else bounds, key=lambda pair: pair[1] - pair[0])
    return int(chosen[0]), int(chosen[1])


def states(proposals: list[dict], duration: float) -> list[dict]:
    output = []
    for rank in range(1, len(proposals) + 1):
        pair = component(proposals[:rank], duration)
        if output and output[-1]["cells"] == pair:
            output[-1]["persistence"] += 1
            output[-1]["last_rank"] = rank
        else:
            output.append({"cells": pair, "persistence": 1,
                           "first_rank": rank, "last_rank": rank})
    return output


def qfield(chunks: list[dict], duration: float, null_log_odds: float,
           offset: int = 0, circular: int = 0) -> np.ndarray:
    n = max(1, int(math.floor(duration * FPS)))
    total, count = np.zeros(n), np.zeros(n)
    if not chunks:
        return total
    scores = [row["z"] for row in chunks]
    if circular:
        circular %= len(scores)
        scores = scores[circular:] + scores[:circular]
    # A +/- one-chunk timestamp perturbation assigns each chunk's evidence to
    # the adjacent native timestamp span, with edge replication.
    for index, score in enumerate(scores):
        target = min(max(index + offset, 0), len(chunks) - 1)
        row = chunks[target]
        lo = max(0, int(math.floor(row["start"] * FPS)))
        hi = min(n, int(math.ceil(row["end"] * FPS)))
        if hi > lo:
            total[lo:hi] += score - null_log_odds
            count[lo:hi] += 1
    field = np.zeros(n)
    covered = count > 0
    field[covered] = total[covered] / count[covered]
    return field


def density(field: np.ndarray, mask: np.ndarray) -> float | None:
    return float(field[mask].mean()) if mask.any() else None


def maximum(field: np.ndarray, mask: np.ndarray) -> float | None:
    return float(field[mask].max()) if mask.any() else None


def warrant(candidate_index: int, visual_states: list[dict], fields: list[np.ndarray],
            whole_shell: bool = False, max_witness: bool = False) -> tuple[bool, list[dict]]:
    audit = []
    for offset, field in zip((-1, 0, 1), fields):
        core = np.zeros(len(field), dtype=bool)
        a, b = visual_states[candidate_index]["cells"]
        core[a:b] = True
        reducer = maximum if max_witness else density
        core_density = reducer(field, core)
        shell_densities = []
        if whole_shell:
            shell = np.zeros(len(field), dtype=bool)
            x, y = visual_states[-1]["cells"]
            shell[x:y] = True
            shell &= ~core
            value = reducer(field, shell)
            if value is not None:
                shell_densities.append(value)
        else:
            for later in range(candidate_index, len(visual_states) - 1):
                inner, outer = visual_states[later]["cells"], visual_states[later + 1]["cells"]
                annulus = np.zeros(len(field), dtype=bool)
                annulus[outer[0]:outer[1]] = True
                annulus[inner[0]:inner[1]] = False
                value = reducer(field, annulus)
                if value is not None:
                    shell_densities.append(value)
        passed = (core_density is not None and bool(shell_densities) and
                  core_density > 0 and core_density > max(shell_densities))
        audit.append({"offset": offset, "core_density": core_density,
                      "shell_densities": shell_densities, "passed": passed})
    return all(row["passed"] for row in audit), audit


def choose(visual_states: list[dict], fields: list[np.ndarray],
           require_text: bool = True, require_persistence: bool = True,
           whole_shell: bool = False, max_witness: bool = False) -> tuple[int | None, dict]:
    closure_persistence = visual_states[-1]["persistence"]
    feasible = []
    for index, state in enumerate(visual_states[:-1]):
        if require_persistence and state["persistence"] <= closure_persistence:
            continue
        passed, audit = warrant(index, visual_states, fields, whole_shell=whole_shell,
                                max_witness=max_witness)
        if not require_text:
            passed = True
        if passed:
            feasible.append({"index": index, "persistence": state["persistence"],
                             "cells": list(state["cells"]), "audit": audit})
    if not feasible:
        return None, {"feasible": [], "reason": "abstain"}
    best_value = max((row["persistence"], row["cells"][1] - row["cells"][0])
                     for row in feasible)
    winners = [row for row in feasible
               if (row["persistence"], row["cells"][1] - row["cells"][0]) == best_value]
    if len(winners) != 1:
        return None, {"feasible": feasible, "reason": "endpoint_tie"}
    return winners[0]["index"], {"feasible": feasible, "chosen": winners[0],
                                  "reason": "warranted_edit"}


def prediction(method: str, key: tuple[str, str], base: dict, selected: int | None,
               visual_states: list[dict], evidence: dict) -> dict:
    # Exact fallback is important: preserve VASTA's score curve and interval.
    if selected is None or not base.get("intervals"):
        intervals = [Interval(float(row[0]), float(row[1]), float(row[2]))
                     for row in base.get("intervals", [])]
        curve = [float(value) for value in base["score_curve"]]
        effective = False
    else:
        a, b = visual_states[selected]["cells"]
        score = max((float(row[2]) for row in base["intervals"]), default=1.0)
        intervals = [Interval(a / FPS, b / FPS, score)]
        curve = intervals_to_curve(intervals, float(base["duration"]))
        effective = True
    evidence = {**evidence, "eligible_edit": selected is not None,
                "effective_edit": effective, "fallback_bit_exact": selected is None}
    return Prediction(method, key[0], key[1], float(base["duration"]),
                      score_curve=curve, intervals=intervals, calls=0,
                      modality_evidence=evidence,
                      raw={"gt_access": False, "version": "pact_v3_max_annulus",
                           "null_log_odds": evidence.get("null_log_odds", NULL_LOG_ODDS)}).to_dict()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--chunk-scores", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    proposal_map, base_map = load_map(args.proposals), load_map(args.base)
    chunks, null_log_odds = load_chunks(args.chunk_scores)
    keys = sorted(set(proposal_map) & set(base_map))
    counts = Counter()
    row_count = 0
    with args.out.open("x", encoding="utf-8") as handle:
        for key in keys:
            base, proposals = base_map[key], proposal_map[key]["proposals"]
            duration = float(base["duration"])
            visual_states = states(proposals, duration)
            native = chunks.get(key, [])
            aligned_fields = [qfield(native, duration, null_log_odds, offset)
                              for offset in (-1, 0, 1)]
            digest = hashlib.sha256((key[0] + "\0" + key[1]).encode()).digest()
            rotation = 0 if len(native) < 2 else 1 + int.from_bytes(digest[:4], "big") % (len(native) - 1)
            shuffled_fields = [qfield(native, duration, null_log_odds, offset, rotation)
                               for offset in (-1, 0, 1)]
            variants = {
                "pact_v3": choose(visual_states, aligned_fields),
                "pact_v3_timestamp_shuffle": choose(visual_states, shuffled_fields),
                "pact_v3_max_shell": choose(visual_states, aligned_fields,
                                             whole_shell=True, max_witness=True),
                "pact_v3_max_shell_shuffle": choose(visual_states, shuffled_fields,
                                                     whole_shell=True, max_witness=True),
                "pact_v3_visual_only": choose(visual_states, aligned_fields, require_text=False),
                "pact_v3_text_only": choose(visual_states, aligned_fields, require_persistence=False),
                "pact_v3_whole_shell": choose(visual_states, aligned_fields, whole_shell=True),
            }
            # Exploratory v4: a boundary edit must be supported by the native
            # alignment but not by the within-video circular timestamp
            # counterfactual.  Visual persistence may equal (but not be weaker
            # than) the fallback closure.  This converts the shuffle from a
            # post-hoc control into a claim-specific authorization test.
            loose_aligned, loose_aligned_evidence = choose(
                visual_states, aligned_fields, require_persistence=False)
            loose_shuffle, loose_shuffle_evidence = choose(
                visual_states, shuffled_fields, require_persistence=False)
            contrast_selected = loose_aligned
            contrast_reason = "native_only_warrant"
            if contrast_selected is not None:
                candidate_persistence = visual_states[contrast_selected]["persistence"]
                closure_persistence = visual_states[-1]["persistence"]
                if candidate_persistence < closure_persistence:
                    contrast_selected, contrast_reason = None, "weaker_than_closure"
                elif loose_shuffle == contrast_selected:
                    contrast_selected, contrast_reason = None, "survives_circular_shift"
            else:
                contrast_reason = "no_native_warrant"
            variants["pact_v4_alignment_contrast"] = (
                contrast_selected,
                {"reason": contrast_reason,
                 "native_selection": loose_aligned,
                 "circular_selection": loose_shuffle,
                 "native_evidence": loose_aligned_evidence,
                 "circular_evidence": loose_shuffle_evidence})
            # Confirmatory appellate variant: the native boundary motion must
            # be stable to +/- one chunk, yet disappear under *every*
            # non-identity circular reassignment of content to timestamps.
            # This is stricter than the legacy single hash-selected rotation.
            orbit_selections = []
            orbit_evidence = []
            for orbit_rotation in range(1, len(native)):
                orbit_fields = [qfield(native, duration, null_log_odds, offset,
                                       orbit_rotation)
                                for offset in (-1, 0, 1)]
                orbit_selected, orbit_audit = choose(
                    visual_states, orbit_fields, require_persistence=False)
                orbit_selections.append(orbit_selected)
                orbit_evidence.append({"rotation": orbit_rotation,
                                       "selection": orbit_selected,
                                       "evidence": orbit_audit})
            appellate_selected = loose_aligned
            appellate_reason = "native_exclusive_full_orbit"
            if appellate_selected is None:
                appellate_reason = "no_native_warrant"
            elif visual_states[appellate_selected]["persistence"] < visual_states[-1]["persistence"]:
                appellate_selected, appellate_reason = None, "weaker_than_closure"
            elif any(value == appellate_selected for value in orbit_selections):
                appellate_selected, appellate_reason = None, "survives_nonidentity_orbit"
            variants["pact_v5_full_orbit_appeal"] = (
                appellate_selected,
                {"reason": appellate_reason,
                 "native_selection": loose_aligned,
                 "native_evidence": loose_aligned_evidence,
                 "full_orbit_selections": orbit_selections,
                 "full_orbit_evidence": orbit_evidence})
            max_native, max_native_evidence = choose(
                visual_states, aligned_fields, require_persistence=False,
                whole_shell=True, max_witness=True)
            max_circular, max_circular_evidence = choose(
                visual_states, shuffled_fields, require_persistence=False,
                whole_shell=True, max_witness=True)
            max_contrast = max_native
            max_reason = "native_only_warrant"
            if max_contrast is not None:
                if visual_states[max_contrast]["persistence"] < visual_states[-1]["persistence"]:
                    max_contrast, max_reason = None, "weaker_than_closure"
                elif max_circular == max_contrast:
                    max_contrast, max_reason = None, "survives_circular_shift"
            else:
                max_reason = "no_native_warrant"
            variants["pact_v4_max_shell_alignment_contrast"] = (
                max_contrast,
                {"reason": max_reason, "native_selection": max_native,
                 "circular_selection": max_circular,
                 "native_evidence": max_native_evidence,
                 "circular_evidence": max_circular_evidence})
            for method, (selected, evidence) in variants.items():
                row = prediction(method, key, base, selected, visual_states,
                                 {**evidence, "n_states": len(visual_states),
                                  "closure_persistence": visual_states[-1]["persistence"],
                                  "n_chunks": len(native), "shuffle_rotation": rotation,
                                  "null_log_odds": null_log_odds})
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                row_count += 1
                counts[method + ":eligible"] += int(selected is not None)
                counts[method + ":effective"] += int(row["modality_evidence"]["effective_edit"])
                if selected is not None:
                    counts[method + ":dataset:" + key[0]] += 1
    print(json.dumps({"videos": len(keys), "rows": row_count,
                      "counts": dict(sorted(counts.items()))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
