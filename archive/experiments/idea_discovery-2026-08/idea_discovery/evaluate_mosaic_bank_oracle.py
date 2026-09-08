#!/usr/bin/env python3
"""Audit a fair K<=3 MOSAIC bank against an equal-cardinality A10 bank.

Ground truth is used only in this isolated ceiling evaluator.  Every selectable
hypothesis is exactly one temporal interval; multi-interval method outputs are
never treated as a single oracle choice.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.label_free_adapt.evaluate import binary_intervals, interval_f1, temporal_iou


def read_predictions(path: Path) -> dict[tuple[str, str], dict]:
    output = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("error") is None:
                output[(row["dataset"], row["video_id"])] = row
    return output


def read_proposals(path: Path) -> dict[tuple[str, str], dict]:
    output = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("error") is None:
                output[(row["dataset"], row["video_id"])] = row
    return output


def first_interval(row: dict) -> list[float] | None:
    """Frozen representative: earliest normalized interval in source output."""
    intervals = sorted(row.get("intervals", []), key=lambda x: (float(x[0]), float(x[1])))
    if not intervals:
        return None
    value = intervals[0]
    return [float(value[0]), float(value[1]), float(value[2]) if len(value) > 2 else 1.0]


def a10_intervals(row: dict, k: int = 3) -> list[tuple[str, list[float]]]:
    answer = []
    for proposal in row.get("proposals", [])[:k]:
        answer.append((f"A10_r{int(proposal['rank'])}",
                       [float(proposal["start"]), float(proposal["end"]),
                        float(proposal["logit"])]))
    return answer


def quality(interval: list[float], targets: list[tuple[float, float]]) -> float:
    return max((temporal_iou(tuple(interval[:2]), target) for target in targets), default=0.0)


def choose(candidates: list[tuple[str, list[float]]], targets: list[tuple[float, float]],
           fallback: str) -> tuple[str, list[float], float]:
    scored = [(quality(interval, targets), name == fallback, -index, name, interval)
              for index, (name, interval) in enumerate(candidates)]
    value, _, _, name, interval = max(scored)
    return name, interval, value


def evaluate_bank(cohort, y, bank, fallback):
    selected, detail = {}, []
    for key in sorted(cohort):
        targets = binary_intervals(y[key])
        winner, interval, best = choose(bank[key], targets, fallback)
        selected[key] = {"intervals": [interval]}
        detail.append({"dataset": key[0], "video_id": key[1], "winner": winner,
                       "winner_iou": best, "k": len(bank[key]),
                       "candidate_iou": {name: quality(value, targets)
                                         for name, value in bank[key]}})
    return selected, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a10", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--a10-proposals", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sources = {"A08": read_predictions(args.a08), "A10": read_predictions(args.a10),
               "A12": read_predictions(args.a12)}
    proposals = read_proposals(args.a10_proposals)
    cohort = set.intersection(*(set(rows) for rows in sources.values()), set(proposals))
    if not cohort:
        raise RuntimeError("empty common cohort")

    y = {}
    for dataset in sorted({key[0] for key in cohort}):
        z = np.load(args.gt_dir / f"{dataset}.npz", allow_pickle=True)
        for i, video_id in enumerate(z["video_ids"]):
            key = (dataset, str(video_id))
            if key in cohort and ("split" not in z or str(z["split"][i]) == "test"):
                y[key] = np.asarray(z["y4"][i], dtype=np.int8)
    cohort &= set(y)

    heterogeneous, a10_only, a10_top1 = {}, {}, {}
    for key in cohort:
        values = []
        for name in ("A08", "A10", "A12"):
            interval = first_interval(sources[name][key])
            if interval is not None:
                values.append((name, interval))
        # Exact endpoint duplicates are one hypothesis. Preserve A10 when tied.
        unique = {}
        for name, interval in values:
            endpoint = (round(interval[0] * 4) / 4, round(interval[1] * 4) / 4)
            if endpoint not in unique or name == "A10":
                unique[endpoint] = (name, [endpoint[0], endpoint[1], interval[2]])
        heterogeneous[key] = list(unique.values())
        a10_only[key] = a10_intervals(proposals[key], 3)
        top = first_interval(sources["A10"][key])
        if top is None:
            raise RuntimeError(f"A10 unexpectedly empty at {key}")
        a10_top1[key] = [("A10", top)]

    arms = {}
    details = {}
    a10_banks = {f"A10_only_K{k}": {key: a10_intervals(proposals[key], k)
                                     for key in cohort}
                 for k in (2, 3, 4, 8)
                 if min(len(proposals[key].get("proposals", [])) for key in cohort) >= k}
    evaluated = [("A10", a10_top1, "A10"),
                 ("heterogeneous_K3", heterogeneous, "A10")]
    evaluated.extend((name, bank, "A10_r1") for name, bank in a10_banks.items())
    for name, bank, fallback in evaluated:
        selected, detail = evaluate_bank(cohort, y, bank, fallback)
        details[name] = detail
        per_dataset = {}
        for dataset in sorted({key[0] for key in cohort}):
            keys = [key for key in cohort if key[0] == dataset]
            metrics = interval_f1({key: y[key] for key in keys},
                                  {key: selected[key] for key in keys})
            per_dataset[dataset] = {"n": len(keys), **metrics,
                                    "winner_counts": dict(Counter(
                                        item["winner"] for item in detail
                                        if item["dataset"] == dataset))}
        macro = {metric: float(np.mean([row[metric] for row in per_dataset.values()]))
                 for metric in ("interval_F1@0.3", "interval_F1@0.5", "interval_F1@0.7")}
        arms[name] = {"per_dataset": per_dataset, "macro": macro,
                      "mean_k": float(np.mean([len(bank[key]) for key in cohort])),
                      "k_counts": dict(Counter(len(bank[key]) for key in cohort))}

    result = {"cohort_size": len(cohort), "cohort_by_dataset": dict(Counter(k[0] for k in cohort)),
              "representative_policy": "earliest normalized single interval per source; exact 4fps duplicates merged",
              "arms": arms,
              "delta_vs_A10": {arm: {metric: values["macro"][metric] - arms["A10"]["macro"][metric]
                                      for metric in values["macro"]}
                                 for arm, values in arms.items() if arm != "A10"},
              "detail": details}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "detail"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
