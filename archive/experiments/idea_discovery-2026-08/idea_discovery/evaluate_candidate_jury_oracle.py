#!/usr/bin/env python3
"""GT-isolated ceiling for selecting one frozen method output per video."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.label_free_adapt.evaluate import binary_intervals, interval_f1, temporal_iou


def load_rows(path: Path, cohort: set[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["dataset"], row["video_id"])
        if key in cohort:
            rows[key] = row
    return rows


def best_iou(row: dict, targets: list[tuple[float, float]]) -> float:
    return max(
        (temporal_iou(tuple(interval[:2]), target)
         for interval in row.get("intervals", []) for target in targets),
        default=0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "JSONL"), required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = [json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    cohort = {(r["dataset"], r["video_id"]) for r in manifest}
    methods = {name: load_rows(Path(path), cohort) for name, path in args.candidate}
    if args.baseline not in methods:
        raise ValueError(f"baseline {args.baseline!r} is not a candidate")
    for name, rows in methods.items():
        missing = cohort - set(rows)
        if missing:
            raise RuntimeError(f"{name} missing {len(missing)} cohort rows; examples={sorted(missing)[:3]}")

    gt = {}
    for dataset in sorted({d for d, _ in cohort}):
        z = np.load(args.gt_dir / f"{dataset}.npz", allow_pickle=True)
        for i, video_id in enumerate(z["video_ids"]):
            if "split" in z and str(z["split"][i]) != "test":
                continue
            gt[(dataset, str(video_id))] = np.asarray(z["y4"][i], dtype=np.int8)

    baseline_rows, oracle_rows, y_subset, detail = {}, {}, {}, []
    for key in sorted(cohort):
        dataset, video_id = key
        y = gt[key]
        targets = binary_intervals(y)
        scored = [(best_iou(rowset[key], targets), name) for name, rowset in methods.items()]
        # Stable tie-breaking deliberately favors the declared baseline.
        scored.sort(key=lambda item: (item[0], item[1] == args.baseline), reverse=True)
        quality, winner = scored[0]
        if args.allow_empty and not targets:
            winner = "EMPTY"
        baseline_rows[key] = {"intervals": methods[args.baseline][key].get("intervals", [])}
        oracle_rows[key] = {"intervals": [] if winner == "EMPTY" else methods[winner][key].get("intervals", [])}
        y_subset[key] = y
        detail.append({"dataset": dataset, "video_id": video_id, "winner": winner,
                       "winner_iou": quality,
                       "candidate_iou": {name: value for value, name in scored}})

    per_dataset = {}
    for dataset in sorted({d for d, _ in cohort}):
        keys = [key for key in cohort if key[0] == dataset]
        y_d = {key: y_subset[key] for key in keys}
        base_d = {key: baseline_rows[key] for key in keys}
        oracle_d = {key: oracle_rows[key] for key in keys}
        per_dataset[dataset] = {
            "n": len(keys), "baseline": interval_f1(y_d, base_d),
            "oracle": interval_f1(y_d, oracle_d),
            "winner_counts": dict(Counter(x["winner"] for x in detail if x["dataset"] == dataset)),
        }
    macro = {}
    for metric in ("interval_F1@0.3", "interval_F1@0.5", "interval_F1@0.7"):
        base = float(np.mean([x["baseline"][metric] for x in per_dataset.values()]))
        oracle = float(np.mean([x["oracle"][metric] for x in per_dataset.values()]))
        macro[metric] = {"baseline": base, "oracle": oracle, "delta": oracle - base}
    output = {"cohort_size": len(cohort), "baseline": args.baseline,
              "allow_empty": args.allow_empty, "per_dataset": per_dataset,
              "macro": macro, "detail": detail}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cohort_size": len(cohort), "macro": macro,
                      "winners": dict(Counter(x["winner"] for x in detail))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
