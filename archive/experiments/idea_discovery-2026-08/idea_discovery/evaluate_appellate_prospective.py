#!/usr/bin/env python3
"""Evaluate frozen Appellate-PACT predictions after prospective GT release."""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.duplex.frame_eval_common import build_gt_array
from scripts.label_free_adapt.evaluate import (
    interval_f1, pooled, temporal_iou, within_video_macro,
)

POSITIVE = {"Hateful", "Offensive"}


def upstream(path: Path) -> dict[str, dict]:
    output = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            spans = []
            for raw in ast.literal_eval(row["Duration"] or "[]"):
                start, end = map(float, raw)
                if end > start:
                    spans.append((start, end))
            output[row["Video_ID"].strip()] = {
                "label": row["Majority_Voting"].strip(), "spans": spans}
    return output


def load_predictions(paths: list[Path]) -> dict[tuple[str, str, str], dict]:
    output = {}
    for path in paths:
        for row in map(json.loads, path.open()):
            output[(row["method"], row["dataset"], row["video_id"])] = row
    return output


def intervals(row: dict) -> list[tuple[float, float]]:
    return [(float(value[0]), float(value[1])) for value in row.get("intervals", [])]


def best_tiou(row: dict, truth: list[tuple[float, float]]) -> float:
    predicted = intervals(row)
    return max((temporal_iou(a, b) for a in predicted for b in truth), default=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--en-gt", type=Path, required=True)
    ap.add_argument("--zh-gt", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, action="append", required=True)
    ap.add_argument("--base-method", default="extent_closure_top8")
    ap.add_argument("--target-method", default="appellate_pact_tristate")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    gold_sources = {
        "MHC_en_train_prospective": upstream(args.en_gt),
        "MHC_zh_train_prospective": upstream(args.zh_gt),
    }
    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines()
                if line.strip()]
    y_by_dataset: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    spans_by_dataset: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(dict)
    exclusions = []
    for row in manifest:
        dataset, video_id = row["dataset"], row["video_id"]
        record = gold_sources[dataset].get(video_id)
        if record is None:
            exclusions.append({"dataset": dataset, "video_id": video_id,
                               "reason": "absent_from_upstream_gt"})
            continue
        positive = record["label"] in POSITIVE
        if positive and not record["spans"]:
            exclusions.append({"dataset": dataset, "video_id": video_id,
                               "reason": "positive_video_without_usable_span"})
            continue
        used = record["spans"] if positive else []
        y_by_dataset[dataset][video_id] = build_gt_array(
            used, float(row["duration"]), fps=4.0)
        spans_by_dataset[dataset][video_id] = used

    predictions = load_predictions(args.predictions)
    methods = sorted({key[0] for key in predictions})
    results = []
    for dataset, y in y_by_dataset.items():
        for method in methods:
            rows = {video_id: row for (name, source, video_id), row in predictions.items()
                    if name == method and source == dataset and video_id in y}
            if not rows:
                continue
            if set(rows) != set(y):
                raise RuntimeError(f"{dataset}/{method}: prediction/GT mismatch: "
                                   f"{len(rows)} vs {len(y)}")
            scores = {video_id: np.asarray(row["score_curve"], float)
                      for video_id, row in rows.items()}
            results.append({"dataset": dataset, "method": method,
                            **pooled(y, scores), **within_video_macro(y, scores),
                            **interval_f1(y, rows), "n_videos": len(rows),
                            "n_nonempty": sum(bool(row.get("intervals"))
                                              for row in rows.values())})

    edits = []
    for dataset, y in y_by_dataset.items():
        base = {video_id: row for (method, source, video_id), row in predictions.items()
                if method == args.base_method and source == dataset and video_id in y}
        target = {video_id: row for (method, source, video_id), row in predictions.items()
                  if method == args.target_method and source == dataset and video_id in y}
        for video_id in sorted(set(base) & set(target)):
            if intervals(base[video_id]) == intervals(target[video_id]):
                continue
            truth = spans_by_dataset[dataset][video_id]
            before, after = best_tiou(base[video_id], truth), best_tiou(target[video_id], truth)
            edits.append({"dataset": dataset, "video_id": video_id,
                          "before_best_tiou": before, "after_best_tiou": after,
                          "gain": after - before,
                          "verdict": "correct" if after > before else
                                     "wrong" if after < before else "tie"})
    edit_summary = {}
    for dataset in list(y_by_dataset) + ["ALL"]:
        group = edits if dataset == "ALL" else [row for row in edits if row["dataset"] == dataset]
        edit_summary[dataset] = {
            "n": len(group),
            "correct": sum(row["verdict"] == "correct" for row in group),
            "wrong": sum(row["verdict"] == "wrong" for row in group),
            "tie": sum(row["verdict"] == "tie" for row in group),
            "mean_gain": float(np.mean([row["gain"] for row in group])) if group else None,
        }
    output = {"results": results, "edit_summary": edit_summary,
              "edits": edits, "exclusions": exclusions,
              "gt_access": True, "fps": 4.0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, ensure_ascii=False,
                                   sort_keys=True) + "\n")
    print(json.dumps({"results": results, "edit_summary": edit_summary,
                      "n_exclusions": len(exclusions)}, indent=2,
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
