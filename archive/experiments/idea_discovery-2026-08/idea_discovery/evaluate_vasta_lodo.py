#!/usr/bin/env python3
"""Leave-one-dataset-out selection audit for VASTA's transcript-veto threshold.

This is a retrospective cross-dataset robustness analysis, not an untouched test.
For each held dataset, select a threshold using only the other datasets and then
report the already-computed metrics on the held dataset.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


PATTERN = re.compile(r"^extent_gate_consensus_text_ge(-?\d+)$")
TARGET = "interval_F1@0.5"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.metrics.read_text())
    rows = [row for row in payload["per_dataset"] if PATTERN.match(row["method"])]
    datasets = sorted({row["dataset"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    by_key = {(row["dataset"], row["method"]): row for row in rows}

    folds = []
    held_rows = []
    for held in datasets:
        train = [dataset for dataset in datasets if dataset != held]
        candidates = []
        for method in methods:
            train_score = float(np.mean([by_key[(dataset, method)][TARGET]
                                         for dataset in train]))
            threshold = int(PATTERN.match(method).group(1))
            candidates.append({
                "method": method,
                "threshold": threshold,
                "train_macro_F1@0.5": train_score,
            })
        # The tie break is declared without held-dataset access: prefer the
        # numerically smaller threshold, i.e. the more conservative/sparser veto.
        candidates.sort(key=lambda item: (-item["train_macro_F1@0.5"],
                                          item["threshold"]))
        selected = candidates[0]
        test = by_key[(held, selected["method"])]
        held_rows.append(test)
        folds.append({
            "held_dataset": held,
            "train_datasets": train,
            "selected_method": selected["method"],
            "selected_threshold": selected["threshold"],
            "train_macro_F1@0.5": selected["train_macro_F1@0.5"],
            "held_metrics": {key: test[key] for key in (
                "frame_PR_AUC", "frame_ROC_AUC", "within_video_macro_ROC_AUC",
                "interval_F1@0.3", "interval_F1@0.5", "interval_F1@0.7")},
            "selection_ranking": candidates,
        })

    metric_names = (
        "frame_PR_AUC", "frame_ROC_AUC", "within_video_macro_ROC_AUC",
        "interval_F1@0.3", "interval_F1@0.5", "interval_F1@0.7")
    output = {
        "protocol": "retrospective leave-one-dataset-out threshold selection",
        "integrity_note": (
            "The rule family was developed on these datasets; this analysis tests "
            "cross-dataset threshold stability and is not an untouched test."),
        "selection_metric": TARGET,
        "all_folds_select_same_threshold": len({fold["selected_threshold"]
                                                 for fold in folds}) == 1,
        "folds": folds,
        "held_dataset_macro": {
            metric: float(np.mean([row[metric] for row in held_rows]))
            for metric in metric_names},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "selected_thresholds": {fold["held_dataset"]: fold["selected_threshold"]
                                for fold in folds},
        "held_dataset_macro": output["held_dataset_macro"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
