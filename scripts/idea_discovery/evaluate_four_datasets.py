#!/usr/bin/env python3
"""Evaluate shared-schema predictions on all four localization datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.label_free_adapt.evaluate import interval_f1, pooled, within_video_macro

DATASETS = ("HateMM", "HateClipSeg", "MHC", "MHC_zh")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    latest = {}
    with args.predictions.open() as handle:
        for line in handle:
            row = json.loads(line)
            latest[(row["method"], row["dataset"], row["video_id"])] = row
    rows = []
    for dataset in DATASETS:
        gt = np.load(args.gt_dir / f"{dataset}.npz", allow_pickle=True)
        y = {str(video_id): np.asarray(gt["y4"][i], dtype=np.int8)
             for i, video_id in enumerate(gt["video_ids"])
             if str(gt["split"][i]) == args.split}
        methods = sorted({key[0] for key in latest if key[1] == dataset})
        for method in methods:
            predictions = {video_id: row for (m, d, video_id), row in latest.items()
                           if m == method and d == dataset and not row.get("error")
                           and row.get("score_curve")}
            scores = {video_id: np.asarray(row["score_curve"], float)
                      for video_id, row in predictions.items()}
            rows.append({"method": method, "dataset": dataset,
                         **pooled(y, scores), **within_video_macro(y, scores),
                         **interval_f1(y, predictions),
                         "n_videos_predicted": len(predictions),
                         "n_videos_overlap": len(set(y) & set(predictions)),
                         "n_nonempty_intervals": sum(bool(x.get("intervals"))
                                                     for x in predictions.values())})
    macros = []
    metric_names = ("frame_ROC_AUC", "frame_PR_AUC", "within_video_macro_ROC_AUC",
                    "interval_F1@0.3", "interval_F1@0.5", "interval_F1@0.7")
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        macro = {"method": method, "dataset": "MACRO", "n_datasets": len(group)}
        for metric in metric_names:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            macro[metric] = float(np.mean(values)) if values else None
        macros.append(macro)
    output = {"per_dataset": rows, "macro": macros,
              "prediction_file": str(args.predictions.resolve())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(macros, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
