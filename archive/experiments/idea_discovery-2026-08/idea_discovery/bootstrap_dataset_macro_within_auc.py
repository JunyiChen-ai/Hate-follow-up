#!/usr/bin/env python3
"""Dataset-stratified paired bootstrap for macro within-video ROC-AUC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


DATASETS = ("HateMM", "HateClipSeg", "MHC", "MHC_zh")


def load(path: Path, method: str) -> dict[tuple[str, str], dict]:
    rows = {}
    for row in map(json.loads, path.open()):
        if row.get("method") == method and not row.get("error") and row.get("score_curve"):
            key = (row["dataset"], row["video_id"])
            if key in rows:
                raise ValueError(f"duplicate prediction key: {key}")
            rows[key] = row
    if not rows:
        raise ValueError(f"no rows for {method!r} in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--candidate-method", required=True)
    parser.add_argument("--baseline-method", required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidate = load(args.candidate_predictions, args.candidate_method)
    baseline = load(args.baseline_predictions, args.baseline_method)
    differences: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for dataset in DATASETS:
        gt = np.load(args.gt_dir / f"{dataset}.npz", allow_pickle=True)
        labels = {
            str(video_id): np.asarray(gt["y4"][index], dtype=np.int8)
            for index, video_id in enumerate(gt["video_ids"])
            if str(gt["split"][index]) == "test"
        }
        values = []
        ids = sorted(
            set(labels)
            & {video_id for ds, video_id in candidate if ds == dataset}
            & {video_id for ds, video_id in baseline if ds == dataset}
        )
        for video_id in ids:
            y = labels[video_id]
            a = np.asarray(candidate[(dataset, video_id)]["score_curve"], dtype=float)
            b = np.asarray(baseline[(dataset, video_id)]["score_curve"], dtype=float)
            if len(a) not in (len(y), len(y) + 1) or len(b) not in (len(y), len(y) + 1):
                raise ValueError(
                    f"unexpected alignment for {(dataset, video_id)}: "
                    f"gt={len(y)} candidate={len(a)} baseline={len(b)}"
                )
            a = a[: len(y)]
            b = b[: len(y)]
            if y.min() != y.max():
                values.append(roc_auc_score(y, a) - roc_auc_score(y, b))
        if not values:
            raise ValueError(f"no non-degenerate common videos for {dataset}")
        differences[dataset] = np.asarray(values, dtype=float)
        counts[dataset] = len(values)

    observed = float(np.mean([values.mean() for values in differences.values()]))
    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty(args.samples, dtype=float)
    for index in range(args.samples):
        dataset_means = []
        for values in differences.values():
            draw = rng.integers(0, len(values), len(values))
            dataset_means.append(float(values[draw].mean()))
        bootstrap[index] = np.mean(dataset_means)
    output = {
        "candidate": args.candidate_method,
        "baseline": args.baseline_method,
        "estimand": "equal-dataset macro of paired within-video ROC-AUC differences",
        "n_defined_by_dataset": counts,
        "difference": observed,
        "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "p_difference_le_zero": float(np.mean(bootstrap <= 0)),
        "samples": args.samples,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
