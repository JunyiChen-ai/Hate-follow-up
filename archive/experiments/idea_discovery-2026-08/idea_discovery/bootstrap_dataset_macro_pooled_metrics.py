#!/usr/bin/env python3
"""Paired video-cluster bootstrap for equal-dataset pooled ROC/PR differences.

Frames are never sampled independently.  Within each dataset we resample common
videos with replacement, concatenate every selected video's complete timeline,
compute the pooled frame metric, and finally average the four dataset metrics.
This matches the reported equal-dataset macro estimand while respecting the
video as the sampling unit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


DATASETS = ("HateMM", "HateClipSeg", "MHC", "MHC_zh")


def load(path: Path, method: str) -> dict[tuple[str, str], dict]:
    rows = {}
    for row in map(json.loads, path.open()):
        if row.get("method") != method or row.get("error") or not row.get("score_curve"):
            continue
        key = (row["dataset"], row["video_id"])
        if key in rows:
            raise ValueError(f"duplicate prediction key: {key}")
        rows[key] = row
    if not rows:
        raise ValueError(f"no rows for method={method!r} in {path}")
    return rows


def metric(name: str, y: np.ndarray, score: np.ndarray) -> float:
    if name == "roc_auc":
        return float(roc_auc_score(y, score))
    if name == "pr_auc":
        return float(average_precision_score(y, score))
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-predictions", type=Path, required=True)
    ap.add_argument("--baseline-predictions", type=Path, required=True)
    ap.add_argument("--candidate-method", required=True)
    ap.add_argument("--baseline-method", required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    candidate = load(args.candidate_predictions, args.candidate_method)
    baseline = load(args.baseline_predictions, args.baseline_method)
    data: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for dataset in DATASETS:
        z = np.load(args.gt_dir / f"{dataset}.npz", allow_pickle=True)
        labels = {
            str(video_id): np.asarray(z["y4"][i], dtype=np.int8)
            for i, video_id in enumerate(z["video_ids"])
            if str(z["split"][i]) == "test"
        }
        ids = sorted(
            set(labels)
            & {v for d, v in candidate if d == dataset}
            & {v for d, v in baseline if d == dataset}
        )
        rows = []
        for video_id in ids:
            y = labels[video_id]
            a = np.asarray(candidate[(dataset, video_id)]["score_curve"], dtype=float)
            b = np.asarray(baseline[(dataset, video_id)]["score_curve"], dtype=float)
            if len(a) not in (len(y), len(y) + 1) or len(b) not in (len(y), len(y) + 1):
                raise ValueError(
                    f"unexpected alignment for {(dataset, video_id)}: "
                    f"gt={len(y)} candidate={len(a)} baseline={len(b)}"
                )
            rows.append((y, a[: len(y)], b[: len(y)]))
        if not rows:
            raise ValueError(f"no common videos for {dataset}")
        data[dataset] = rows

    point = {}
    per_dataset = {}
    for name in ("roc_auc", "pr_auc"):
        differences = []
        per_dataset[name] = {}
        for dataset, rows in data.items():
            y = np.concatenate([x[0] for x in rows])
            a = np.concatenate([x[1] for x in rows])
            b = np.concatenate([x[2] for x in rows])
            ca, cb = metric(name, y, a), metric(name, y, b)
            per_dataset[name][dataset] = {
                "candidate": ca, "baseline": cb, "difference": ca - cb,
                "n_videos": len(rows), "n_frames": int(len(y)),
            }
            differences.append(ca - cb)
        point[name] = float(np.mean(differences))

    rng = np.random.default_rng(args.seed)
    boot = {name: np.empty(args.samples, dtype=float) for name in point}
    for sample in range(args.samples):
        sample_differences = {name: [] for name in point}
        for rows in data.values():
            take = rng.integers(0, len(rows), len(rows))
            y = np.concatenate([rows[i][0] for i in take])
            a = np.concatenate([rows[i][1] for i in take])
            b = np.concatenate([rows[i][2] for i in take])
            for name in point:
                sample_differences[name].append(metric(name, y, a) - metric(name, y, b))
        for name in point:
            boot[name][sample] = np.mean(sample_differences[name])

    result = {
        "candidate": args.candidate_method,
        "baseline": args.baseline_method,
        "estimand": "equal-dataset macro of pooled-frame metric differences under paired video-cluster resampling",
        "resampling_unit": "video within dataset",
        "per_dataset": per_dataset,
        "paired_bootstrap": {
            name: {
                "difference": point[name],
                "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
                "p_difference_le_zero": float(np.mean(values <= 0)),
            }
            for name, values in boot.items()
        },
        "samples": args.samples,
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
