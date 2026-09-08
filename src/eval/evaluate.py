#!/usr/bin/env python3
"""Isolated evaluator for the shared prediction schema.

This is the only module in this package allowed to receive a ground-truth NPZ.
Inference manifests and adapters contain no label path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_intervals(y: np.ndarray, rate: float = 4.0) -> list[tuple[float, float]]:
    """Convert a binary frame grid to half-open intervals in seconds."""
    indices = np.flatnonzero(np.diff(np.r_[0, y.astype(np.int8), 0]))
    return [(float(a) / rate, float(b) / rate)
            for a, b in indices.reshape(-1, 2)]


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union > 0 else 0.0


def interval_f1(y_by_video: dict[str, np.ndarray], rows: dict[str, dict],
                thresholds=(0.3, 0.5, 0.7)) -> dict:
    """Greedy one-to-one event matching with micro event counts.

    Missing prediction rows are treated as empty interval sets, so coverage
    failures contribute false negatives instead of disappearing silently.
    """
    totals = {threshold: [0, 0, 0] for threshold in thresholds}  # TP, FP, FN
    for video_id in sorted(y_by_video):
        gt_intervals = binary_intervals(y_by_video[video_id])
        predicted = [(float(x[0]), float(x[1]))
                     for x in rows.get(video_id, {}).get("intervals", []) if len(x) >= 2]
        for threshold in thresholds:
            candidates = sorted(
                ((temporal_iou(pred, gt), pi, gi)
                 for pi, pred in enumerate(predicted)
                 for gi, gt in enumerate(gt_intervals)), reverse=True)
            used_pred, used_gt = set(), set()
            for overlap, pi, gi in candidates:
                if overlap < threshold:
                    break
                if pi not in used_pred and gi not in used_gt:
                    used_pred.add(pi); used_gt.add(gi)
            tp = len(used_pred)
            totals[threshold][0] += tp
            totals[threshold][1] += len(predicted) - tp
            totals[threshold][2] += len(gt_intervals) - tp
    output = {}
    for threshold, (tp, fp, fn) in totals.items():
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        output[f"interval_F1@{threshold}"] = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0)
        output[f"interval_precision@{threshold}"] = precision
        output[f"interval_recall@{threshold}"] = recall
    return output


def within_video_macro(y_by_video: dict[str, np.ndarray],
                       s_by_video: dict[str, np.ndarray]) -> dict:
    values = []
    skipped = []
    for video_id in sorted(set(y_by_video) & set(s_by_video)):
        y = np.asarray(y_by_video[video_id], dtype=np.int8)
        s = np.asarray(s_by_video[video_id], dtype=np.float64)
        n = min(len(y), len(s))
        y, s = y[:n], s[:n]
        ok = np.isfinite(s)
        y, s = y[ok], s[ok]
        if len(y) == 0 or y.min() == y.max():
            skipped.append(video_id)
            continue
        values.append(float(roc_auc_score(y, s)))
    return {
        "within_video_macro_ROC_AUC": float(np.mean(values)) if values else None,
        "within_video_macro_ROC_AUC_std": float(np.std(values)) if values else None,
        "n_videos_defined": len(values),
        "n_videos_skipped": len(skipped),
        "skipped_ids": skipped,
    }


def pooled(y_by_video: dict[str, np.ndarray], s_by_video: dict[str, np.ndarray]) -> dict:
    ys, ss = [], []
    for video_id in sorted(set(y_by_video) & set(s_by_video)):
        n = min(len(y_by_video[video_id]), len(s_by_video[video_id]))
        y = np.asarray(y_by_video[video_id][:n], dtype=np.int8)
        s = np.asarray(s_by_video[video_id][:n], dtype=np.float64)
        ok = np.isfinite(s)
        ys.append(y[ok]); ss.append(s[ok])
    if not ys or sum(map(len, ys)) == 0:
        return {"n_frames": 0}
    y, s = np.concatenate(ys), np.concatenate(ss)
    return {
        "n_frames": len(y),
        "frame_ROC_AUC": float(roc_auc_score(y, s)) if y.min() != y.max() else None,
        "frame_PR_AUC": float(average_precision_score(y, s)),
        "base_rate": float(y.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-npz", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--dataset", required=True,
                    help="Dataset name in prediction records; prevents cross-dataset mixing")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gt = np.load(args.gt_npz, allow_pickle=True)
    y_by_video = {
        str(v): np.asarray(gt["y4"][i], dtype=np.int8)
        for i, v in enumerate(gt["video_ids"])
        if str(gt["split"][i]) == args.split
    }
    # Append-only runners may retry failed records. The final record for each
    # (method, dataset, video) is authoritative, matching Retrieval-hate.
    latest = {}
    with open(args.predictions, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            latest[(row.get("method", "unknown"), row.get("dataset", "unknown"),
                    row["video_id"])] = row
    groups = {}
    for (method, dataset, video_id), row in latest.items():
        groups.setdefault((method, dataset), {})[video_id] = row
    result = []
    for (method, dataset), rows in sorted(groups.items()):
        if dataset != args.dataset:
            continue
        valid = {video_id: row for video_id, row in rows.items()
                 if not row.get("error") and row.get("score_curve")}
        missing_prediction_ids = sorted(set(y_by_video) - set(valid))
        scores = {video_id: np.asarray(row["score_curve"], dtype=float)
                  for video_id, row in valid.items()}
        result.append({
            "method": method, "dataset": dataset,
            **pooled(y_by_video, scores),
            **within_video_macro(y_by_video, scores),
            **interval_f1(y_by_video, valid),
            "n_videos_in_gt": len(y_by_video),
            "n_records_latest": len(rows),
            "n_videos_predicted": len(valid),
            "n_videos_missing_predictions": len(missing_prediction_ids),
            "missing_prediction_ids": missing_prediction_ids,
            "frame_metric_coverage": "complete_case_only",
            "interval_metric_coverage": "all_gt_missing_predictions_as_empty",
            "n_errors": sum(bool(row.get("error")) for row in rows.values()),
            "n_nonempty_intervals": sum(bool(row.get("intervals")) for row in valid.values()),
        })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
