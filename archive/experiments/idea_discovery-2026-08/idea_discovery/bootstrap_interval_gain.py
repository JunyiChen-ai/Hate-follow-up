#!/usr/bin/env python3
"""Dataset-balanced paired bootstrap for interval-F1 differences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.label_free_adapt.evaluate import binary_intervals, temporal_iou


GT_NAMES = ("HateMM", "HateClipSeg", "MHC", "MHC_zh")


def load_predictions(path, method=None):
    out = {}
    for line in Path(path).open(encoding="utf-8"):
        r = json.loads(line)
        if not r.get("error") and (method is None or r.get("method") == method):
            out[(r["dataset"], r["video_id"])] = r
    return out


def counts(gt, pred, threshold):
    truth = binary_intervals(gt)
    predicted = [(float(x[0]), float(x[1])) for x in pred.get("intervals", [])]
    candidates = sorted(((temporal_iou(p, g), pi, gi)
                         for pi, p in enumerate(predicted)
                         for gi, g in enumerate(truth)), reverse=True)
    up, ug = set(), set()
    for overlap, pi, gi in candidates:
        if overlap < threshold: break
        if pi not in up and gi not in ug: up.add(pi); ug.add(gi)
    return np.asarray([len(up), len(predicted) - len(up), len(truth) - len(ug)], int)


def f1(c):
    tp, fp, fn = map(float, c)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate-method")
    ap.add_argument("--baseline-method")
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=.5)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cand = load_predictions(args.candidate, args.candidate_method)
    base = load_predictions(args.baseline, args.baseline_method)
    per_dataset = {}
    for ds in GT_NAMES:
        z = np.load(args.gt_dir / f"{ds}.npz", allow_pickle=True)
        gt = {str(v): np.asarray(z["y4"][i], np.int8) for i, v in enumerate(z["video_ids"])
              if str(z["split"][i]) == "test"}
        ids = sorted(v for v in gt if (ds, v) in cand and (ds, v) in base)
        per_dataset[ds] = (ids,
            np.asarray([counts(gt[v], cand[(ds,v)], args.threshold) for v in ids]),
            np.asarray([counts(gt[v], base[(ds,v)], args.threshold) for v in ids]))
    rng = np.random.default_rng(args.seed)
    diffs = []
    for _ in range(args.samples):
        dc, db = [], []
        for ids, cc, bc in per_dataset.values():
            take = rng.integers(0, len(ids), len(ids))
            dc.append(f1(cc[take].sum(0))); db.append(f1(bc[take].sum(0)))
        diffs.append(float(np.mean(dc) - np.mean(db)))
    point_c = np.mean([f1(x[1].sum(0)) for x in per_dataset.values()])
    point_b = np.mean([f1(x[2].sum(0)) for x in per_dataset.values()])
    result = {"threshold": args.threshold, "datasets": list(GT_NAMES),
              "n_by_dataset": {d: len(v[0]) for d,v in per_dataset.items()},
              "candidate_macro_f1": float(point_c), "baseline_macro_f1": float(point_b),
              "difference": float(point_c-point_b), "bootstrap_samples": args.samples,
              "seed": args.seed, "difference_ci95": np.quantile(diffs,[.025,.975]).tolist(),
              "p_difference_le_zero": float(np.mean(np.asarray(diffs) <= 0))}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
