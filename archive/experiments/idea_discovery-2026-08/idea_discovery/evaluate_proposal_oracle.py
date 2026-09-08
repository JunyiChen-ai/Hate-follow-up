#!/usr/bin/env python3
"""Measure the event-recall ceiling of a label-free temporal proposal bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def tiou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.proposals.read_text().splitlines() if x.strip()]
    gt = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        z = np.load(args.gt_dir / f"{dataset}.npz", allow_pickle=True)
        for vid, spans in zip(z["video_ids"], z["spans"]):
            gt[(dataset, str(vid))] = [tuple(map(float, x[:2])) for x in spans]

    result = {"n_records": len(rows), "n_errors": sum(r["error"] is not None for r in rows),
              "by_m": {}}
    available = max((len(r["proposals"]) for r in rows), default=0)
    for m in (x for x in (1, 2, 4, 8, 16, 32) if x <= available):
        entry = {}
        for threshold in (0.3, 0.5, 0.7):
            per_dataset = {}
            total_hit = total_gt = 0
            for dataset in sorted({r["dataset"] for r in rows}):
                hit = count = 0
                for row in rows:
                    if row["dataset"] != dataset:
                        continue
                    props = [(p["start"], p["end"]) for p in row["proposals"][:m]]
                    for event in gt.get((dataset, row["video_id"]), []):
                        count += 1
                        hit += int(any(tiou(event, p) >= threshold for p in props))
                per_dataset[dataset] = {"hit": hit, "events": count,
                                        "recall": hit / count if count else None}
                total_hit += hit; total_gt += count
            entry[str(threshold)] = {"datasets": per_dataset, "pooled": total_hit / total_gt if total_gt else None,
                                     "hit": total_hit, "events": total_gt}
        result["by_m"][str(m)] = entry
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
