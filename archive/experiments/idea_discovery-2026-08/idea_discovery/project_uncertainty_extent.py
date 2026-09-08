#!/usr/bin/env python3
"""Label-free proposal-disagreement calibrated temporal extent.

The rank-1 proposal is expanded toward the top-K support envelope in direct
proportion to its mean disagreement with the remaining proposals.  There is no
dataset threshold or fitted parameter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl, intervals_to_curve


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return intersection / union if union > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    rows = [json.loads(line) for line in args.proposals.read_text().splitlines() if line.strip()]
    disagreements = []
    for row in rows:
        proposals = sorted(row["proposals"], key=lambda value: int(value["rank"]))
        if not proposals:
            continue
        rank1 = (float(proposals[0]["start"]), float(proposals[0]["end"]))
        others = [(float(value["start"]), float(value["end"])) for value in proposals[1:]]
        agreement = float(np.mean([iou(rank1, value) for value in others])) if others else 1.0
        disagreement = 1.0 - agreement
        support_start = min(float(value["start"]) for value in proposals)
        support_end = max(float(value["end"]) for value in proposals)
        start = rank1[0] - disagreement * (rank1[0] - support_start)
        end = rank1[1] + disagreement * (support_end - rank1[1])
        duration = float(row["duration"])
        confidence = float(proposals[0].get("logit", 1.0))
        # A bounded monotone score is used only for the dense curve; interval
        # boundaries and the disagreement factor do not depend on it.
        score_value = float(1.0 / (1.0 + np.exp(-confidence)))
        interval = Interval(max(0.0, start), min(duration, end), score_value)
        append_jsonl(args.out, Prediction(
            "uncertainty_extent_v1", row["dataset"], row["video_id"], duration,
            score_curve=intervals_to_curve([interval], duration), intervals=[interval], calls=0,
            modality_evidence={"proposal_mean_iou": agreement,
                               "proposal_disagreement": disagreement,
                               "rank1": list(rank1), "support_envelope": [support_start, support_end]},
            raw={"gt_access": False, "fitted_parameters": 0,
                 "rule": "rank1_to_envelope_linear_by_one_minus_mean_iou"}))
        disagreements.append(disagreement)
    print(json.dumps({"n": len(disagreements), "mean_disagreement": float(np.mean(disagreements)),
                      "min": float(np.min(disagreements)), "max": float(np.max(disagreements))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
