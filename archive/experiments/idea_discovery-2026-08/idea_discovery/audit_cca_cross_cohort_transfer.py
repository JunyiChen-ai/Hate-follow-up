#!/usr/bin/env python3
"""Audit CCA authority learned on cohort A and applied to disjoint cohort B."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from scripts.label_free_adapt.evaluate import interval_f1


def load(path: Path, method: str | None = None) -> dict[tuple[str, str], dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(row["dataset"], row["video_id"]): row for row in rows
            if method is None or row.get("method") == method}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-a08", type=Path, required=True)
    parser.add_argument("--calib-a12", type=Path, required=True)
    parser.add_argument("--calib-text", type=Path, required=True)
    parser.add_argument("--calib-dataset", required=True)
    parser.add_argument("--deploy-visual-or", type=Path, required=True)
    parser.add_argument("--deploy-visual-or-method")
    parser.add_argument("--deploy-vasta", type=Path, required=True)
    parser.add_argument("--deploy-vasta-method")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    a08, a12, text = load(args.calib_a08), load(args.calib_a12), load(args.calib_text)
    keys = sorted(key for key in set(a08) & set(a12) & set(text)
                  if key[0] == args.calib_dataset)
    anchors = {float(text[key]["config"]["null_log_odds"]) for key in keys}
    if len(anchors) != 1:
        raise RuntimeError(f"calibration anchor mismatch: {anchors}")
    anchor = anchors.pop(); agree = conflict = 0
    for key in keys:
        e08, e12 = bool(a08[key].get("intervals")), bool(a12[key].get("intervals"))
        if e08 == e12 and float(text[key]["log_odds"]) < anchor:
            if e08: conflict += 1
            else: agree += 1
    probability = float(beta.sf(0.5, agree + 1, conflict + 1))
    authorized = probability > args.confidence

    visual = load(args.deploy_visual_or, args.deploy_visual_or_method)
    vasta = load(args.deploy_vasta, args.deploy_vasta_method)
    visual_by_id = {key[1]: row for key, row in visual.items() if key[0] == args.dataset}
    vasta_by_id = {key[1]: row for key, row in vasta.items() if key[0] == args.dataset}
    gt = np.load(args.gt, allow_pickle=True)
    y = {str(video_id): np.asarray(gt["y4"][i], dtype=np.int8)
         for i, video_id in enumerate(gt["video_ids"])}
    selected = vasta_by_id if authorized else visual_by_id
    correct = wrong = 0
    for video_id in set(visual_by_id) & set(vasta_by_id) & set(y):
        if visual_by_id[video_id].get("intervals") and not vasta_by_id[video_id].get("intervals"):
            if y[video_id].any(): wrong += 1
            else: correct += 1
    output = {
        "calibration": {"n": len(keys), "veto_agrees_negative": agree,
                        "veto_conflicts_positive": conflict,
                        "probability_reliability_above_half": probability,
                        "confidence": args.confidence, "authorized": authorized},
        "deployment": {"dataset": args.dataset, "n": len(selected),
                       "candidate_veto_correct": correct, "candidate_veto_wrong": wrong,
                       "candidate_veto_net": correct - wrong,
                       "selected_action": "VASTA" if authorized else "visual_OR",
                       **interval_f1(y, selected)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
