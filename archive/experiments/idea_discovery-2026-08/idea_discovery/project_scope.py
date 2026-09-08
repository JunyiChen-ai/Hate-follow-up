#!/usr/bin/env python3
"""SCOPE: self-consistent proposal-conditioned multimodal authority.

Proposal majority overlap is the single state variable.  A stable visual state
keeps the rank-1 interval and closes outside intervention.  An unstable state
recovers proposal support and opens the frozen VASTA existence arbitration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl
from scripts.idea_discovery.project_uncertainty_extent import iou


def load(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row for row in map(json.loads, path.open())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--rank1", type=Path, required=True)
    parser.add_argument("--extent", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    proposals, rank1, extent = load(args.proposals), load(args.rank1), load(args.extent)
    a08, a12, text = load(args.a08), load(args.a12), load(args.text)
    keys = sorted(set(proposals) & set(rank1) & set(extent) & set(a08) & set(a12) & set(text))
    anchors = {float(text[key]["config"]["null_log_odds"]) for key in keys}
    if len(anchors) != 1:
        raise RuntimeError(f"expected one explicit text null, got {anchors}")
    anchor = anchors.pop()
    counts = {"stable_visual": 0, "unstable_visual": 0, "unstable_veto": 0}
    for key in keys:
        bank = sorted(proposals[key]["proposals"], key=lambda value: int(value["rank"]))
        first = (float(bank[0]["start"]), float(bank[0]["end"]))
        mean_iou = sum(iou(first, (float(v["start"]), float(v["end"]))) for v in bank[1:]) / max(1, len(bank) - 1)
        # IoU >= .5 is the conventional majority-overlap criterion, fixed
        # independently of all dataset labels and evaluation results.
        stable = mean_iou >= 0.5
        e08, e12 = bool(a08[key].get("intervals")), bool(a12[key].get("intervals"))
        agreement = e08 == e12
        z = float(text[key]["log_odds"])
        arbitration_keep = e08 if agreement else z >= anchor
        keep = True if stable else arbitration_keep
        source = rank1[key] if stable else extent[key]
        state = "stable_visual" if stable else "unstable_visual"
        counts[state] += 1
        counts["unstable_veto"] += int(not stable and not keep)
        intervals = [Interval(float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 1.0)
                     for v in source["intervals"]] if keep else []
        curve = source["score_curve"] if keep else [0.0] * len(source["score_curve"])
        append_jsonl(args.out, Prediction(
            "scope_v1", key[0], key[1], float(source["duration"]),
            score_curve=curve, intervals=intervals, calls=0,
            modality_evidence={
                "proposal_mean_iou": mean_iou, "visual_state": state,
                "a08_nonempty": e08, "a12_nonempty": e12,
                "visual_agreement": agreement, "text_log_odds": z,
                "null_anchor": anchor, "arbitration_open": not stable,
                "arbitration_keep": arbitration_keep, "keep": keep,
            },
            raw={"gt_access": False, "overlap_threshold": 0.5,
                 "threshold_semantics": "canonical proposal majority overlap",
                 "stable_action": "rank1_and_closed_authority",
                 "unstable_action": "support_recovery_and_vasta_arbitration"}))
    print(json.dumps({"n": len(keys), "counts": counts, "null_anchor": anchor}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
