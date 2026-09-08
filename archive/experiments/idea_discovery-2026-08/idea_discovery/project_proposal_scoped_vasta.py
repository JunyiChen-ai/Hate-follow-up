#!/usr/bin/env python3
"""Project proposal-scoped transcript arbitration over frozen visual states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row for row in map(json.loads, path.open())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extent", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--whole-text", type=Path, required=True)
    parser.add_argument("--scoped-text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    extent, a08, a12 = load(args.extent), load(args.a08), load(args.a12)
    whole, scoped = load(args.whole_text), load(args.scoped_text)
    keys = sorted(set(extent) & set(a08) & set(a12) & set(whole))
    anchors = {float(whole[key]["config"]["null_log_odds"]) for key in keys}
    if len(anchors) != 1:
        raise RuntimeError(f"expected one whole anchor, got {anchors}")
    anchor = anchors.pop(); changed = 0
    for key in keys:
        e08, e12 = bool(a08[key].get("intervals")), bool(a12[key].get("intervals"))
        agreement = e08 == e12
        whole_keep = float(whole[key]["log_odds"]) >= anchor
        scoped_row = scoped.get(key)
        scoped_keep = not bool(scoped_row["local_veto"]) if scoped_row else whole_keep
        keep = e08 if agreement else scoped_keep
        changed += int(not agreement and scoped_keep != whole_keep)
        source = extent[key]
        intervals = [Interval(float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 1.0)
                     for v in source["intervals"]] if keep else []
        curve = source["score_curve"] if keep else [0.0] * len(source["score_curve"])
        append_jsonl(args.out, Prediction(
            "proposal_scoped_vasta_v1", key[0], key[1], float(source["duration"]),
            score_curve=curve, intervals=intervals, calls=0,
            modality_evidence={"visual_agreement": agreement, "a08_nonempty": e08,
                               "a12_nonempty": e12, "whole_keep": whole_keep,
                               "scoped_available": scoped_row is not None,
                               "scoped_keep": scoped_keep, "keep": keep},
            raw={"gt_access": False, "text_scope": "rank1_visual_proposal",
                 "text_can_create_or_move_boundary": False}))
    print(json.dumps({"n": len(keys), "changed_from_whole_gate": changed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
