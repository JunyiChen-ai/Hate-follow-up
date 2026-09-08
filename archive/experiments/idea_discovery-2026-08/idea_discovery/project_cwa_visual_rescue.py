#!/usr/bin/env python3
"""Allow a reproducible native-frame warrant to rescue a VASTA text veto."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row for row in map(json.loads, path.open())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vasta", type=Path, required=True)
    parser.add_argument("--extent", type=Path, required=True)
    parser.add_argument("--warrants", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", choices=("stable", "offset0", "offset1", "inverted"),
                        default="stable")
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    vasta, extent, warrants = load(args.vasta), load(args.extent), load(args.warrants)
    rescued = 0
    for key in sorted(set(vasta) & set(extent)):
        base, source = vasta[key], extent[key]
        witness = warrants.get(key, {})
        margins = [float(value) for value in witness.get("margins", [])]
        if args.policy == "stable":
            support = len(margins) == 2 and all(value > 0 for value in margins)
        elif args.policy == "offset0":
            support = len(margins) == 2 and margins[0] > 0
        elif args.policy == "offset1":
            support = len(margins) == 2 and margins[1] > 0
        else:
            support = len(margins) == 2 and all(value < 0 for value in margins)
        was_veto = bool(base.get("modality_evidence", {}).get("text_veto"))
        rescue = was_veto and support
        keep = bool(base.get("intervals")) or rescue
        rescued += int(rescue)
        intervals = [Interval(float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 1.0)
                     for v in source["intervals"]] if keep else []
        curve = source["score_curve"] if keep else [0.0] * len(source["score_curve"])
        append_jsonl(args.out, Prediction(
            f"cwa_visual_rescue_{args.policy}_v1", key[0], key[1], float(source["duration"]),
            score_curve=curve, intervals=intervals, calls=0,
            modality_evidence={"vasta_keep": bool(base.get("intervals")),
                               "visual_support_warrant": support, "rescued": rescue,
                               "policy": args.policy, "margins": margins},
            raw={"gt_access": False, "base": "frozen_vasta_v3",
                 "authority": "visual support can reject text veto; text cannot move boundaries"}))
    print(json.dumps({"n": len(set(vasta) & set(extent)), "rescued": rescued}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
