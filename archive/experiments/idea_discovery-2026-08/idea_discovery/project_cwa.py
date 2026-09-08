#!/usr/bin/env python3
"""Project CWA warrants with an exact frozen-VASTA fallback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row
            for row in map(json.loads, path.open())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extent", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--whole-text", type=Path, required=True)
    parser.add_argument("--warrants", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    manifest, extent = load(args.manifest), load(args.extent)
    a08, a12, whole = load(args.a08), load(args.a12), load(args.whole_text)
    warrants = load(args.warrants)
    keys = sorted(set(manifest) & set(extent) & set(a08) & set(a12) & set(whole))
    anchors = {float(whole[key]["config"]["null_log_odds"]) for key in keys}
    if len(anchors) != 1:
        raise RuntimeError(f"expected one whole-text null anchor, got {anchors}")
    anchor = anchors.pop()
    counts = {"support": 0, "oppose": 0, "abstain": 0, "consensus": 0}

    for key in keys:
        e08, e12 = bool(a08[key].get("intervals")), bool(a12[key].get("intervals"))
        agreement = e08 == e12
        vasta_keep = e08 if agreement else float(whole[key]["log_odds"]) >= anchor
        warrant = "consensus" if agreement else warrants.get(key, {}).get("warrant", "abstain")
        keep = e08 if agreement else True if warrant == "support" else False if warrant == "oppose" else vasta_keep
        counts[warrant] += 1
        source = extent[key]
        intervals = [Interval(float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 1.0)
                     for v in source["intervals"]] if keep else []
        curve = source["score_curve"] if keep else [0.0] * len(source["score_curve"])
        append_jsonl(args.out, Prediction(
            "cwa_v1", key[0], key[1], float(source["duration"]),
            score_curve=curve, intervals=intervals, calls=0,
            modality_evidence={"visual_agreement": agreement, "warrant": warrant,
                               "vasta_fallback_keep": vasta_keep, "keep": keep},
            raw={"gt_access": False, "fallback": "frozen_vasta_v3",
                 "authority": "non_compensatory"}))
    print(json.dumps({"n": len(keys), "counts": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
