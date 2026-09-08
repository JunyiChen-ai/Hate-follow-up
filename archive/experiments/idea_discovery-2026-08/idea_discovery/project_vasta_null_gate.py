#!/usr/bin/env python3
"""Project VASTA using a label-free empty-transcript calibration anchor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path: Path):
    return {(row["dataset"], row["video_id"]): row
            for row in map(json.loads, path.open())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extent", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    manifest, extent = load(args.manifest), load(args.extent)
    a08, a12, text = load(args.a08), load(args.a12), load(args.text)
    keys = sorted(set(manifest) & set(extent) & set(a08) & set(a12) & set(text))
    anchors = {float(text[key].get("config", {}).get("null_log_odds"))
               for key in keys
               if text[key].get("config", {}).get("null_log_odds") is not None}
    if len(anchors) != 1:
        raise RuntimeError(f"expected one explicit null anchor, got {anchors}")
    null_anchor = anchors.pop()
    natural_null_scores = [float(text[key]["log_odds"]) for key in keys
                           if not str(manifest[key].get("transcript", "")).strip()]

    for key in keys:
        e08, e12 = bool(a08[key].get("intervals")), bool(a12[key].get("intervals"))
        log_odds = float(text[key]["log_odds"])
        agreement = e08 == e12
        keep = e08 if agreement else log_odds >= null_anchor
        row = extent[key]
        intervals = ([Interval(float(value[0]), float(value[1]),
                               float(value[2]) if len(value) > 2 else 1.0)
                      for value in row["intervals"]] if keep else [])
        curve = row["score_curve"] if keep else [0.0] * len(row["score_curve"])
        append_jsonl(args.out, Prediction(
            "vasta_null_anchor", key[0], key[1], float(row["duration"]),
            score_curve=curve, intervals=intervals, calls=0,
            modality_evidence={
                "visual_agreement": agreement,
                "a08_nonempty": e08,
                "a12_nonempty": e12,
                "text_log_odds": log_odds,
                "null_anchor": null_anchor,
                "text_veto": (not agreement and log_odds < null_anchor),
            },
            raw={"gt_access": False, "extent_source": row["method"],
                 "calibration": "single explicit empty-transcript query"}))
    print(json.dumps({"n": len(keys), "n_natural_null": len(natural_null_scores),
                      "null_anchor": null_anchor}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
