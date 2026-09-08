#!/usr/bin/env python3
"""Frozen, inference-only VASTA factorial for a sealed cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row for row in map(json.loads, path.open())}


def emit(out: Path, method: str, key: tuple[str, str], source: dict, keep: bool, evidence: dict) -> None:
    intervals = [Interval(float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 1.0)
                 for v in source["intervals"]] if keep else []
    curve = source["score_curve"] if keep else [0.0] * len(source["score_curve"])
    append_jsonl(out, Prediction(
        method, key[0], key[1], float(source["duration"]), score_curve=curve,
        intervals=intervals, calls=0, modality_evidence=evidence,
        raw={"gt_access": False, "protocol": "frozen_vasta_sealed_20260828"}))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rank1", type=Path, required=True)
    parser.add_argument("--extent", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    manifest, rank1, extent = load(args.manifest), load(args.rank1), load(args.extent)
    a08, a12, text = load(args.a08), load(args.a12), load(args.text)
    keys = sorted(set(manifest) & set(rank1) & set(extent) & set(a08) & set(a12) & set(text))
    if len(keys) != len(manifest):
        raise RuntimeError(f"incomplete common cohort: common={len(keys)} manifest={len(manifest)}")
    anchors = {float(text[key]["config"]["null_log_odds"]) for key in keys}
    if len(anchors) != 1:
        raise RuntimeError(f"expected one explicit text null, got {anchors}")
    anchor = anchors.pop()

    for key in keys:
        e08, e12 = bool(a08[key].get("intervals")), bool(a12[key].get("intervals"))
        agreement = e08 == e12
        z = float(text[key]["log_odds"])
        visual_or = e08 or e12
        vasta = e08 if agreement else z >= anchor
        emit(args.out, "sealed_a10_rank1", key, rank1[key], True, {"substrate": "rank1"})
        emit(args.out, "sealed_extent_top8", key, extent[key], True, {"substrate": "top8_envelope"})
        emit(args.out, "sealed_visual_or", key, extent[key], visual_or,
             {"a08_nonempty": e08, "a12_nonempty": e12})
        emit(args.out, "sealed_vasta", key, extent[key], vasta,
             {"visual_agreement": agreement, "a08_nonempty": e08,
              "a12_nonempty": e12, "text_log_odds": z, "null_anchor": anchor,
              "text_veto": not agreement and z < anchor})
    print(json.dumps({"videos": len(keys), "rows": len(keys) * 4, "null_anchor": anchor}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
