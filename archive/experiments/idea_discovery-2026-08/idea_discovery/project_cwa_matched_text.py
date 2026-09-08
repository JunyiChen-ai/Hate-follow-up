#!/usr/bin/env python3
"""Cache-only CWA Stage-A: require a veto to beat null and length controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row
            for row in map(json.loads, path.open())}


def tie(key: tuple[str, str], salt: str) -> str:
    return hashlib.sha256(f"{salt}|{key[0]}|{key[1]}".encode()).hexdigest()


def matched_controls(key: tuple[str, str], manifest: dict, pool: list[tuple[str, str]]) -> list[tuple[str, str]]:
    target = len(str(manifest[key].get("transcript", "")))
    candidates = [other for other in pool if other != key and other[0] == key[0]]
    ranked_a = sorted(candidates, key=lambda other: (
        abs(len(str(manifest[other].get("transcript", ""))) - target), tie(other, "a")))
    ranked_b = sorted(candidates, key=lambda other: (
        abs(len(str(manifest[other].get("transcript", ""))) - target), tie(other, "b")))
    first = ranked_a[0]
    second = next((other for other in ranked_b if other != first), first)
    return [first, second]


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
    anchors = {float(text[key]["config"]["null_log_odds"]) for key in keys}
    if len(anchors) != 1:
        raise RuntimeError(f"expected one null anchor, got {anchors}")
    anchor = anchors.pop()
    controls = {key: matched_controls(key, manifest, keys) for key in keys}
    counts = {"consensus": 0, "oppose": 0, "abstain_keep": 0}

    for key in keys:
        e08, e12 = bool(a08[key].get("intervals")), bool(a12[key].get("intervals"))
        agreement = e08 == e12
        z = float(text[key]["log_odds"])
        control_keys = controls[key]
        control_scores = [float(text[other]["log_odds"]) for other in control_keys]
        # A negative witness must be stronger than the content-free reference
        # and both independently tie-broken, length-matched corpus controls.
        oppose = not agreement and z < anchor and all(z < value for value in control_scores)
        keep = e08 if agreement else not oppose
        state = "consensus" if agreement else "oppose" if oppose else "abstain_keep"
        counts[state] += 1
        source = extent[key]
        intervals = [Interval(float(v[0]), float(v[1]), float(v[2]) if len(v) > 2 else 1.0)
                     for v in source["intervals"]] if keep else []
        curve = source["score_curve"] if keep else [0.0] * len(source["score_curve"])
        append_jsonl(args.out, Prediction(
            "cwa_matched_text_v1", key[0], key[1], float(source["duration"]),
            score_curve=curve, intervals=intervals, calls=0,
            modality_evidence={
                "visual_agreement": agreement, "a08_nonempty": e08, "a12_nonempty": e12,
                "text_log_odds": z, "null_anchor": anchor, "control_keys": control_keys,
                "control_scores": control_scores, "warrant": state, "keep": keep,
            },
            raw={"gt_access": False, "matching": "same_dataset_nearest_transcript_length_two_hash_ties",
                 "authority": "non_compensatory", "no_cross_modal_score_comparison": True}))
    print(json.dumps({"n": len(keys), "counts": counts, "null_anchor": anchor}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
