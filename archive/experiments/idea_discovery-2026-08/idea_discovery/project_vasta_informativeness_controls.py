#!/usr/bin/env python3
"""Length/emptiness/random controls for VASTA's null-calibrated veto."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

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
    null = [float(text[key]["log_odds"]) for key in keys
            if not str(manifest[key].get("transcript", "")).strip()]
    anchor = float(median(null))
    disagreements = defaultdict(list)
    for key in keys:
        if bool(a08[key].get("intervals")) != bool(a12[key].get("intervals")):
            disagreements[key[0]].append(key)

    selected = {name: set() for name in ("shortest", "empty_only", "random")}
    for dataset, candidates in disagreements.items():
        n_veto = sum(float(text[key]["log_odds"]) < anchor for key in candidates)
        selected["shortest"].update(sorted(
            candidates,
            key=lambda key: (len(str(manifest[key].get("transcript", "")).strip()), key)
        )[:n_veto])
        selected["empty_only"].update(
            key for key in candidates
            if not str(manifest[key].get("transcript", "")).strip())
        selected["random"].update(sorted(
            candidates,
            key=lambda key: hashlib.sha256(
                f"vasta-info-control-v1\0{key[0]}\0{key[1]}".encode()).hexdigest()
        )[:n_veto])

    for key in keys:
        e08, e12 = bool(a08[key].get("intervals")), bool(a12[key].get("intervals"))
        agreement = e08 == e12
        visual_keep = e08 if agreement else True
        row = extent[key]
        base = [Interval(float(value[0]), float(value[1]),
                         float(value[2]) if len(value) > 2 else 1.0)
                for value in row["intervals"]]
        for control in ("shortest", "empty_only", "random"):
            keep = visual_keep and key not in selected[control]
            append_jsonl(args.out, Prediction(
                f"vasta_control_{control}", key[0], key[1], float(row["duration"]),
                score_curve=row["score_curve"] if keep else [0.0] * len(row["score_curve"]),
                intervals=base if keep else [], calls=0,
                modality_evidence={"visual_agreement": agreement,
                                   "control_veto": key in selected[control]},
                raw={"gt_access": False, "control": control}))
    print(json.dumps({"null_anchor": anchor,
                      "veto_counts": {name: len(values)
                                      for name, values in selected.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
