#!/usr/bin/env python3
"""Frozen quantifier decoder for NCL lattice evidence; contains no GT access."""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path):
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, path.open())}


def state(row, k=8):
    by = defaultdict(list)
    for item in row["observations"]:
        if int(item["rank"]) <= k:
            by[int(item["rank"])].append(item["winner"])
    stable = {rank: (values[0] if len(values) == 2 and values[0] == values[1] else "U")
              for rank, values in by.items()}
    if any(value in "BCD" for value in stable.values()):
        return "C+", stable
    if len(stable) == k and all(value in "AE" for value in stable.values()):
        return "C0", stable
    return "FALLBACK", stable


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--visual-warrant", type=Path)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--include-base", action="store_true")
    args = ap.parse_args()
    if args.out.exists(): raise RuntimeError(f"refusing existing output: {args.out}")
    evidence, base = load(args.evidence), load(args.base)
    warrants = load(args.visual_warrant) if args.visual_warrant else {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for key in sorted(evidence):
        verdict, winners = state(evidence[key], args.k); source = base[key]
        keep = verdict != "C0"
        intervals = ([Interval(float(x[0]), float(x[1]), float(x[2]) if len(x)>2 else 1.)
                      for x in source["intervals"]] if keep else [])
        curve = source["score_curve"] if keep else [0.] * len(source["score_curve"])
        append_jsonl(args.out, Prediction(
            "ncl_lattice_v1", key[0], key[1], float(source["duration"]),
            score_curve=curve, intervals=intervals, calls=0,
            modality_evidence={"certificate": verdict, "stable_winners": winners,
                               "k": args.k},
            raw={"gt_access": False, "base_method": source["method"],
                 "decoder": "exists B/C/D; forall A/E; otherwise fallback"}))
        if args.visual_warrant:
            margins = list(warrants.get(key, {}).get("modality_evidence", {}).get("margins", []))
            visual_counterwitness = (len(margins) == 2 and all(float(x) >= 0 for x in margins)
                                     and any(float(x) > 0 for x in margins))
            guarded_keep = keep or visual_counterwitness
            guarded_intervals = ([Interval(float(x[0]), float(x[1]),
                                                   float(x[2]) if len(x)>2 else 1.)
                                  for x in source["intervals"]] if guarded_keep else [])
            guarded_curve = source["score_curve"] if guarded_keep else [0.] * len(source["score_curve"])
            append_jsonl(args.out, Prediction(
                "ncl_bicameral_v2", key[0], key[1], float(source["duration"]),
                score_curve=guarded_curve, intervals=guarded_intervals, calls=0,
                modality_evidence={"text_certificate": verdict,
                                   "visual_native_margins": margins,
                                   "visual_counterwitness": visual_counterwitness,
                                   "keep": guarded_keep},
                raw={"gt_access": False, "base_method": source["method"],
                     "decoder": "text universal null subject to native visual veto"}))
        if args.include_base:
            append_jsonl(args.out, Prediction(
                "ncl_frozen_extent_control", key[0], key[1], float(source["duration"]),
                score_curve=source["score_curve"],
                intervals=[Interval(float(x[0]), float(x[1]), float(x[2]) if len(x)>2 else 1.)
                           for x in source["intervals"]], calls=0,
                modality_evidence={"control": "bit_exact_extent_without_null_competition"},
                raw={"gt_access": False, "base_method": source["method"]}))
    print(json.dumps({"n": len(evidence), "out": str(args.out)}))


if __name__ == "__main__": main()
