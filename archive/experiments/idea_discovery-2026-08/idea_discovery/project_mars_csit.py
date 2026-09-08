#!/usr/bin/env python3
"""MARS pilot: conditional shell-interaction tournament over RCWL arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MIDPOINT = "fact_less_t3al_dualgeo_midpoint_v5"


def load(path):
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, Path(path).open())}


def contains(outer, inner, eps=1e-6):
    return outer[0] <= inner[0] + eps and outer[1] >= inner[1] - eps


def interaction(core, expanded, shifts):
    a, b = core["arm_log_odds"], expanded["arm_log_odds"]
    aligned = (b["keep"] - a["keep"]) - (a["remove"] - b["remove"])
    null = []
    for shift in shifts:
        prefix = f"shift{shift}_"
        null.append((b[prefix + "keep"] - a[prefix + "keep"])
                    - (a[prefix + "remove"] - b[prefix + "remove"]))
    corrected = aligned - float(np.median(null))
    return {"aligned": aligned, "shift_controls": null,
            "null_corrected": corrected, "expand": bool(corrected > 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--tribunal", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    trials = load(args.tribunal); audit = {"n": 0, "covered": 0, "eligible": 0,
        "changed": 0, "non_nested_fallback": 0, "selected": {}}
    with args.out.open("w") as handle:
        for row in map(json.loads, args.base.open()):
            if row["method"] != MIDPOINT:
                continue
            audit["n"] += 1; key = row["dataset"], row["video_id"]
            trial = trials.get(key); selected = None; rounds = []
            if trial:
                audit["covered"] += 1
                if trial.get("relation", {}).get("stance") == "endorsement":
                    candidates = sorted(trial["candidates"],
                        key=lambda c: (c["interval"][1] - c["interval"][0], c["name"]))
                    nested = all(contains(candidates[i + 1]["interval"], candidates[i]["interval"])
                                 for i in range(len(candidates) - 1))
                    if not nested:
                        audit["non_nested_fallback"] += 1
                    elif candidates:
                        audit["eligible"] += 1; selected = candidates[0]
                        shifts = trial.get("raw", {}).get("shifts", [4, 8, 12])
                        for expanded in candidates[1:]:
                            decision = interaction(selected, expanded, shifts)
                            rounds.append({"from": selected["name"], "to": expanded["name"], **decision})
                            if not decision["expand"]:
                                break
                            selected = expanded
            output = dict(row); output["method"] = "mars_csit_wholeshell_pilot_v1"
            if selected:
                new = [float(selected["interval"][0]), float(selected["interval"][1]), 1.0]
                old = row.get("intervals", [])
                changed = not old or list(map(float, old[0][:2])) != new[:2]
                output["intervals"] = [new]; audit["changed"] += int(changed)
                audit["selected"][selected["name"]] = audit["selected"].get(selected["name"], 0) + 1
            else:
                changed = False
            output["raw"] = {**row.get("raw", {}), "mars": {
                "gt_access": False, "pilot": "whole_shell",
                "dense_curve_bit_identical": True, "selected": selected,
                "rounds": rounds, "changed_interval": changed,
                "rule": "sequentially expand iff aligned shell interaction minus median shift interaction > 0",
            }}
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
