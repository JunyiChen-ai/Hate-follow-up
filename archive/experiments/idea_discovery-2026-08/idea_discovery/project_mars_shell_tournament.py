#!/usr/bin/env python3
"""Project MARS left/right shell decisions onto a LESS midpoint baseline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


MIDPOINT = "fact_less_t3al_dualgeo_midpoint_v5"


def load(path):
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, Path(path).open())}


def decide_endpoint(trial, side, policy):
    lattice = trial["interval_lattice"]; index = 0 if side == "left" else 1
    tight = float(lattice["tight"][index])
    decisions = [d for d in trial["decisions"] if d["side"] == side]
    by_from = {float(d["from"]): d for d in decisions}; current = tight; audit = []
    while current in by_from:
        decision = by_from[current]
        if policy == "soft_text":
            weight = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0,
                float(decision["text_null_corrected_margin"])))))
            target = float(decision["to"])
            updated = current + weight * (target - current)
            audit.append({"from": current, "to": target, "weight": weight,
                          "updated": updated, "policy": policy})
            current = updated
            # Soft barycentric projection consumes the full fixed chain once;
            # the next shell is measured relative to its original lattice node.
            next_key = target
            if next_key not in by_from: break
            decision = by_from[next_key]
            weight = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0,
                float(decision["text_null_corrected_margin"])))))
            target2 = float(decision["to"])
            updated = current + weight * (target2 - current)
            audit.append({"from": next_key, "to": target2, "weight": weight,
                          "updated": updated, "policy": policy})
            current = updated
            break
        elif policy == "text":
            extend = decision["text_state"] == "extend"
        else:
            extend = decision["extend"]
        audit.append({"from": current, "to": float(decision["to"]),
                      "extend": extend, "policy": policy})
        if not extend: break
        current = float(decision["to"])
    return current, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--tribunal", type=Path, required=True)
    ap.add_argument("--policy", choices=("text", "bicameral", "soft_text"), default="text")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): raise RuntimeError(f"refusing existing output: {args.out}")
    trials = load(args.tribunal); counts = {"n": 0, "covered": 0, "eligible": 0,
        "changed": 0, "left_extensions": 0, "right_extensions": 0}
    with args.out.open("w") as handle:
        for row in map(json.loads, args.base.open()):
            if row["method"] != MIDPOINT: continue
            counts["n"] += 1; key = row["dataset"], row["video_id"]
            trial = trials.get(key); chosen = None; endpoint_audit = {}
            if trial:
                counts["covered"] += 1
                if trial.get("relation", {}).get("stance") == "endorsement":
                    counts["eligible"] += 1
                    left, la = decide_endpoint(trial, "left", args.policy)
                    right, ra = decide_endpoint(trial, "right", args.policy)
                    endpoint_audit = {"left": la, "right": ra}
                    counts["left_extensions"] += sum(x.get("extend", 0) for x in la)
                    counts["right_extensions"] += sum(x.get("extend", 0) for x in ra)
                    if right > left: chosen = [left, right, 1.0]
            output = dict(row); output["method"] = f"mars_shell_{args.policy}_pilot_v1"
            changed = False
            if chosen:
                old = row.get("intervals", [])
                changed = not old or list(map(float, old[0][:2])) != chosen[:2]
                output["intervals"] = [chosen]; counts["changed"] += int(changed)
            output["raw"] = {**row.get("raw", {}), "mars": {
                "gt_access": False, "policy": args.policy, "chosen": chosen,
                "endpoint_audit": endpoint_audit, "changed_interval": changed,
                "dense_curve_bit_identical": True}}
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__": main()
