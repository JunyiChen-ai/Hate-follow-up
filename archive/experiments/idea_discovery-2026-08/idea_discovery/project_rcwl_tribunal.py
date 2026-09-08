#!/usr/bin/env python3
"""Apply a frozen RCWL tribunal to LESS interval hypotheses.

The MLLM only selects one existing geometry.  LESS's dense score curve remains
bit-identical.  Videos without a strictly positive aligned and null-corrected
endorsement certificate exactly fall back to the midpoint baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MIDPOINT = "fact_less_t3al_dualgeo_midpoint_v5"


def load_tribunal(path):
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, Path(path).open())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--tribunal", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--selection", choices=("minimal", "strongest"), default="minimal")
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    tribunal = load_tribunal(args.tribunal); audit = {
        "base_rows": 0, "tribunal_covered": 0, "endorsement": 0,
        "certificate_pass": 0, "changed": 0, "selected_tight": 0,
        "selected_midpoint": 0, "selected_broad": 0,
    }
    with args.out.open("w") as handle:
        for row in map(json.loads, args.base.open()):
            if row["method"] != MIDPOINT:
                continue
            audit["base_rows"] += 1; key = row["dataset"], row["video_id"]
            trial = tribunal.get(key); selected = None
            if trial:
                audit["tribunal_covered"] += 1
                if trial.get("relation", {}).get("stance") == "endorsement":
                    audit["endorsement"] += 1
                    feasible = [c for c in trial["candidates"]
                                if c["aligned_min_effect"] > 0
                                and c["null_corrected_certificate"] > 0]
                    if feasible:
                        audit["certificate_pass"] += 1
                        if args.selection == "minimal":
                            # RCWL is a minimal-witness method: feasibility is
                            # a hard gate, then length is the primary objective.
                            selected = min(feasible, key=lambda c: (
                                c["interval"][1] - c["interval"][0],
                                -c["null_corrected_certificate"],
                                -c["aligned_min_effect"]))
                        else:
                            selected = max(feasible, key=lambda c: (
                                c["null_corrected_certificate"],
                                c["aligned_min_effect"],
                                -(c["interval"][1] - c["interval"][0])))
            output = dict(row); output["method"] = f"rcwl_tribunal_{args.selection}_v2"
            if selected is not None:
                old = [[float(x[0]), float(x[1])] for x in row.get("intervals", [])]
                new = [float(selected["interval"][0]), float(selected["interval"][1]), 1.0]
                output["intervals"] = [new]
                changed = not old or old[0][:2] != new[:2]
                audit["changed"] += int(changed)
                audit[f"selected_{selected['name']}"] += 1
            else:
                changed = False
            output["raw"] = {**row.get("raw", {}),
                "rcwl": {
                    "gt_access": False, "dense_curve_exact_fallback": True,
                    "tribunal_available": trial is not None,
                    "selected": selected,
                    "changed_interval": changed,
                    "rule": "stance=endorsement AND aligned_min>0 AND null_corrected>0",
                }}
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
