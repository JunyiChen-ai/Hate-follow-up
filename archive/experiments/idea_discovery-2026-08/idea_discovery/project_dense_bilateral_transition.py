#!/usr/bin/env python3
"""Full-cohort dense-only reconstruction of the AMBI transition control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.run_ambi_boundary_zoom import (
    BANK_METHODS, boundary_from_membership, control_membership, interval,
    load, local_times,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--field", type=Path, required=True)
    ap.add_argument("--field-method", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    bank = load(args.bank, BANK_METHODS)
    fields = load(args.field, {"field": args.field_method})["field"]
    changed = 0
    with args.out.open("w") as handle:
        for key, row in sorted(fields.items()):
            candidates = {name: interval(rows.get(key)) for name, rows in bank.items()}
            tight, midpoint, broad = (candidates[x] for x in ("tight", "midpoint", "broad"))
            output = dict(row); output["method"] = "dense_bilateral_transition_v1"
            if any(x is None for x in (tight, midpoint, broad)):
                output["intervals"] = [] if midpoint is None else [[*midpoint, 1.0]]
                audit = {"fallback": "missing_lattice"}
            else:
                duration = float(row["duration"])
                scores = np.clip(np.asarray(row["score_curve"], float), 1e-6, 1 - 1e-6)
                field = np.log(scores / (1 - scores)); field -= field.mean()
                endpoints, audit = {}, {}
                for side_index, side in enumerate(("left", "right")):
                    times = local_times(tight, broad, midpoint, duration, side_index)
                    memberships = control_membership(field, times, duration)
                    endpoints[side], transition = boundary_from_membership(times, memberships, side)
                    audit[side] = {"times": times.tolist(), "membership": memberships,
                                   "transition": transition}
                start, end = endpoints["left"], endpoints["right"]
                if start >= end:
                    start, end = midpoint; audit["fallback"] = "invalid_geometry"
                else:
                    audit["fallback"] = None
                    changed += int((start, end) != midpoint)
                output["intervals"] = [[start, end, 1.0]]
            output["raw"] = {**output.get("raw", {}), "gt_access": False,
                             "module": "dense_bilateral_transition",
                             "boundary_audit": audit}
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps({"n": len(fields), "changed": changed}, indent=2))


if __name__ == "__main__":
    main()
