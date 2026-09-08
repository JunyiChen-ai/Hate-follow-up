#!/usr/bin/env python3
"""Label-free self-consistency fallbacks for AMBI bilateral edits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = {
    "tight": "fact_less_t3al_dualgeo_shorter_v5",
    "midpoint": "fact_less_t3al_dualgeo_midpoint_v5",
}


def load(path: Path, method: str | None = None):
    return {(row["dataset"], row["video_id"]): row for row in map(json.loads, path.open())
            if method is None or row.get("method") == method}


def interval(row):
    return tuple(map(float, row["intervals"][0][:2])) if row and row.get("intervals") else None


def direction(value: float, reference: float, epsilon: float = 1e-8) -> int:
    return int(value > reference + epsilon) - int(value < reference - epsilon)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path)
    parser.add_argument("--base-method")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    ambi = load(args.predictions, "ambi_zoom_v1")
    control = load(args.predictions, "ambi_fused_transition_control_v1")
    bank = {name: load(args.bank, method) for name, method in METHODS.items()}
    if bool(args.base_predictions) != bool(args.base_method):
        raise ValueError("--base-predictions and --base-method must be supplied together")
    selected_base = (load(args.base_predictions, args.base_method)
                     if args.base_predictions else bank["midpoint"])
    counts = {"direction": 0, "core_direction": 0, "fused_core": 0,
              "ambi_authorized_fused": 0}
    with args.out.open("w") as handle:
        for key, row in sorted(ambi.items()):
            base = interval(selected_base.get(key))
            core = interval(bank["tight"].get(key))
            ai, ci = interval(row), interval(control.get(key))
            for method, preserve_core in (("ambi_direction_consensus_v1", False),
                                          ("ambi_core_direction_consensus_v1", True)):
                output = dict(row); output["method"] = method
                accepted = [False, False]
                if base is None or ai is None or ci is None:
                    chosen = base
                else:
                    values = list(base)
                    for side in (0, 1):
                        da, dc = direction(ai[side], base[side]), direction(ci[side], base[side])
                        allow = da != 0 and da == dc
                        if allow and preserve_core and core is not None:
                            allow = ai[side] <= core[0] if side == 0 else ai[side] >= core[1]
                        if allow:
                            values[side] = ai[side]; accepted[side] = True
                    chosen = tuple(values) if values[0] < values[1] else base
                output["intervals"] = [] if chosen is None else [[chosen[0], chosen[1], 1.0]]
                output["raw"] = {**output.get("raw", {}), "gt_access": False,
                                 "self_consistency": "ambi_fused_direction_agreement",
                                 "preserve_tight_core": preserve_core,
                                 "accepted_sides": accepted}
                counts["core_direction" if preserve_core else "direction"] += sum(accepted)
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")
            # Matched kill control: the fused field proposes endpoints and the
            # same tight-core authority is applied, without any MLLM membership.
            output = dict(control[key]); output["method"] = "fused_core_preserving_v1"
            accepted = [False, False]
            if base is None or ci is None:
                chosen = base
            else:
                values = list(base)
                for side in (0, 1):
                    allow = direction(ci[side], base[side]) != 0
                    if allow and core is not None:
                        allow = ci[side] <= core[0] if side == 0 else ci[side] >= core[1]
                    if allow:
                        values[side] = ci[side]; accepted[side] = True
                chosen = tuple(values) if values[0] < values[1] else base
            output["intervals"] = [] if chosen is None else [[chosen[0], chosen[1], 1.0]]
            output["raw"] = {**output.get("raw", {}), "gt_access": False,
                             "self_consistency": "fused_transition_only",
                             "preserve_tight_core": True, "accepted_sides": accepted}
            counts["fused_core"] += sum(accepted)
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")

            # Preferred separation of powers: AMBI supplies semantic authority,
            # while the dense fused field supplies the endpoint coordinate.
            output = dict(row); output["method"] = "ambi_authorized_fused_core_v1"
            accepted = [False, False]
            if base is None or ai is None or ci is None:
                chosen = base
            else:
                values = list(base)
                for side in (0, 1):
                    da, dc = direction(ai[side], base[side]), direction(ci[side], base[side])
                    allow = da != 0 and da == dc
                    if allow and core is not None:
                        allow = ci[side] <= core[0] if side == 0 else ci[side] >= core[1]
                    if allow:
                        values[side] = ci[side]; accepted[side] = True
                chosen = tuple(values) if values[0] < values[1] else base
            output["intervals"] = [] if chosen is None else [[chosen[0], chosen[1], 1.0]]
            output["raw"] = {**output.get("raw", {}), "gt_access": False,
                             "self_consistency": "ambi_authorizes_fused_coordinate",
                             "preserve_tight_core": True, "accepted_sides": accepted}
            counts["ambi_authorized_fused"] += sum(accepted)
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps({"n": len(ambi), "accepted_endpoint_actions": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
