#!/usr/bin/env python3
"""Build a label-free dense-only control with AMBI's exact action budget.

The action count is read from an AMBI projection, but ranking and endpoint
selection use only the fused transition field.  Ground truth is never loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path, method: str | None = None):
    rows = {}
    for line in path.open():
        row = json.loads(line)
        if method is None or row.get("method") == method:
            rows[(row["dataset"], row["video_id"])] = row
    return rows


def interval(row):
    return tuple(map(float, row["intervals"][0][:2])) if row and row.get("intervals") else None


def transition_strength(audit: dict, side: str) -> float:
    values = list(map(float, audit.get(side, {}).get("control_membership", [])))
    if len(values) < 2:
        return float("-inf")
    return max(abs(b - a) for a, b in zip(values, values[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoom", type=Path, required=True)
    ap.add_argument("--semantic", type=Path, required=True)
    ap.add_argument("--semantic-method", default="ambi_authorized_fused_core_v1")
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--base-method", required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--tight-method", default="fact_less_t3al_dualgeo_shorter_v5")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    ambi = load(args.zoom, "ambi_zoom_v1")
    fused = load(args.zoom, "ambi_fused_transition_control_v1")
    semantic = load(args.semantic, args.semantic_method)
    base = load(args.base, args.base_method)
    tight = load(args.bank, args.tight_method)

    target = sum(sum(bool(x) for x in row.get("raw", {}).get("accepted_sides", []))
                 for row in semantic.values())
    eligible = []
    for key in sorted(ambi):
        bi, fi, ti = interval(base.get(key)), interval(fused.get(key)), interval(tight.get(key))
        if bi is None or fi is None:
            continue
        audit = ambi[key].get("raw", {}).get("boundary_audit", {})
        for index, side in enumerate(("left", "right")):
            if abs(fi[index] - bi[index]) <= 1e-8:
                continue
            preserves = ti is None or (fi[index] <= ti[0] if index == 0 else fi[index] >= ti[1])
            if not preserves:
                continue
            strength = transition_strength(audit, side)
            if strength != float("-inf"):
                # Deterministic key tie-break makes this a frozen label-free ordering.
                eligible.append((strength, key[0], key[1], index, fi[index]))
    eligible.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    chosen = {(ds, vid, side): (value, strength)
              for strength, ds, vid, side, value in eligible[:target]}

    accepted = 0
    with args.out.open("w") as handle:
        for key in sorted(ambi):
            source = base.get(key) or ambi[key]
            output = dict(source)
            output["method"] = "fused_core_matched_action_v1"
            bi = interval(base.get(key))
            accepted_sides = [False, False]
            strengths = [None, None]
            result = bi
            if bi is not None:
                values = list(bi)
                for side in (0, 1):
                    item = chosen.get((key[0], key[1], side))
                    if item is not None:
                        values[side], strengths[side] = item
                        accepted_sides[side] = True
                if values[0] < values[1]:
                    result = tuple(values)
                    accepted += sum(accepted_sides)
                else:
                    accepted_sides = [False, False]
            output["intervals"] = [] if result is None else [[result[0], result[1], 1.0]]
            output["raw"] = {**output.get("raw", {}), "gt_access": False,
                             "control": "global_fused_transition_rank",
                             "matched_semantic_method": args.semantic_method,
                             "target_endpoint_actions": target,
                             "accepted_sides": accepted_sides,
                             "transition_strengths": strengths}
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps({"n_videos": len(ambi), "target_actions": target,
                      "eligible_actions": len(eligible), "accepted_actions": accepted,
                      "weakest_selected_strength": eligible[target-1][0] if target else None}, indent=2))


if __name__ == "__main__":
    main()
