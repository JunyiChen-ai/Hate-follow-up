#!/usr/bin/env python3
"""Parameter-free endpoint transport authorized by within-video gradient rank.

For each side, the candidate coordinate is the strongest direction-compatible
transition inside the interval envelope supplied by the frozen proposal bank.
Its authority is its average-tie percentile among every transition in that
video.  The endpoint moves continuously by the positive excess over the null
median.  No action budget, threshold, radius, gain, or dataset parameter is
used.  A reversed-field arm is emitted as a nuisance control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


BANK = {"tight": "fact_less_t3al_dualgeo_shorter_v5", "mid": "fact_less_t3al_dualgeo_midpoint_v5", "broad": "fact_less_t3al_dualgeo_union_v5"}


def load(path: Path, method: str) -> dict[tuple[str, str], dict]:
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, path.open()) if r.get("method") == method}


def interval(row: dict | None) -> tuple[float, float] | None:
    return tuple(map(float, row["intervals"][0][:2])) if row and row.get("intervals") else None


def endpoint(field: np.ndarray, duration: float, candidates: list[float], incumbent: float, side: int) -> tuple[float, dict]:
    rate = len(field) / duration
    lo = max(0, int(np.floor(min(candidates) * rate)))
    hi = min(len(field) - 1, int(np.ceil(max(candidates) * rate)))
    gradient = np.diff(field)
    directed = gradient if side == 0 else -gradient
    if hi <= lo or not len(gradient):
        return incumbent, {"fallback": True}
    indices = np.arange(lo, min(hi, len(gradient)))
    best = int(indices[np.argmax(directed[indices])])
    ranks = rankdata(directed, method="average") / len(directed)
    authority = max(0.0, 2.0 * float(ranks[best]) - 1.0)
    coordinate = (best + 0.5) / rate
    moved = incumbent + authority * (coordinate - incumbent)
    return float(moved), {"fallback": False, "candidate": coordinate, "rank": float(ranks[best]), "authority": authority, "envelope": [min(candidates), max(candidates)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--field-method", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-method", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    fields, base = load(args.field, args.field_method), load(args.base, args.base_method)
    bank = {name: load(args.bank, method) for name, method in BANK.items()}
    methods = ("orbit_ranked_endpoints_v1", "orbit_ranked_endpoints_reverse_v1")
    with args.out.open("w") as handle:
        for key, row in sorted(base.items()):
            source_interval = interval(row)
            proposals = {name: interval(rows.get(key)) for name, rows in bank.items()}
            for method in methods:
                output = dict(fields.get(key, row)); output["method"] = method
                audits = []; result = source_interval
                if source_interval is not None and all(x is not None for x in proposals.values()) and key in fields:
                    score = np.asarray(fields[key]["score_curve"], dtype=np.float64)
                    if method.endswith("reverse_v1"):
                        score = score[::-1]
                    values = list(source_interval)
                    for side in (0, 1):
                        values[side], audit = endpoint(score, float(row["duration"]), [x[side] for x in proposals.values()], source_interval[side], side)
                        audits.append(audit)
                    tight = proposals["tight"]
                    values[0] = min(values[0], tight[0]); values[1] = max(values[1], tight[1])
                    if values[0] < values[1]:
                        result = tuple(values)
                output["intervals"] = [] if result is None else [[result[0], result[1], 1.0]]
                output["raw"] = {**output.get("raw", {}), "gt_access": False,
                    "module": "orbit_ranked_endpoint_transport_v1", "dataset_parameters": 0,
                    "action_budget": None, "numeric_parameters": 0,
                    "tight_core_preserved": True, "endpoint_audit": audits}
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
