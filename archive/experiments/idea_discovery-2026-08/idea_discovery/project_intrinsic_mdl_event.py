#!/usr/bin/env python3
"""Score-only, sample-adaptive MDL event decoder.

The decoder compares a one-state timeline against every oriented contiguous
two-state explanation whose endpoints occur at intrinsic score changes.  The
alternative must pay both its extra Gaussian-mean parameter and the exact
uniform code length of identifying one span from the video's own endpoint
lattice.  No proposal detector, label, threshold, duration prior, or dataset
parameter is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def load(path: Path, method: str) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row
            for row in map(json.loads, path.open()) if row.get("method") == method}


def decode(values: np.ndarray) -> tuple[tuple[int, int] | None, dict]:
    n = len(values)
    if n < 3 or float(np.ptp(values)) <= np.finfo(float).eps:
        return None, {"state": "ONE_STATE", "reason": "constant_or_too_short"}
    # Exact score-change boundaries plus video edges form the sample's own
    # finite endpoint alphabet.  Numerical equality is intentional: upstream
    # evidence fields are piecewise constant by construction.
    endpoints = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, n]
    prefix = np.r_[0.0, np.cumsum(values)]
    prefix2 = np.r_[0.0, np.cumsum(values * values)]
    total, total2 = float(prefix[-1]), float(prefix2[-1])
    null_sse = max(np.finfo(float).tiny, total2 - total * total / n)
    null_code = n * math.log(null_sse / n)
    span_count = len(endpoints) * (len(endpoints) - 1) // 2
    # One extra mean plus an exact index into the finite span alphabet.
    alternative_penalty = math.log(n) + 2.0 * math.log(max(1, span_count))
    best = (math.inf, 0, 0, 0.0, 0.0)
    for left_index, left in enumerate(endpoints[:-1]):
        right = endpoints[left_index + 1:]
        inside_n = right - left
        outside_n = n - inside_n
        valid = outside_n > 0
        if not np.any(valid):
            continue
        right = right[valid]
        inside_n = inside_n[valid]
        outside_n = outside_n[valid]
        inside_sum = prefix[right] - prefix[left]
        outside_sum = total - inside_sum
        oriented = inside_sum / inside_n > outside_sum / outside_n
        if not np.any(oriented):
            continue
        right, inside_n, outside_n = right[oriented], inside_n[oriented], outside_n[oriented]
        inside_sum, outside_sum = inside_sum[oriented], outside_sum[oriented]
        inside_sse = ((prefix2[right] - prefix2[left])
                      - inside_sum * inside_sum / inside_n)
        outside_sse = ((total2 - (prefix2[right] - prefix2[left]))
                       - outside_sum * outside_sum / outside_n)
        sse = np.maximum(np.finfo(float).tiny, inside_sse + outside_sse)
        code = n * np.log(sse / n) + alternative_penalty
        index = int(np.argmin(code))
        record = (float(code[index]), int(left), int(right[index]),
                  float(inside_sum[index] / inside_n[index]),
                  float(outside_sum[index] / outside_n[index]))
        if record[:3] < best[:3]:
            best = record
    if not np.isfinite(best[0]) or best[0] >= null_code:
        return None, {"state": "ONE_STATE", "null_code": null_code,
                      "best_two_state_code": None if not np.isfinite(best[0]) else best[0],
                      "endpoint_count": int(len(endpoints)), "span_count": int(span_count)}
    return (best[1], best[2]), {
        "state": "LOCAL_EVENT", "null_code": null_code,
        "best_two_state_code": best[0], "code_gain": null_code - best[0],
        "inside_mean": best[3], "outside_mean": best[4],
        "endpoint_count": int(len(endpoints)), "span_count": int(span_count),
        "alternative_penalty": alternative_penalty,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    rows = load(args.prediction, args.method)
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    states = {"ONE_STATE": 0, "LOCAL_EVENT": 0}
    fractions = []
    with args.out.open("w") as handle:
        for key, row in sorted(rows.items()):
            score = np.asarray(row["score_curve"], float)
            logit = np.log(np.clip(score, 1e-6, 1 - 1e-6) /
                           np.clip(1 - score, 1e-6, 1))
            span, audit = decode(logit)
            states[audit["state"]] += 1
            rate = float(row.get("native_rate", 4.0) or 4.0)
            intervals = []
            if span is not None:
                left, right = span
                fractions.append((right - left) / len(score))
                intervals = [[left / rate, min(float(row["duration"]), right / rate),
                              float(score[left:right].mean())]]
            output = dict(row)
            output["method"] = "intrinsic_mdl_event_decoder_v1"
            output["intervals"] = intervals
            output["raw"] = {**output.get("raw", {}), "gt_access": False,
                             "module": "intrinsic_score_change_mdl_event_decoder",
                             "candidate_source": "score_change_endpoints_only",
                             "dataset_parameters": 0, "label_selected_parameters": 0,
                             "score_threshold": None, "duration_prior": None,
                             "decoder": audit, "code_sha256": code_hash}
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
    summary = {"n": len(rows), "states": states,
               "duration_fraction_mean": float(np.mean(fractions)) if fractions else None,
               "duration_fraction_median": float(np.median(fractions)) if fractions else None}
    args.out.with_suffix(".audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
