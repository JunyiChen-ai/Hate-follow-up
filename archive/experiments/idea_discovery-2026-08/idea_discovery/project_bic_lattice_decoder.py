#!/usr/bin/env python3
"""Per-video BIC decoder over a nested endpoint lattice.

All lattice members have identical model complexity.  The selected span is
the one whose inside/outside two-state Gaussian model best explains that
video's dense evidence trace.  No labels, global threshold, or fitted dataset
constant enter the decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.run_ambi_boundary_zoom import BANK_METHODS, interval, load


def bic(signal: np.ndarray, span: tuple[float, float], duration: float) -> float:
    n = len(signal)
    lo = int(np.clip(np.floor(span[0] / max(duration, 1e-12) * n), 0, n))
    hi = int(np.clip(np.ceil(span[1] / max(duration, 1e-12) * n), 0, n))
    inside = np.zeros(n, dtype=bool); inside[lo:hi] = True
    if not inside.any() or inside.all():
        return float("inf")
    residual = signal.copy()
    residual[inside] -= np.mean(signal[inside])
    residual[~inside] -= np.mean(signal[~inside])
    variance = max(np.finfo(np.float64).tiny, float(np.mean(residual * residual)))
    # Common two-mean + one-variance model: equal parameter count for every
    # lattice member, retained explicitly to make the criterion auditable.
    return float(n * np.log(variance) + 3.0 * np.log(n))


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
    selected = {name: 0 for name in BANK_METHODS}
    with args.out.open("w") as handle:
        for key, row in sorted(fields.items()):
            candidates = {name: interval(rows.get(key)) for name, rows in bank.items()}
            candidates = {name: value for name, value in candidates.items() if value is not None}
            output = dict(row); output["method"] = "bic_lattice_decoder_v1"
            if not candidates:
                output["intervals"] = []; audit = {"fallback": "empty_lattice"}
            else:
                scores = np.clip(np.asarray(row["score_curve"], float), 1e-6, 1 - 1e-6)
                signal = np.log(scores / (1 - scores))
                records = {name: bic(signal, span, float(row["duration"]))
                           for name, span in candidates.items()}
                name = min(records, key=lambda x: (records[x], x))
                span = candidates[name]; selected[name] += 1
                output["intervals"] = [[span[0], span[1], 1.0]]
                audit = {"criterion": "per_video_two_state_gaussian_BIC",
                         "bic": records, "selected": name, "fallback": None}
            output["raw"] = {**output.get("raw", {}), "gt_access": False,
                             "module": "bic_nested_lattice_decoder",
                             "dataset_parameters": 0,
                             "label_selected_parameters": 0,
                             "decoder": audit}
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps({"n": len(fields), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
