#!/usr/bin/env python3
"""Cross-fitted interval witnesses with held-out-modality randomization tests.

For each modality m, the other two modalities select a maximum positive
excursion.  Modality m is never used by that selector and subsequently tests
the selected span against its complete cyclic orbit.  This yields three honest
interval-specific randomization p-values under the corresponding held-out
relative-phase nulls.  Coordinate-wise median endpoints provide a symmetric
prediction readout; the three certificates remain attached to their own spans.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_multiscale_scan_boundary import MAIN
from scripts.idea_discovery.project_role_orthogonal_transport import load
from scripts.idea_discovery.project_shapley_boundary import midrank


def max_positive_excursion(signal: np.ndarray) -> tuple[int, int]:
    """Maximum-sum nonempty subarray, with shortest/earliest deterministic ties."""
    best_sum = -np.inf
    best = (0, 1)
    running_sum = 0.0
    running_start = 0
    for end, value in enumerate(signal, 1):
        if running_sum <= 0:
            running_sum = float(value)
            running_start = end - 1
        else:
            running_sum += float(value)
        candidate = (running_start, end)
        if (running_sum > best_sum
                or (running_sum == best_sum
                    and (candidate[1] - candidate[0], candidate[0])
                    < (best[1] - best[0], best[0]))):
            best_sum = running_sum
            best = candidate
    return best


def contrast(values: np.ndarray, lo: int, hi: int) -> float:
    inside = float(np.mean(values[lo:hi]))
    outside_n = len(values) - (hi - lo)
    if outside_n == 0:
        return 0.0
    outside = float((values.sum() - values[lo:hi].sum()) / outside_n)
    return inside - outside


def heldout_test(values: np.ndarray, lo: int, hi: int) -> tuple[float, float, float]:
    observed = contrast(values, lo, hi)
    if np.ptp(values) <= 1e-12 or hi - lo == len(values):
        return observed, 1.0, 0.5
    width = hi - lo
    # Every rolled inside sum is one circular window sum of fixed width.  The
    # vectorized construction is exactly equivalent to enumerating all rolls.
    circular = np.r_[values, values[: width - 1]] if width > 1 else values
    prefix = np.r_[0.0, np.cumsum(circular)]
    inside_sums = prefix[width: width + len(values)] - prefix[: len(values)]
    outside_n = len(values) - width
    orbit = inside_sums / width - (values.sum() - inside_sums) / outside_n
    # Complete orbit includes identity; inclusive ties are conservative.
    tolerance = 32.0 * np.finfo(float).eps * max(
        1.0, abs(observed), float(np.max(np.abs(orbit))))
    pvalue = float(np.mean(orbit >= observed - tolerance))
    pvalue = max(pvalue, 1.0 / len(values))
    return observed, pvalue, 1.0 / (2.0 * np.sqrt(pvalue))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", type=Path, required=True)
    ap.add_argument("--prediction-method", required=True)
    ap.add_argument("--fields", type=Path, required=True)
    ap.add_argument("--fields-method", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    predictions = load(args.prediction, args.prediction_method)
    fields = load(args.fields, args.fields_method)
    audit = {"n": 0, "constant_heldout": 0, "invalid_intervals": 0}
    with args.out.open("w") as handle:
        for key in sorted(set(predictions) & set(fields)):
            row = predictions[key]
            scores = np.asarray(row["score_curve"], dtype=float)
            n = min(len(scores), max(1, int(np.floor(float(row["duration"]) * 4.0))))
            scores = scores[:n]
            evidence = fields[key]["modality_evidence"]["main_effects"]
            modality = {name: midrank(np.asarray(evidence[name], dtype=float)[:n])
                        for name in MAIN}
            witnesses = []
            for heldout in MAIN:
                selectors = [name for name in MAIN if name != heldout]
                selector_signal = np.mean(
                    np.stack([modality[name] for name in selectors]), axis=0)
                selector_signal = selector_signal - np.median(selector_signal)
                lo, hi = max_positive_excursion(selector_signal)
                statistic, pvalue, evalue = heldout_test(modality[heldout], lo, hi)
                audit["constant_heldout"] += int(np.ptp(modality[heldout]) <= 1e-12)
                witnesses.append({
                    "heldout_modality": heldout,
                    "selector_modalities": selectors,
                    "lo": int(lo), "hi": int(hi),
                    "heldout_contrast": statistic,
                    "conditional_orbit_pvalue": pvalue,
                    "conditional_evalue": evalue,
                })
            lo = int(np.median([w["lo"] for w in witnesses]))
            hi = int(np.median([w["hi"] for w in witnesses]))
            if hi <= lo:
                audit["invalid_intervals"] += 1
                hi = min(n, lo + 1)
            rate = float(row.get("native_rate", 4.0) or 4.0)
            aggregate_e = float(np.mean([w["conditional_evalue"] for w in witnesses]))
            output = dict(row)
            output["method"] = "crossfitted_interval_witness_v1"
            output["score_curve"] = scores.tolist()
            output["intervals"] = [[lo / rate,
                                    min(float(row["duration"]), hi / rate),
                                    aggregate_e]]
            output["modality_evidence"] = {
                **output.get("modality_evidence", {}),
                "crossfitted_interval_witnesses": witnesses,
                "crossfitted_mean_evalue": aggregate_e,
            }
            output["raw"] = {
                **output.get("raw", {}),
                "module": "crossfitted_interval_witness",
                "selector": "heldout_free_pair_mean_max_positive_excursion",
                "selector_center": "within_video_median",
                "prediction_readout": "coordinatewise_median_of_three_fold_spans",
                "certificate": "complete_heldout_modality_cyclic_orbit_inclusive_upper_tail",
                "certificate_scope": "each_fold_specific_selected_interval_under_its_heldout_relative_phase_null",
                "aggregate_e_rule": "arithmetic_mean_of_three_valid_evalues",
                "score_threshold": None,
                "duration_tuning_parameter": None,
                "numeric_weights": 0,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
                "gt_access": False,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
            audit["n"] += 1
    args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
