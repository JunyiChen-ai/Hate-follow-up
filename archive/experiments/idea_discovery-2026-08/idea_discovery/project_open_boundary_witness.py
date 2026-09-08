#!/usr/bin/env python3
"""Open only those visual proposal boundaries warranted by held-out modalities.

A visual-only proposal fixes two candidate endpoints.  Language and audio are
therefore honest held-out witnesses for each endpoint.  Each witness compares
the proposal interior with the adjacent exterior and is calibrated over its
complete cyclic orbit.  An endpoint is retained only when the arithmetic mean
of the two valid e-values exceeds the betting break-even value one; otherwise
that side remains open at the video edge.

The rule has no dataset parameter, label-selected threshold, duration cutoff,
or temporal window size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_role_orthogonal_transport import load
from scripts.idea_discovery.project_shapley_boundary import midrank


HELDOUT = ("language", "audio")


def endpoint_contrast(values: np.ndarray, lo: int, hi: int, side: str) -> float:
    """Interior-minus-adjacent-exterior contrast for one proposed endpoint."""
    inside = float(np.mean(values[lo:hi]))
    exterior = values[:lo] if side == "left" else values[hi:]
    if len(exterior) == 0:
        return 0.0
    return inside - float(np.mean(exterior))


def complete_orbit_test(values: np.ndarray, lo: int, hi: int,
                        side: str) -> tuple[float, float, float]:
    """Inclusive upper-tail p-value over every cyclic shift, identity included."""
    observed = endpoint_contrast(values, lo, hi, side)
    if np.ptp(values) <= 1e-12:
        return observed, 1.0, 0.5
    orbit = np.asarray([
        endpoint_contrast(np.roll(values, shift), lo, hi, side)
        for shift in range(len(values))
    ], dtype=float)
    tolerance = 32.0 * np.finfo(float).eps * max(
        1.0, abs(observed), float(np.max(np.abs(orbit))))
    pvalue = float(np.mean(orbit >= observed - tolerance))
    pvalue = max(pvalue, 1.0 / len(values))
    # Shafer-style p-to-e calibrator: valid for every super-uniform p-value.
    evalue = 1.0 / (2.0 * np.sqrt(pvalue))
    return observed, pvalue, evalue


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction", type=Path, required=True)
    ap.add_argument("--prediction-method", required=True)
    ap.add_argument("--visual-proposals", type=Path, required=True)
    ap.add_argument("--visual-method", required=True)
    ap.add_argument("--fields", type=Path, required=True)
    ap.add_argument("--fields-method", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    predictions = load(args.prediction, args.prediction_method)
    proposals = load(args.visual_proposals, args.visual_method)
    fields = load(args.fields, args.fields_method)
    keys = sorted(set(predictions) & set(proposals) & set(fields))
    audit = {
        "n": 0, "left_retained": 0, "right_retained": 0,
        "both_open": 0, "both_retained": 0, "invalid_visual_proposal": 0,
    }
    with args.out.open("w") as handle:
        for key in keys:
            source = predictions[key]
            duration = float(source["duration"])
            rate = float(source.get("native_rate", 4.0) or 4.0)
            n = min(len(source["score_curve"]),
                    max(1, int(np.floor(duration * rate))))
            score = np.asarray(source["score_curve"], dtype=float)[:n]
            candidate = proposals[key].get("intervals", [])
            if not candidate:
                audit["invalid_visual_proposal"] += 1
                lo, hi = 0, n
            else:
                lo = int(np.clip(np.round(float(candidate[0][0]) * rate), 0, n - 1))
                hi = int(np.clip(np.round(float(candidate[0][1]) * rate), lo + 1, n))

            evidence = fields[key]["modality_evidence"]["main_effects"]
            modality = {
                name: midrank(np.asarray(evidence[name], dtype=float)[:n])
                for name in HELDOUT
            }
            endpoints = {}
            retained = {}
            for side in ("left", "right"):
                at_edge = (lo == 0) if side == "left" else (hi == n)
                witnesses = []
                for name in HELDOUT:
                    statistic, pvalue, evalue = complete_orbit_test(
                        modality[name], lo, hi, side)
                    witnesses.append({
                        "heldout_modality": name,
                        "contrast": statistic,
                        "conditional_orbit_pvalue": pvalue,
                        "conditional_evalue": evalue,
                    })
                mean_e = float(np.mean([x["conditional_evalue"] for x in witnesses]))
                keep = bool(at_edge or mean_e > 1.0)
                retained[side] = keep
                endpoints[side] = {
                    "candidate_index": int(lo if side == "left" else hi),
                    "already_at_video_edge": at_edge,
                    "heldout_witnesses": witnesses,
                    "mean_evalue": mean_e,
                    "retained": keep,
                }

            pred_lo = lo if retained["left"] else 0
            pred_hi = hi if retained["right"] else n
            audit["left_retained"] += int(retained["left"])
            audit["right_retained"] += int(retained["right"])
            audit["both_open"] += int(not retained["left"] and not retained["right"])
            audit["both_retained"] += int(retained["left"] and retained["right"])

            output = dict(source)
            output["method"] = "open_boundary_crossmodal_witness_v1"
            output["score_curve"] = score.tolist()
            output["intervals"] = [[
                pred_lo / rate,
                min(duration, pred_hi / rate),
                float(np.mean([endpoints[x]["mean_evalue"] for x in endpoints])),
            ]]
            output["modality_evidence"] = {
                **output.get("modality_evidence", {}),
                "open_boundary_witnesses": endpoints,
            }
            output["raw"] = {
                **output.get("raw", {}),
                "module": "open_boundary_crossmodal_witness",
                "candidate_source": "visual_only_interval",
                "candidate_alignment": "seconds_to_native_grid_by_round_then_clamp",
                "endpoint_statistic": "proposal_interior_minus_adjacent_side_exterior",
                "calibration": "complete_cyclic_orbit_identity_included_inclusive_upper_tail",
                "heldout_modalities": list(HELDOUT),
                "endpoint_combiner": "arithmetic_mean_of_valid_evalues",
                "endpoint_decision": "retain_iff_mean_evalue_gt_betting_break_even_one",
                "uncertified_boundary_action": "open_to_corresponding_video_edge",
                "numeric_weights": 0,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
                "duration_tuning_parameter": None,
                "gt_access": False,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
            audit["n"] += 1

    args.out.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
