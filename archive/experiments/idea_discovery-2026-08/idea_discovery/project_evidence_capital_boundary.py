#!/usr/bin/env python3
"""Threshold-free evidence-capital transport for multimodal boundaries.

A visual proposal supplies candidate endpoints.  Language and audio test each
endpoint without participating in its selection.  Their arithmetic-mean
e-value is valid under arbitrary dependence.  Instead of thresholding that
evidence, its capital share e/(1+e) transports the open video edge toward the
candidate endpoint.  Thus weak evidence honestly leaves a wide interval while
strong cross-modal evidence makes the localization sharper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_open_boundary_witness import (
    HELDOUT, complete_orbit_test,
)
from scripts.idea_discovery.project_role_orthogonal_transport import load
from scripts.idea_discovery.project_shapley_boundary import midrank


def capital_share(evalue: float) -> float:
    """Unit-capital allocation implied by nonnegative evidence capital."""
    return float(evalue / (1.0 + evalue))


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
    audit = {"n": 0, "invalid_visual_proposal": 0, "duration_fractions": []}
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
            shares = {}
            for side in ("left", "right"):
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
                shares[side] = capital_share(mean_e)
                endpoints[side] = {
                    "candidate_index": int(lo if side == "left" else hi),
                    "heldout_witnesses": witnesses,
                    "mean_evalue": mean_e,
                    "capital_share": shares[side],
                }

            # Each endpoint is a convex transport from its open video edge to
            # the visual candidate.  No accept/reject threshold is introduced.
            pred_lo = shares["left"] * lo
            pred_hi = n - shares["right"] * (n - hi)
            pred_lo_seconds = float(pred_lo / rate)
            pred_hi_seconds = min(duration, float(pred_hi / rate))
            duration_fraction = ((pred_hi_seconds - pred_lo_seconds) / duration
                                 if duration > 0 else 0.0)
            audit["duration_fractions"].append(duration_fraction)

            output = dict(source)
            output["method"] = "evidence_capital_boundary_transport_v1"
            output["score_curve"] = score.tolist()
            output["intervals"] = [[
                pred_lo_seconds, pred_hi_seconds,
                float(np.mean([endpoints[x]["mean_evalue"] for x in endpoints])),
            ]]
            output["modality_evidence"] = {
                **output.get("modality_evidence", {}),
                "evidence_capital_boundaries": endpoints,
            }
            output["raw"] = {
                **output.get("raw", {}),
                "module": "threshold_free_evidence_capital_boundary_transport",
                "candidate_source": "visual_only_interval",
                "candidate_alignment": "seconds_to_native_grid_by_round_then_clamp",
                "endpoint_statistic": "proposal_interior_minus_adjacent_side_exterior",
                "calibration": "complete_cyclic_orbit_identity_included_inclusive_upper_tail",
                "heldout_modalities": list(HELDOUT),
                "endpoint_combiner": "arithmetic_mean_of_valid_evalues",
                "boundary_transport": "open_edge_plus_e_over_one_plus_e_fraction_to_candidate",
                "numeric_weights": 0,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
                "score_threshold": None,
                "duration_tuning_parameter": None,
                "gt_access": False,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
            audit["n"] += 1

    fractions = np.asarray(audit.pop("duration_fractions"), dtype=float)
    audit["duration_fraction_mean"] = float(np.mean(fractions))
    audit["duration_fraction_median"] = float(np.median(fractions))
    audit["duration_fraction_q10_q90"] = np.quantile(fractions, [0.1, 0.9]).tolist()
    args.out.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
