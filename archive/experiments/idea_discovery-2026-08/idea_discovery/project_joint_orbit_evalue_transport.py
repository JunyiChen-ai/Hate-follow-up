#!/usr/bin/env python3
"""Certificate-preserving transport from joint-orbit p-fields to logits.

The joint max reference already adjusts each local pair statistic over the
three-pair-by-time family.  Their minimum is therefore the family-adjusted
local certificate.  The parameter-free calibrator e(p)=1/(2*sqrt(p)) converts
a valid p-variable into an e-variable.  The uncentered log-e evidence multiplies
the nominal odds directly; it does not preserve the video-level logit mean.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid


MAIN = ("visual", "language", "audio")
PAIRS = (("visual", "language", "visual_language"),
         ("visual", "audio", "visual_audio"),
         ("language", "audio", "language_audio"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--fields-method", required=True)
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--nominal-method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    fields = load(args.fields, args.fields_method)
    nominal = load(args.nominal, args.nominal_method)
    all_null = 0
    with args.out.open("w") as handle:
        for key in sorted(set(fields) & set(nominal)):
            row = nominal[key]
            p = np.clip(np.asarray(row["score_curve"], dtype=float), 1e-6, 1 - 1e-6)
            z = np.log(p / (1 - p))
            evidence = fields[key]["modality_evidence"]
            main = {name: np.asarray(evidence["main_effects"][name], dtype=float)
                    for name in MAIN}
            adjusted = []
            for first, second, pair in PAIRS:
                if np.ptp(main[first]) <= 1e-12 or np.ptp(main[second]) <= 1e-12:
                    continue
                # Stored witness is 0.5-p from an inclusive joint upper tail.
                adjusted.append(np.clip(0.5 - np.asarray(
                    evidence["synergy_fields"][pair], dtype=float), 1e-12, 1.0))
            if adjusted:
                joint_p = np.min(np.stack(adjusted), axis=0)
                evalue = 1.0 / (2.0 * np.sqrt(joint_p))
            else:
                all_null += 1
                joint_p = np.ones(len(z), dtype=float)
                evalue = np.ones(len(z), dtype=float)
            log_e = np.log(evalue)
            # Do not renormalize the e-variable: the transported odds are the
            # nominal odds multiplied by the valid joint-orbit e-evidence.
            transported = z + log_e
            output = dict(row)
            output["method"] = "joint_orbit_evalue_transport_v1"
            output["score_curve"] = sigmoid(transported).tolist()
            output["modality_evidence"] = {
                **output.get("modality_evidence", {}),
                "joint_adjusted_p": joint_p.tolist(),
                "joint_evalue": evalue.tolist(),
            }
            output["raw"] = {
                **output.get("raw", {}),
                "gt_access": False,
                "module": "joint_orbit_evalue_transport",
                "field_method": args.fields_method,
                "certificate_family": "all_three_pairs_by_all_temporal_positions",
                "p_to_e_calibrator": "one_over_two_sqrt_p",
                "odds_multiplier": "joint_evalue_without_renormalization",
                "video_logit_mean_preserved": False,
                "rank_rearrangement": False,
                "alpha_threshold": None,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps({"n": len(set(fields) & set(nominal)),
                      "all_pairs_null": all_null}, indent=2))


if __name__ == "__main__":
    main()
