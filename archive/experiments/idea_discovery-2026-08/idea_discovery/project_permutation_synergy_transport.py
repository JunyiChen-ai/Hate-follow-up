#!/usr/bin/env python3
"""Exact dense rank transport from permutation-calibrated synergy fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid
from scripts.idea_discovery.project_shapley_boundary import midrank
from scripts.idea_discovery.project_shared_shapley_rank_transport import exact_transport


MAIN = ("visual", "language", "audio")
SYNERGY = ("visual_language", "visual_audio", "language_audio")


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
    with args.out.open("w") as handle:
        for key in sorted(set(fields) & set(nominal)):
            row = nominal[key]
            evidence = fields[key]["modality_evidence"]
            evidence_lengths = [len(evidence["main_effects"][name]) for name in MAIN]
            evidence_lengths += [len(evidence["synergy_fields"][name]) for name in SYNERGY]
            n = min(len(row["score_curve"]), *evidence_lengths)
            if n < 1:
                continue
            p = np.clip(np.asarray(row["score_curve"], float)[:n], 1e-6, 1 - 1e-6)
            z = np.log(p / (1 - p))
            reference = z - z.mean()
            main_fields = [midrank(np.asarray(evidence["main_effects"][name], float)[:n])
                           for name in MAIN]
            synergy_fields = [midrank(np.asarray(evidence["synergy_fields"][name], float)[:n])
                              for name in SYNERGY]
            variants = {
                "permutation_pair_complete_rank_transport_v3": main_fields + synergy_fields,
            }
            for method, coordinates in variants.items():
                consensus = np.mean(np.stack([midrank(reference), *coordinates]), axis=0)
                residual = exact_transport(reference, consensus, key)
                output = dict(row)
                output["method"] = method
                output["score_curve"] = sigmoid(float(z.mean()) + residual).tolist()
                output["raw"] = {
                    **output.get("raw", {}),
                    "gt_access": False,
                    "module": "permutation_calibrated_synergy_rank_transport",
                    "field_method": args.fields_method,
                    "transport": "exact_order_statistic_permutation",
                    "empirical_marginal_preserved_exactly": True,
                    "floating_tie_rule": "merge_abs_1e-12_or_rel_1e-10_before_midrank",
                    "alignment_rule": "common_prefix_minimum_across_nominal_and_all_evidence_fields",
                    "aligned_length": n,
                    "dataset_parameters": 0,
                    "label_selected_parameters": 0,
                }
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
