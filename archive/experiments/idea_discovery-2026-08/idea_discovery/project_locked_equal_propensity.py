#!/usr/bin/env python3
"""Frozen equal-evidence video propensity for future confirmation cohorts.

The rule was chosen during development on the 611-video cohort.  This script
emits no alternatives and records that provenance explicitly; it is suitable
for a future untouched confirmation run, not for retroactively declaring the
development cohort held out.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_pos_less import load_chunks
from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    rows = load(args.base, args.base_method)
    chunks = load_chunks()
    with args.out.open("w") as handle:
        for key, row in sorted(rows.items()):
            p = np.clip(np.asarray(row["score_curve"], float), 1e-6, 1 - 1e-6)
            z = np.log(p / (1 - p))
            base_propensity = float(z.mean())
            values = [float(item.get("z_masked", item.get("z_isolated", -20)))
                      for item in chunks.get(key, [])]
            values = [value for value in values if math.isfinite(value)]
            language = max(values) if values else base_propensity
            dense_peak = float(np.max(z))
            propensity = (base_propensity + language + dense_peak) / 3
            output = dict(row)
            output["method"] = "locked_equal_evidence_propensity_v3"
            output["score_curve"] = sigmoid(propensity + (z - z.mean())).tolist()
            output["raw"] = {
                **output.get("raw", {}),
                "gt_access": False,
                "module": "locked_equal_evidence_video_propensity",
                "rule": "equal_logit_mean_of_base_masked_language_and_dense_peak",
                "rule_weights": [1 / 3, 1 / 3, 1 / 3],
                "development_selected_rule": True,
                "development_cohort": "four_dataset_611_video_seen_cohort",
                "eligible_evidence_status": "exploratory_until_untouched_confirmation",
                "label_selected_rule": True,
                "empirical_marginal_preserved_exactly": False,
                "centered_logit_residual_multiset_preserved": True,
                "within_video_order_preserved": True,
                "dataset_parameters": 0,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
