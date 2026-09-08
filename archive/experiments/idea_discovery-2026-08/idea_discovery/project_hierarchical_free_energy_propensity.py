#!/usr/bin/env python3
"""Hierarchical existential free energy for label-free video propensity.

Localization asks whether at least one temporal unit contains hateful evidence.
For each video, log-mean-exp first pools candidates within the dense multimodal
timeline and within timestamped MLLM transcript chunks.  A second log-mean-exp
pools the two evidence reservoirs with equal reservoir mass, preventing the
much denser frame stream from dominating merely by having more samples.
The centered localization residual is preserved exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid


def logmeanexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("logmeanexp requires at least one value")
    maximum = float(np.max(values))
    return maximum + float(np.log(np.mean(np.exp(np.clip(values - maximum, -60, 0)))))


def load_chunks(path: Path) -> dict[tuple[str, str], list[float]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    with path.open() as handle:
        for row in map(json.loads, handle):
            value = float(row["log_odds"])
            if np.isfinite(value):
                grouped.setdefault((row["dataset"], str(row["video_id"])), []).append(value)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-method", required=True)
    parser.add_argument("--propensity-source", type=Path)
    parser.add_argument("--propensity-method")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    rows = load(args.base, args.base_method)
    if bool(args.propensity_source) != bool(args.propensity_method):
        raise ValueError("--propensity-source and --propensity-method must be supplied together")
    propensity_rows = (load(args.propensity_source, args.propensity_method)
                       if args.propensity_source else rows)
    chunks = load_chunks(args.chunks)
    missing_chunks = 0
    with args.out.open("w") as handle:
        for key, row in sorted(rows.items()):
            if key not in propensity_rows:
                continue
            p = np.clip(np.asarray(row["score_curve"], dtype=float), 1e-6, 1 - 1e-6)
            z = np.log(p / (1 - p))
            source_p = np.clip(np.asarray(propensity_rows[key]["score_curve"], dtype=float),
                               1e-6, 1 - 1e-6)
            source_z = np.log(source_p / (1 - source_p))
            dense_free_energy = logmeanexp(source_z)
            chunk_values = np.asarray(chunks.get(key, []), dtype=float)
            if len(chunk_values):
                chunk_free_energy = logmeanexp(chunk_values)
                reservoirs = np.asarray([dense_free_energy, chunk_free_energy])
            else:
                # A missing reservoir receives no synthetic vote; this is the
                # unique available-reservoir reduction, not a tuned fallback.
                missing_chunks += 1
                chunk_free_energy = None
                reservoirs = np.asarray([dense_free_energy])
            propensity = logmeanexp(reservoirs)
            output = dict(row)
            output["method"] = "hierarchical_existential_free_energy_v1"
            output["score_curve"] = sigmoid(propensity + (z - z.mean())).tolist()
            output["raw"] = {
                **output.get("raw", {}),
                "gt_access": False,
                "module": "hierarchical_existential_free_energy",
                "within_reservoir_pool": "logmeanexp",
                "across_reservoir_pool": "equal_mass_logmeanexp",
                "reservoirs": ["dense_multimodal_timeline", "timestamped_mllm_chunks"],
                "dense_propensity_source_method": (args.propensity_method
                                                    or args.base_method),
                "temperature": "native_logit_unit_no_free_parameter",
                "dense_free_energy": dense_free_energy,
                "chunk_free_energy": chunk_free_energy,
                "video_propensity": propensity,
                # The centered residual is retained, but replacing its offset
                # by the free-energy propensity changes the video's logit mean.
                "video_mean_preserved": False,
                "centered_logit_residual_preserved": True,
                "development_selected_design": True,
                "eligible_evidence_status": "exploratory_until_untouched_confirmation",
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
    audit = {"n": len(rows), "missing_chunk_reservoir": missing_chunks}
    args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
