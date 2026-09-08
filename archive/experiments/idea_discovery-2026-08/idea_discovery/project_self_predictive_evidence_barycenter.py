#!/usr/bin/env python3
"""Sample-adaptive multimodal evidence barycenter.

Each modality receives authority only to the extent that its aligned temporal
field is predictable from the other two modalities.  Authority is estimated
inside the current video, without labels, dataset statistics, thresholds, or
numeric fusion weights.  The output preserves the nominal video's logit mean
and transports its empirical temporal marginal to the consensus ordering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


NAMES = ("visual", "language", "audio")


def load(path: Path, method: str) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row
            for row in map(json.loads, path.open())
            if row.get("method") == method}


def midrank(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(len(values), dtype=float)
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0) - 0.5


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first, second = first - first.mean(), second - second.mean()
    scale = float(np.linalg.norm(first) * np.linalg.norm(second))
    return 0.0 if scale <= np.finfo(float).eps else float(first @ second / scale)


def exact_transport(reference: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    """Assign sorted reference values according to consensus order."""
    order = np.lexsort((np.arange(len(coordinate)), coordinate))
    output = np.empty(len(reference), dtype=float)
    output[order] = np.sort(reference, kind="mergesort")
    return output


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


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
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    audit = {"eligible": 0, "fallback": 0, "authority_sums": []}

    with args.out.open("w") as handle:
        for key in sorted(set(fields) & set(nominal)):
            row, evidence = nominal[key], fields[key]["modality_evidence"]
            streams = [np.asarray(evidence["main_effects"][name], float)
                       for name in NAMES]
            n = min(len(row["score_curve"]), *(len(value) for value in streams))
            if n < 2:
                continue
            streams = [midrank(value[:n]) for value in streams]
            authorities = []
            for index, stream in enumerate(streams):
                peers = [value for j, value in enumerate(streams) if j != index]
                predicted = midrank(np.mean(np.stack(peers), axis=0))
                authorities.append(max(0.0, correlation(stream, predicted)))
            total = float(sum(authorities))
            reference_probability = np.clip(
                np.asarray(row["score_curve"], float)[:n], 1e-6, 1 - 1e-6)
            reference = np.log(reference_probability / (1 - reference_probability))
            nominal_coordinate = midrank(reference)
            if total <= np.finfo(float).eps:
                consensus = nominal_coordinate
                audit["fallback"] += 1
            else:
                evidence_coordinate = sum(weight * stream for weight, stream
                                          in zip(authorities, streams)) / total
                # The frozen semantic posterior and the independently
                # reconstructed evidence field each receive one unit of mass.
                # This is a symmetry constraint, not a tuned coefficient.
                consensus = (nominal_coordinate + evidence_coordinate) / 2.0
                audit["eligible"] += 1
            transported = exact_transport(reference - reference.mean(), consensus)
            score = sigmoid(float(reference.mean()) + transported)
            output = dict(row)
            output["method"] = "anchored_self_predictive_evidence_barycenter_v2"
            output["score_curve"] = score.tolist()
            output["intervals"] = []
            output["modality_evidence"] = {
                **output.get("modality_evidence", {}),
                "self_predictive_authority": dict(zip(NAMES, authorities)),
                "authority_sum": total,
            }
            output["raw"] = {
                **output.get("raw", {}), "gt_access": False,
                "module": "sample_adaptive_self_predictive_evidence_barycenter",
                "authority": "positive_aligned_predictability_from_other_modalities",
                "fusion": "equal_mass_nominal_and_authority_normalized_evidence_barycenter",
                "fallback": "nominal_order_if_no_positive_predictability",
                "nominal_logit_mean_preserved": True,
                "nominal_logit_multiset_preserved": True,
                "dataset_parameters": 0, "label_selected_parameters": 0,
                "numeric_fusion_weights": 0, "code_sha256": code_hash,
            }
            audit["authority_sums"].append(total)
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")

    summary = {key: value for key, value in audit.items() if key != "authority_sums"}
    values = audit["authority_sums"]
    summary["authority_sum_mean"] = float(np.mean(values)) if values else None
    summary["authority_sum_median"] = float(np.median(values)) if values else None
    args.out.with_suffix(".audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
