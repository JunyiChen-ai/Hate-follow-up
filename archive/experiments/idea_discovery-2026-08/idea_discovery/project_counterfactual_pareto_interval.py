#!/usr/bin/env python3
"""Counterfactual Pareto interval localization over a joint orbit certificate.

Candidate endpoints are the intrinsic change points of direct modality evidence
and timestamped semantic chunks.  Every candidate is evaluated by four
within-video criteria: direct multimodal contrast, concentration of the joint
orbit certificate, free-energy sufficiency when only the interval is retained,
and free-energy necessity measured by deleting it.  Criterion midranks are
combined by their minimum, so one modality/criterion cannot compensate for a
failed one.  No target label, threshold, duration prior, or numeric weight is
used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from scripts.idea_discovery.project_hierarchical_free_energy_propensity import logmeanexp
from scripts.idea_discovery.project_multiscale_scan_boundary import MAIN
from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid
from scripts.idea_discovery.project_shapley_boundary import midrank


def load_chunks(path: Path) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    with path.open() as handle:
        for row in map(json.loads, handle):
            grouped.setdefault((row["dataset"], str(row["video_id"])), []).append(row)
    return grouped


def reservoir_free_energy(dense: np.ndarray, chunks: np.ndarray) -> float:
    reservoirs = []
    if len(dense):
        reservoirs.append(logmeanexp(dense))
    if len(chunks):
        reservoirs.append(logmeanexp(chunks))
    if not reservoirs:
        return -np.inf
    return logmeanexp(np.asarray(reservoirs))


def criterion_midrank(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.ones(len(values), dtype=float)
    return (rankdata(values, method="average") - 1) / (len(values) - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--final-method", required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--fields-method", required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    final = load(args.final, args.final_method)
    fields = load(args.fields, args.fields_method)
    chunks = load_chunks(args.chunks)
    audit = {"n": 0, "candidates": 0, "fallback_full": 0}
    with args.out.open("w") as handle:
        for key in sorted(set(final) & set(fields)):
            audit["n"] += 1
            row = final[key]
            raw_scores = np.asarray(row["score_curve"], dtype=float)
            contract_n = max(1, int(np.floor(float(row["duration"]) * 4.0)))
            raw_scores = raw_scores[:contract_n]
            p = np.clip(raw_scores, 1e-6, 1 - 1e-6)
            z = np.log(p / (1 - p))
            n = len(z)
            evidence = fields[key]["modality_evidence"]
            direct_coordinates = [midrank(np.asarray(
                evidence["main_effects"][name], dtype=float)[:n]) for name in MAIN]
            direct = np.mean(np.stack(direct_coordinates), axis=0)
            stored_p = row.get("modality_evidence", {}).get("joint_adjusted_p")
            certificate = (-np.log(np.clip(np.asarray(stored_p, dtype=float)[:n], 1e-12, 1.0))
                           if stored_p is not None else np.zeros(n, dtype=float))
            frame_adjusted_p = (np.asarray(stored_p, dtype=float)[:n]
                                if stored_p is not None else np.ones(n, dtype=float))

            endpoints = {0, n}
            for coordinate in direct_coordinates:
                endpoints.update((np.flatnonzero(np.diff(coordinate) != 0) + 1).tolist())
            local_chunks = chunks.get(key, [])
            chunk_centers, chunk_values = [], []
            rate = float(row.get("native_rate", 4.0) or 4.0)
            for chunk in local_chunks:
                start = float(chunk["start"]); end = float(chunk["end"])
                endpoints.add(max(0, min(n, int(np.floor(start * rate)))))
                endpoints.add(max(0, min(n, int(np.ceil(end * rate)))))
                chunk_centers.append(0.5 * (start + end) * rate)
                chunk_values.append(float(chunk["log_odds"]))
            endpoints = sorted(endpoints)
            candidates = [(lo, hi) for i, lo in enumerate(endpoints)
                          for hi in endpoints[i + 1:] if not (lo == 0 and hi == n)]
            audit["candidates"] += len(candidates)
            if not candidates:
                audit["fallback_full"] += 1
                selected = (0, n); bottleneck = 1.0
                selected_criteria = [1.0] * 4
            else:
                chunk_centers_array = np.asarray(chunk_centers, dtype=float)
                chunk_values_array = np.asarray(chunk_values, dtype=float)
                full_free_energy = reservoir_free_energy(z, chunk_values_array)
                criteria = []
                for lo, hi in candidates:
                    inside_count = hi - lo
                    outside_count = n - inside_count
                    direct_inside = float(np.mean(direct[lo:hi]))
                    direct_outside = float((direct.sum() - direct[lo:hi].sum()) /
                                           outside_count) if outside_count else direct_inside
                    direct_contrast = direct_inside - direct_outside
                    certificate_inside = float(np.mean(certificate[lo:hi]))
                    certificate_outside = float((certificate.sum() - certificate[lo:hi].sum()) /
                                                outside_count) if outside_count else 0.0
                    certificate_contrast = certificate_inside - certificate_outside
                    chunk_inside = ((chunk_centers_array >= lo) & (chunk_centers_array < hi)
                                    if len(chunk_centers_array) else np.zeros(0, dtype=bool))
                    sufficiency = reservoir_free_energy(z[lo:hi],
                                                        chunk_values_array[chunk_inside])
                    outside_dense = np.r_[z[:lo], z[hi:]]
                    outside_chunks = chunk_values_array[~chunk_inside]
                    outside_free_energy = reservoir_free_energy(outside_dense, outside_chunks)
                    necessity = (full_free_energy - outside_free_energy
                                 if np.isfinite(outside_free_energy) else np.inf)
                    criteria.append((direct_contrast, certificate_contrast,
                                     sufficiency, necessity))
                criterion_array = np.asarray(criteria, dtype=float)
                ranks = np.column_stack([criterion_midrank(criterion_array[:, index])
                                         for index in range(criterion_array.shape[1])])
                bottlenecks = np.min(ranks, axis=1)
                sums = np.sum(ranks, axis=1)
                lengths = np.asarray([hi - lo for lo, hi in candidates])
                # Lexicographic: noncompensatory bottleneck, then total support,
                # then the shortest equally warranted explanation.
                order = np.lexsort((lengths, -sums, -bottlenecks))
                chosen = int(order[0])
                selected = candidates[chosen]
                bottleneck = float(bottlenecks[chosen])
                selected_criteria = ranks[chosen].tolist()
            lo, hi = selected
            # Every local p was already adjusted by one global maximum over all
            # pairs and times. Therefore the minimum adjusted p inside any
            # adaptively selected subset is no smaller than the global adjusted
            # minimum and remains selection-safe. No interval/length prior is
            # needed. Its p-to-e image is carried as evidence, not a posterior.
            selection_safe_p = float(np.min(frame_adjusted_p[lo:hi]))
            selection_safe_e = 1.0 / (2.0 * np.sqrt(selection_safe_p))
            output = dict(row)
            output["method"] = "counterfactual_pareto_interval_certificate_v1"
            output["score_curve"] = raw_scores.tolist()
            # The public prediction contract is the floor-truncated 4 fps
            # timeline.  Certificate arrays must obey the same contract rather
            # than retaining a possible terminal ceil-grid sample.
            output["modality_evidence"] = {
                **output.get("modality_evidence", {}),
                "joint_adjusted_p": frame_adjusted_p.tolist(),
                "joint_evalue": (1.0 / (2.0 * np.sqrt(frame_adjusted_p))).tolist(),
            }
            output["intervals"] = [[lo / rate,
                                    min(float(row["duration"]), hi / rate),
                                    float(sigmoid(bottleneck))]]
            output["raw"] = {
                **output.get("raw", {}),
                "gt_access": False,
                "module": "counterfactual_pareto_interval_certificate",
                "candidate_endpoints": "intrinsic_modality_and_chunk_change_points",
                "criteria": ["direct_contrast", "joint_certificate_contrast",
                             "retention_sufficiency", "deletion_necessity"],
                "aggregation": "within_video_midrank_pareto_bottleneck",
                "selection_certificate": "global_maxT_adjusted_min_p_within_selected_span",
                "selection_safe_interval_pvalue": selection_safe_p,
                "selection_safe_interval_evalue": selection_safe_e,
                "grid_contract_applied_before_candidates": "floor_truncated_4fps",
                "numeric_weights": 0,
                "score_threshold": None,
                "duration_tuning_parameter": None,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
                "n_candidates": len(candidates),
                "selected_criterion_ranks": selected_criteria,
                "pareto_bottleneck": bottleneck,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
    args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
