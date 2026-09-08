#!/usr/bin/env python3
"""Per-video transcript-to-visual prototype diffusion without labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def load(path: Path, method: str) -> dict[tuple[str, str], dict]:
    return {(r["dataset"], r["video_id"]): r for r in map(json.loads, path.open())
            if r.get("method") == method}


def chunks(path: Path) -> dict[tuple[str, str], list[dict]]:
    output = defaultdict(list)
    for row in map(json.loads, path.open()):
        output[(str(row["dataset"]), str(row["video_id"]))].append(row)
    return output


def midrank(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(len(values), dtype=float)
    return (rankdata(values, method="average") - 0.5) / len(values) - 0.5


def exact_transport(reference: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.arange(len(coordinate)), coordinate))
    output = np.empty(len(reference), dtype=float)
    output[order] = np.sort(reference, kind="mergesort")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--visual-method", required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    visual, transcript = load(args.visual, args.visual_method), chunks(args.chunks)
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    effective = fallback = 0
    with args.out.open("w") as handle:
        for key, row in sorted(visual.items()):
            feature_path = (args.feature_root / key[0] / "coca_vitL14_4fps" /
                            f"{key[1]}.npy")
            features = np.asarray(np.load(feature_path), dtype=float)
            features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
            n = min(len(features), len(row["score_curve"]))
            features = features[:n]
            raw = np.clip(np.asarray(row["score_curve"], float)[:n], 1e-6, 1 - 1e-6)
            raw_logit = np.log(raw / (1 - raw))
            times = (np.arange(n) + 0.5) / 4.0
            records = transcript.get(key, [])
            null = float(np.median([float(x.get("config", {}).get("null_log_odds", -12.0))
                                    for x in records])) if records else -12.0
            language = np.full(n, null, dtype=float)
            covered = np.zeros(n, dtype=bool)
            for record in records:
                mask = ((times >= float(record["start"])) &
                        (times < float(record["end"])))
                language[mask] = np.maximum(language[mask], float(record["log_odds"]))
                covered |= mask
            language_coordinate = midrank(language)
            if np.ptp(language_coordinate) <= 1e-12:
                consensus = midrank(raw_logit)
                prototype_norm = 0.0
                fallback += 1
            else:
                # A signed prototype is the least-squares cross-covariance
                # between frozen visual features and timestamped language rank.
                prototype = features.T @ language_coordinate
                prototype_norm = float(np.linalg.norm(prototype))
                if prototype_norm <= 1e-12:
                    consensus = midrank(raw_logit)
                    fallback += 1
                else:
                    projected = features @ (prototype / prototype_norm)
                    # Equal mass for the frozen semantic axis, its language-
                    # conditioned visual diffusion, and language itself.
                    consensus = np.mean(np.stack([
                        midrank(raw_logit), midrank(projected), language_coordinate]), axis=0)
                    effective += 1
            transported = exact_transport(raw_logit - raw_logit.mean(), consensus)
            score = 1.0 / (1.0 + np.exp(-np.clip(raw_logit.mean() + transported, -30, 30)))
            output = dict(row)
            output["method"] = "crossmodal_prototype_diffusion_v1"
            output["score_curve"] = score.tolist()
            output["intervals"] = []
            output["modality_evidence"] = {
                **output.get("modality_evidence", {}),
                "transcript_chunks": len(records), "covered_fraction": float(covered.mean()),
                "prototype_norm": prototype_norm,
            }
            output["raw"] = {**output.get("raw", {}), "gt_access": False,
                             "module": "per_video_transcript_conditioned_visual_prototype_diffusion",
                             "prototype": "visual_language_cross_covariance",
                             "fusion": "equal_mass_three_coordinate_rank_barycenter",
                             "single_visual_encoder": True, "dataset_parameters": 0,
                             "label_selected_parameters": 0, "numeric_fusion_weights": 0,
                             "empirical_visual_logit_multiset_preserved": True,
                             "code_sha256": code_hash}
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
    audit = {"n": len(visual), "effective": effective, "fallback": fallback}
    args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
