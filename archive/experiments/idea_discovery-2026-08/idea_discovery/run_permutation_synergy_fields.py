#!/usr/bin/env python3
"""Per-video exact-orbit multimodal temporal synergy fields.

Each modality keeps its empirical values and autocorrelation.  Circularly
shifting one stream relative to another gives the complete per-video temporal
null.  The zero-lag rank supplies a sample-adaptive authority; no target label,
dataset parameter, alpha level, or hand-set temporal window is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from scripts.idea_discovery.run_less import collect
from scripts.label_free_adapt.schema import Prediction, append_jsonl


NAMES = ("visual", "language", "audio")


def support(values: np.ndarray) -> np.ndarray:
    """Map ordinal evidence to [0,1] with average ranks and exact ties."""
    if len(values) <= 1:
        return np.full(len(values), 0.5, dtype=float)
    return (rankdata(values, method="average") - 1) / (len(values) - 1)


def orbit_field(streams: list[np.ndarray], multipliers: list[int]) -> tuple[np.ndarray, float, float]:
    """Return pointwise conjunction, zero-lag orbit rank, and null mean."""
    n = len(streams[0])
    observed_field = np.prod(np.stack(streams), axis=0)
    observed = float(observed_field.mean())
    if n <= 1:
        return observed_field, 0.5, observed
    null = np.empty(n, dtype=float)
    for shift in range(n):
        shifted = [streams[0]]
        shifted.extend(np.roll(stream, (multiplier * shift) % n)
                       for stream, multiplier in zip(streams[1:], multipliers))
        null[shift] = float(np.prod(np.stack(shifted), axis=0).mean())
    authority = float(rankdata(null, method="average")[0] / n)
    return observed_field, authority, float(null.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a10", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--audio-mode", choices=("absolute", "within_video"),
                        default="within_video")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    args.visual_mode = "majority"
    args.text_shift = "aligned"
    data = collect(args)
    authority_values = {"visual_language": [], "visual_audio": [],
                        "language_audio": []}
    for key, row in sorted(data.items()):
        streams = [support(row["evidence"][:, index]) for index in range(3)]
        definitions = {
            "visual_language": ([streams[0], streams[1]], [1]),
            "visual_audio": ([streams[0], streams[2]], [1]),
            "language_audio": ([streams[1], streams[2]], [1]),
        }
        fields = {}
        authorities = {}
        null_means = {}
        for name, (members, multipliers) in definitions.items():
            field, authority, null_mean = orbit_field(members, multipliers)
            # Centering makes zero the video's own temporal-null expectation;
            # authority scales evidence without an arbitrary accept/reject alpha.
            fields[name] = (authority * (field - null_mean)).tolist()
            authorities[name] = authority
            null_means[name] = null_mean
            authority_values[name].append(authority)
        append_jsonl(args.out, Prediction(
            "permutation_calibrated_synergy_fields_v1", key[0], key[1],
            row["duration"],
            score_curve=np.mean(np.stack(streams), axis=0).tolist(),
            intervals=[], calls=0,
            modality_evidence={
                "main_effects": {name: (stream - stream.mean()).tolist()
                                 for name, stream in zip(NAMES, streams)},
                "synergy_fields": fields,
                "orbit_authority": authorities,
                "orbit_null_mean": null_means,
            },
            raw={
                "gt_access": False,
                "null": "complete_within_video_circular_orbit",
                "marginals_preserved": True,
                "autocorrelation_preserved": True,
                "interactions": "three_complete_pairwise_circular_orbits_only",
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
            },
        ))
    audit = {
        "n": len(data),
        "authority_mean": {name: float(np.mean(values))
                           for name, values in authority_values.items()},
        "authority_nonzero": {name: int(np.count_nonzero(values))
                              for name, values in authority_values.items()},
    }
    args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
