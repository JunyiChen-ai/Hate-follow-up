#!/usr/bin/env python3
"""Boundary-safe local multiscale randomization fields.

For every video and modality pair, a zero-lag local conjunction is compared
with the distribution of *maximal* local conjunctions produced by every
admissible non-wrapping relative shift.  This gives simultaneous temporal
calibration rather than a single global synchrony weight.  Three scales are
derived from the sample length, and fused equally without labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from scripts.idea_discovery.run_less import collect
from scripts.idea_discovery.run_permutation_synergy_fields import NAMES, support
from scripts.label_free_adapt.schema import Prediction, append_jsonl


PAIRS = ((0, 1, "visual_language"), (0, 2, "visual_audio"),
         (1, 2, "language_audio"))


def moving_mean(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.astype(float, copy=True)
    sums = np.r_[0.0, np.cumsum(values, dtype=float)]
    return (sums[width:] - sums[:-width]) / width


def expand_centers(values: np.ndarray, width: int, length: int) -> np.ndarray:
    if len(values) == length:
        return values
    centers = np.arange(len(values), dtype=float) + (width - 1) / 2
    return np.interp(np.arange(length, dtype=float), centers, values,
                     left=float(values[0]), right=float(values[-1]))


def local_maxstat_field(first: np.ndarray, second: np.ndarray,
                        width: int, radius: int) -> np.ndarray:
    n = len(first)
    core_start, core_end = radius, n - radius
    core_first = first[core_start:core_end]
    observed = moving_mean(core_first * second[core_start:core_end], width)
    null_maxima = []
    # Yuan-Shou-style fixed central core: every lag sees the same samples and
    # exactly the same number of width-sized windows.
    for lag in range(-radius, radius + 1):
        if lag == 0:
            continue
        shifted_second = second[core_start + lag:core_end + lag]
        local = moving_mean(core_first * shifted_second, width)
        if len(local):
            null_maxima.append(float(np.max(local)))
    if not null_maxima:
        return np.zeros(n, dtype=float)
    null = np.sort(np.asarray(null_maxima, dtype=float))
    # Mid-CDF handles ties conservatively and yields a signed field around 0.
    lower = np.searchsorted(null, observed, side="left")
    upper = np.searchsorted(null, observed, side="right")
    calibrated = (0.5 * (lower + upper) + 0.5) / (len(null) + 1) - 0.5
    core_field = expand_centers(calibrated, width, len(core_first))
    output = np.zeros(n, dtype=float)
    output[core_start:core_end] = core_field
    return output


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
    scale_counts = []
    for key, row in sorted(data.items()):
        streams = [support(row["evidence"][:, index]) for index in range(3)]
        n = len(streams[0])
        # A length-derived geometric hierarchy: local, mesoscopic, contextual.
        scales = sorted(set((1, max(1, round(n ** 0.5)),
                             max(1, round(n ** (2 / 3))))))
        scales = [min(width, max(1, n // 2)) for width in scales]
        scales = sorted(set(scales))
        radius = max(scales)
        if n - 2 * radius < radius:
            radius = max(1, n // 3)
        scales = [min(width, max(1, n - 2 * radius)) for width in scales]
        scales = sorted(set(scales))
        scale_counts.append(len(scales))
        fields = {}
        per_scale = {}
        for first, second, name in PAIRS:
            values = [local_maxstat_field(streams[first], streams[second], width, radius)
                      for width in scales]
            fields[name] = np.mean(np.stack(values), axis=0).tolist()
            per_scale[name] = {str(width): value.tolist()
                               for width, value in zip(scales, values)}
        append_jsonl(args.out, Prediction(
            "local_multiscale_maxstat_synergy_fields_v1", key[0], key[1],
            row["duration"], score_curve=np.mean(np.stack(streams), axis=0).tolist(),
            intervals=[], calls=0,
            modality_evidence={
                "main_effects": {name: (stream - stream.mean()).tolist()
                                 for name, stream in zip(NAMES, streams)},
                "synergy_fields": fields,
                "per_scale_synergy_fields": per_scale,
                "sample_adaptive_scales": scales,
                "fixed_core_radius": radius,
            },
            raw={
                "gt_access": False,
                "null": "fixed_central_core_equal_window_nonwrapping_shifts",
                "temporal_calibration": "shiftwise_max_statistic_empirical_cdf",
                "scale_rule": "{1,round(n^1/2),round(n^2/3)}_equal_fusion",
                "wraparound": False,
                "equal_window_count_across_lags": True,
                "dataset_parameters": 0,
                "label_selected_parameters": 0,
            },
        ))
    audit = {"n": len(data), "scale_count_histogram": {
        str(count): scale_counts.count(count) for count in sorted(set(scale_counts))
    }}
    args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
