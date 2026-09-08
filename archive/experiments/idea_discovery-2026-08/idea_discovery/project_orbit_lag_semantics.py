#!/usr/bin/env python3
"""Per-video orbit-calibrated visual-language lag correction.

Transcript semantics are circularly aligned to the dense visual evidence.
The admission weight is the information concentration of the entire orbit
correlation distribution.  A reversed-transcript branch is emitted as the
mechanism falsification control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_orbit_authorized_semantics import chunks, text_field
from scripts.idea_discovery.project_role_orthogonal_transport import load, sigmoid


METHODS = ("orbit_lag_base_v1", "orbit_lag_aligned_v1",
           "orbit_lag_reverse_v1", "orbit_lag_timestamp_v1")


def robust(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, float) - np.mean(values)
    scale = np.median(np.abs(x - np.median(x)))
    return x / scale if scale > np.finfo(float).eps else np.zeros_like(x)


def orbit_align(reference: np.ndarray, evidence: np.ndarray) -> tuple[np.ndarray, float, dict]:
    r, e = robust(reference), robust(evidence)
    if not np.any(r) or not np.any(e):
        return evidence.copy(), 0.0, {"fallback": True}
    r /= np.linalg.norm(r); e /= np.linalg.norm(e)
    # correlation[k] = sum_i r[i] e[i+k], hence roll(e, -k) aligns that lag.
    correlation = np.fft.ifft(np.conj(np.fft.fft(r)) * np.fft.fft(e)).real
    shift = int(np.argmax(correlation))
    aligned = np.roll(evidence, -shift)
    standardized = robust(correlation)
    standardized -= np.max(standardized)
    probability = np.exp(np.clip(standardized, -700.0, 0.0))
    probability /= np.sum(probability)
    entropy = -float(np.sum(probability * np.log(np.maximum(probability, np.finfo(float).tiny))))
    max_entropy = float(np.log(len(probability)))
    weight = max(0.0, min(1.0, 1.0 - entropy / max_entropy)) if max_entropy > 0 else 0.0
    direct = float(np.dot(r, robust(aligned) / max(np.linalg.norm(robust(aligned)), 1e-12)))
    return aligned, weight, {"fallback": False, "lag_bins": shift,
                             "orbit_information_weight": weight,
                             "best_correlation": float(correlation[shift]),
                             "aligned_direct_correlation": direct,
                             "orbit_entropy": entropy}


def fuse(base: np.ndarray, evidence: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0:
        return base.copy()
    clipped = np.clip(base, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    correction = robust(evidence)
    scale = np.median(np.abs(logits - np.median(logits)))
    changed = logits + weight * scale * correction
    target = float(np.mean(base)); lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if float(sigmoid(changed + mid).mean()) < target: lo = mid
        else: hi = mid
    return sigmoid(changed + (lo + hi) / 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--base-method", required=True)
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): raise RuntimeError(f"refusing existing output: {args.out}")
    bases, semantic = load(args.base, args.base_method), chunks(args.chunks)
    with args.out.open("w") as handle:
        for key, row in sorted(bases.items()):
            base = np.asarray(row["score_curve"], float)
            text = text_field(semantic.get(key, []), float(row["duration"]), len(base))
            variants = {m: base.copy() for m in METHODS}; audit = {}
            if text is not None:
                base_logit = np.log(np.clip(base, 1e-6, 1-1e-6) /
                                    np.clip(1-base, 1e-6, 1-1e-6))
                aligned, weight, factual = orbit_align(base_logit, text)
                reverse, reverse_weight, reversed_audit = orbit_align(base_logit, text[::-1])
                variants["orbit_lag_aligned_v1"] = fuse(base, aligned, weight)
                variants["orbit_lag_reverse_v1"] = fuse(base, reverse, reverse_weight)
                variants["orbit_lag_timestamp_v1"] = fuse(base, text, weight)
                audit = {"factual": factual, "reverse": reversed_audit}
            for method, scores in variants.items():
                output = dict(row); output["method"] = method
                output["score_curve"] = scores.tolist()
                output["raw"] = {**output.get("raw", {}), "gt_access": False,
                                 "module": "orbit_calibrated_crossmodal_lag",
                                 "dataset_parameters": 0,
                                 "label_selected_parameters": 0,
                                 "video_mean_preserved": True,
                                 "intervals_preserved": True,
                                 "orbit_lag": audit}
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__": main()
