#!/usr/bin/env python3
"""REML-selected per-video random-walk diffusion.

The smoothing strength is the observation-noise / latent-random-walk ratio.
It is estimated independently for every video by profiling out the unknown
scale in the non-constant chain-Laplacian eigenspace.  Search bounds come only
from the video's spectrum and floating-point precision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.optimize import minimize_scalar

from scripts.idea_discovery.project_role_orthogonal_transport import load
from scripts.idea_discovery.project_trace_gcv import reconstruct, smooth


def select(signal: np.ndarray) -> tuple[np.ndarray, float, dict]:
    n = len(signal)
    if n < 4 or not np.all(np.isfinite(signal)):
        return signal.copy(), 0.0, {"fallback": True}
    eigenvalues = 2.0 - 2.0 * np.cos(np.pi * np.arange(1, n) / n)
    coefficients = dct(signal, type=2, norm="ortho")[1:]
    eps = np.finfo(np.float64).eps
    lower = np.log(eps / float(eigenvalues.max()))
    upper = np.log(1.0 / (eps * float(eigenvalues.min())))

    def objective(log_strength: float) -> float:
        strength = float(np.exp(log_strength))
        # Marginal covariance after profiling the latent RW scale:
        # variance_k is proportional to 1 + 1/(lambda * eigenvalue_k).
        variance = 1.0 + 1.0 / (strength * eigenvalues)
        scaled_energy = float(np.mean(coefficients * coefficients / variance))
        return float(np.sum(np.log(variance)) +
                     len(coefficients) * np.log(max(eps, scaled_energy)))

    result = minimize_scalar(objective, bounds=(lower, upper), method="bounded",
                             options={"xatol": float(np.sqrt(eps))})
    strength = float(np.exp(result.x))
    fitted = smooth(signal, np.ones(n - 1, dtype=np.float64), strength)
    return fitted, strength, {
        "fallback": False,
        "strength": strength,
        "profile_reml": float(result.fun),
        "optimizer_success": bool(result.success),
        "search_bounds_source": "video_chain_spectrum_and_machine_epsilon",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    rows = load(args.base, args.base_method)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for _, row in sorted(rows.items()):
            scores = np.asarray(row["score_curve"], dtype=np.float64)
            clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
            logits = np.log(clipped / (1.0 - clipped))
            fitted, strength, audit = select(logits - logits.mean())
            output = dict(row)
            output["method"] = "reml_random_walk_diffusion_v1"
            output["score_curve"] = reconstruct(scores, fitted).tolist()
            output["raw"] = {
                **output.get("raw", {}), "gt_access": False,
                "module": "per_video_reml_random_walk_diffusion",
                "dataset_parameters": 0, "label_selected_parameters": 0,
                "video_mean_preserved": True, "intervals_preserved": True,
                "reml": audit,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
