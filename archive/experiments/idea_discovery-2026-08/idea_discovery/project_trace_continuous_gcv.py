#!/usr/bin/env python3
"""Parameter-free continuous-GCV temporal proximal inference.

Unlike the earlier TRACE grid, this version has no hand-written lambda path.
Each video's regularization strength is obtained by continuous GCV over a
machine-precision interval derived solely from that video's chain spectrum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.optimize import minimize_scalar
from scipy.fft import dct, idct

from scripts.idea_discovery.project_role_orthogonal_transport import load
from scripts.idea_discovery.project_trace_gcv import laplacian_diagonals, reconstruct, smooth


def select(signal: np.ndarray) -> tuple[np.ndarray, float, dict]:
    n = len(signal)
    if n < 3 or not np.any(np.isfinite(signal)):
        return signal.copy(), 0.0, {"fallback": True}
    edges = np.ones(n - 1, dtype=np.float64)
    diagonal, off = laplacian_diagonals(edges)
    eigenvalues = eigvalsh_tridiagonal(diagonal, off, check_finite=False)
    coefficients = dct(signal, type=2, norm="ortho")
    positive = eigenvalues[eigenvalues > np.finfo(np.float64).eps]
    if not len(positive):
        return signal.copy(), 0.0, {"fallback": True}
    eps = np.finfo(np.float64).eps
    # Keep the banded system inside the numerically resolvable condition range.
    # Using epsilon itself can make 1 + lambda*L singular after rounding for
    # short or nearly constant traces.
    lower = np.log(np.sqrt(eps) / float(positive.max()))
    upper = np.log(1.0 / (np.sqrt(eps) * float(positive.min())))

    def objective(log_strength: float) -> float:
        strength = float(np.exp(log_strength))
        fitted = idct(coefficients / (1.0 + strength * eigenvalues),
                      type=2, norm="ortho")
        residual = signal - fitted
        trace = float(np.sum(1.0 / (1.0 + strength * eigenvalues)))
        denominator = max(eps, 1.0 - trace / n)
        return float(np.mean(residual * residual) / (denominator * denominator))

    result = minimize_scalar(
        objective,
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": float(np.sqrt(eps))},
    )
    strength = float(np.exp(result.x))
    fitted = idct(coefficients / (1.0 + strength * eigenvalues),
                  type=2, norm="ortho")
    return fitted, strength, {
        "fallback": False,
        "selected": strength,
        "gcv": float(result.fun),
        "optimizer_success": bool(result.success),
        "search_bounds_source": "video_chain_spectrum_and_machine_epsilon",
        "numeric_parameters_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    base = load(args.base, args.base_method)
    with args.out.open("w") as handle:
        for key, row in sorted(base.items()):
            scores = np.asarray(row["score_curve"], dtype=np.float64)
            logits = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1.0))
            signal = logits - logits.mean()
            fitted, strength, audit = select(signal)
            output = dict(row)
            output["method"] = "trace_continuous_gcv_v1"
            output["score_curve"] = reconstruct(scores, fitted).tolist()
            output["raw"] = {
                **output.get("raw", {}),
                "gt_access": False,
                "module": "continuous_gcv_temporal_proximal",
                "dataset_parameters": 0,
                "video_mean_preserved": True,
                "intervals_preserved": True,
                "trace": audit,
            }
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
