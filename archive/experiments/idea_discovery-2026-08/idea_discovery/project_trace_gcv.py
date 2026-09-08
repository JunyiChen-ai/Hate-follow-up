#!/usr/bin/env python3
"""TRACE: transcript-graph proximal regularization with label-free GCV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh_tridiagonal, solve_banded

from scripts.idea_discovery.project_role_orthogonal_transport import DATA_DIR, load, sigmoid


METHODS = (
    "trace_base_v1",
    "trace_aligned_gcv_v1",
    "trace_shifted_gcv_v1",
    "trace_uniform_gcv_v1",
)
LAMBDAS = (0.01, 0.04, 0.16, 0.64, 2.56)


def resize(values: np.ndarray, n: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == n:
        return values
    return np.interp(np.linspace(0, len(values) - 1, n), np.arange(len(values)), values)


def graph_edges(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    norms = np.linalg.norm(features, axis=1)
    normalized = features / np.maximum(norms[:, None], 1e-9)
    weights = np.maximum(0.0, np.sum(normalized[:-1] * normalized[1:], axis=1))
    weights[(norms[:-1] < 1e-8) | (norms[1:] < 1e-8)] = 0.0
    return weights


def laplacian_diagonals(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(edges) + 1
    diagonal = np.zeros(n, dtype=float)
    diagonal[:-1] += edges
    diagonal[1:] += edges
    return diagonal, -edges


def smooth(signal: np.ndarray, edges: np.ndarray, strength: float) -> np.ndarray:
    diagonal, off = laplacian_diagonals(edges)
    band = np.zeros((3, len(signal)), dtype=float)
    band[1] = 1.0 + strength * diagonal
    band[0, 1:] = strength * off
    band[2, :-1] = strength * off
    return solve_banded((1, 1), band, signal, check_finite=False)


def gcv_select(signal: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, float, dict]:
    if len(signal) < 3 or not np.any(edges > 1e-8):
        return signal.copy(), 0.0, {"selected": 0.0, "gcv": {}, "fallback": True}
    diagonal, off = laplacian_diagonals(edges)
    eigenvalues = eigvalsh_tridiagonal(diagonal, off, check_finite=False)
    records = {}
    candidates = []
    n = len(signal)
    for strength in LAMBDAS:
        fitted = smooth(signal, edges, strength)
        residual = signal - fitted
        trace_smoother = float(np.sum(1.0 / (1.0 + strength * eigenvalues)))
        denominator = max(1e-12, 1.0 - trace_smoother / n)
        score = float(np.mean(residual * residual) / (denominator * denominator))
        records[str(strength)] = score
        candidates.append((score, strength, fitted))
    score, strength, fitted = min(candidates, key=lambda item: (item[0], item[1]))
    return fitted, strength, {"selected": strength, "gcv": records, "fallback": False}


def reconstruct(base: np.ndarray, smoothed_residual: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(base, 1e-6, 1 - 1e-6) / np.clip(1 - base, 1e-6, 1))
    smoothed_residual = resize(smoothed_residual, len(base))
    smoothed_residual -= np.mean(smoothed_residual)
    changed = np.mean(logits) + smoothed_residual
    target = float(np.mean(base))
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if float(np.mean(sigmoid(changed + mid))) < target:
            lo = mid
        else:
            hi = mid
    return sigmoid(changed + (lo + hi) / 2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-method", required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--direct-text-root", action="store_true",
                        help="feature-root directly contains dataset subdirectories")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    rows = load(args.base, args.base_method)
    counts = {method: 0 for method in METHODS}
    selected = {method: {str(value): 0 for value in (0.0,) + LAMBDAS} for method in METHODS[1:]}
    max_mean_error = 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as output:
        for key, row in sorted(rows.items()):
            base = np.asarray(row["score_curve"], dtype=float)
            variants = {method: base.copy() for method in METHODS}
            audits = {}
            text_root = (args.feature_root if args.direct_text_root
                         else args.feature_root / "bert_sentence_1fps")
            path = text_root / DATA_DIR[key[0]] / f"{key[1]}.npy"
            if path.exists():
                features = np.load(path)
                n = len(features)
                logits = np.log(np.clip(base, 1e-6, 1 - 1e-6) / np.clip(1 - base, 1e-6, 1))
                signal = resize(logits, n)
                signal -= np.mean(signal)
                aligned_edges = graph_edges(features)
                shifted_edges = graph_edges(np.roll(features, max(1, n // 2), axis=0))
                uniform_edges = np.ones(max(0, n - 1), dtype=float)
                for method, edges in zip(METHODS[1:], (aligned_edges, shifted_edges, uniform_edges)):
                    fitted, strength, audit = gcv_select(signal, edges)
                    variants[method] = (base.copy() if strength == 0.0
                                        else reconstruct(base, fitted))
                    audits[method] = audit
                    selected[method][str(strength)] += 1
            for method in METHODS:
                scores = variants[method]
                counts[method] += int(not np.array_equal(scores, base))
                max_mean_error = max(max_mean_error,
                                     abs(float(np.mean(scores)) - float(np.mean(base))))
                result = dict(row)
                result["method"] = method
                result["score_curve"] = scores.tolist()
                result["raw"] = {
                    **result.get("raw", {}),
                    "gt_access": False,
                    "trace": audits.get(method, {"fallback": True, "selected": 0.0}),
                    "lambda_path": list(LAMBDAS),
                    "video_mean_preserved": True,
                    "intervals_preserved": True,
                }
                output.write(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({
        "n": len(rows), "coverage_nonidentical": counts,
        "selected_lambdas": selected, "max_video_mean_error": max_mean_error,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
