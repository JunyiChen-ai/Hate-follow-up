#!/usr/bin/env python3
"""Per-video empirical-Bayes temporal diffusion without dataset parameters.

The centered logit trace is modeled as a latent random walk observed with
independent noise.  Observation noise and latent increment energy are
estimated from the same video.  Their variance ratio is the closed-form
regularization strength; no label, split statistic, search grid, or dataset
constant is used.

Three estimators are emitted for mechanism falsification.  They are fixed by
the assumed noise model rather than selected with localization labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.idea_discovery.project_role_orthogonal_transport import load
from scripts.idea_discovery.project_trace_gcv import reconstruct, smooth


# Gaussian consistency constants are distributional identities, not tuned
# hyperparameters: median(|N(0,1)|) and median(chi-square_1).
NORMAL_ABS_MEDIAN = 0.6744897501960817
CHISQ1_MEDIAN = 0.4549364231195727


def robust_variance(values: np.ndarray, scale: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return 0.0
    centered = values - np.median(values)
    sigma = np.median(np.abs(centered)) / (NORMAL_ABS_MEDIAN * scale)
    return float(sigma * sigma)


def strengths(signal: np.ndarray) -> dict[str, tuple[float, dict]]:
    n = len(signal)
    eps = np.finfo(np.float64).eps
    if n < 4 or not np.all(np.isfinite(signal)):
        return {name: (0.0, {"fallback": True}) for name in
                ("difference", "curvature", "spectral")}

    first = np.diff(signal)
    second = np.diff(signal, n=2)

    # Under iid observation noise e_t, Var(Delta e)=2 sigma_e^2 and
    # Var(Delta^2 e)=6 sigma_e^2.
    noise_from_first = robust_variance(first, np.sqrt(2.0))
    noise_from_second = robust_variance(second, np.sqrt(6.0))

    # Random-walk increments contain latent innovation plus differenced noise.
    first_energy = float(np.mean((first - np.mean(first)) ** 2))

    def ratio(noise: float) -> tuple[float, dict]:
        innovation = max(eps, first_energy - 2.0 * noise)
        value = float(noise / innovation)
        return value, {"fallback": False, "noise_variance": noise,
                       "innovation_variance": innovation,
                       "strength": value}

    # Orthogonal spectral control/alternative: for a chain, the upper half of
    # DCT frequencies is noise-dominated under the same smooth-latent model.
    # The split is the exact Nyquist half, not a fitted fraction.
    mirrored = np.r_[signal, signal[-2:0:-1]]
    spectrum = np.fft.rfft(mirrored).real[:n]
    high = spectrum[(n + 1) // 2:]
    spectral_noise = (float(np.median(high * high)) / CHISQ1_MEDIAN / n
                      if len(high) else 0.0)

    return {
        "difference": ratio(noise_from_first),
        "curvature": ratio(noise_from_second),
        "spectral": ratio(spectral_noise),
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
            signal = logits - logits.mean()
            for estimator, (strength, audit) in strengths(signal).items():
                fitted = (signal.copy() if strength == 0.0 else
                          smooth(signal, np.ones(len(signal) - 1), strength))
                output = dict(row)
                output["method"] = f"self_noise_{estimator}_diffusion_v1"
                output["score_curve"] = reconstruct(scores, fitted).tolist()
                output["raw"] = {
                    **output.get("raw", {}),
                    "gt_access": False,
                    "module": "per_video_empirical_bayes_diffusion",
                    "noise_estimator": estimator,
                    "dataset_parameters": 0,
                    "label_selected_parameters": 0,
                    "video_mean_preserved": True,
                    "intervals_preserved": True,
                    "self_noise": audit,
                }
                handle.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
