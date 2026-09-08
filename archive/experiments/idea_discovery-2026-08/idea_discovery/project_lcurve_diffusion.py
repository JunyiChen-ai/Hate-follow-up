#!/usr/bin/env python3
"""Per-video L-curve-corner temporal diffusion.

The strength maximizes curvature of the log residual-energy versus log
roughness curve. Bounds arise from the video's chain spectrum and machine
precision; the finite-difference step is a floating-point accuracy rule.
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
    n = len(signal); eps = np.finfo(float).eps
    if n < 4 or not np.all(np.isfinite(signal)):
        return signal.copy(), 0.0, {"fallback": True}
    eigen = 2.0 - 2.0 * np.cos(np.pi * np.arange(1, n) / n)
    z2 = dct(signal, type=2, norm="ortho")[1:] ** 2
    if float(np.sum(z2)) <= eps:
        return signal.copy(), 0.0, {"fallback": True}
    lower = np.log(np.sqrt(eps) / float(eigen.max()))
    upper = np.log(1.0 / (np.sqrt(eps) * float(eigen.min())))
    h = float(eps ** 0.2)

    def point(log_strength: float) -> np.ndarray:
        strength = float(np.exp(log_strength)); q = strength * eigen
        residual = max(np.finfo(float).tiny,
                       float(np.sum(z2 * (q / (1.0 + q)) ** 2)))
        roughness = max(np.finfo(float).tiny,
                        float(np.sum(eigen * z2 / (1.0 + q) ** 2)))
        return 0.5 * np.log([residual, roughness])

    def curvature(log_strength: float) -> float:
        left, center, right = (point(log_strength - h), point(log_strength),
                               point(log_strength + h))
        first = (right - left) / (2.0 * h)
        second = (right - 2.0 * center + left) / (h * h)
        numerator = abs(first[0] * second[1] - first[1] * second[0])
        denominator = max(np.finfo(float).tiny,
                          float((first @ first) ** 1.5))
        return float(numerator / denominator)

    result = minimize_scalar(lambda x: -curvature(x), bounds=(lower + h, upper - h),
                             method="bounded", options={"xatol": np.sqrt(eps)})
    strength = float(np.exp(result.x))
    fitted = smooth(signal, np.ones(n - 1), strength)
    return fitted, strength, {"fallback": False, "strength": strength,
                              "lcurve_curvature": curvature(result.x),
                              "optimizer_success": bool(result.success),
                              "bounds_source": "video_spectrum_machine_precision",
                              "derivative_step_source": "machine_epsilon_power_rule"}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--base-method", required=True); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): raise RuntimeError(f"refusing existing output: {args.out}")
    rows = load(args.base, args.base_method)
    with args.out.open("w") as handle:
        for _, row in sorted(rows.items()):
            scores = np.asarray(row["score_curve"], float); clipped=np.clip(scores,1e-6,1-1e-6)
            logits=np.log(clipped/(1-clipped)); fitted,strength,audit=select(logits-logits.mean())
            output=dict(row); output["method"]="lcurve_diffusion_v1"
            output["score_curve"]=reconstruct(scores,fitted).tolist()
            output["raw"]={**output.get("raw",{}),"gt_access":False,
                           "module":"per_video_lcurve_diffusion","dataset_parameters":0,
                           "label_selected_parameters":0,"video_mean_preserved":True,
                           "intervals_preserved":True,"lcurve":audit}
            handle.write(json.dumps(output,separators=(",",":"))+"\n")


if __name__=="__main__": main()
