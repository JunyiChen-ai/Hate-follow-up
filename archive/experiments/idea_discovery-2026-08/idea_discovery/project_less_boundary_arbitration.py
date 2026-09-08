#!/usr/bin/env python3
"""Factorial controls and posterior-driven boundary arbitration for LESS.

The two geometry sources define an uncertainty bracket for each endpoint.  The
hierarchical evidence posterior then resolves the start at the strongest upward
transition and the end at the strongest downward transition inside its bracket.
No ground-truth labels or fitted numeric weights are used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path: Path, method: str | None = None):
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if method is None or row["method"] == method:
                rows[(row["dataset"], row["video_id"])] = row
    return rows


def pair(row):
    values = row.get("intervals") or []
    return list(map(float, values[0][:2])) if values else None


def midpoint(left, right):
    return [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2]


def transition_time(curve, duration, a, b, direction):
    """Select strongest directed transition in the closed endpoint bracket."""
    if not curve or b <= a:
        return (a + b) / 2
    values = np.asarray(curve, dtype=float)
    # Central differences put a transition score at each frame time.
    gradient = np.gradient(values)
    times = (np.arange(len(values), dtype=float) + 0.5) * duration / len(values)
    mask = (times >= a) & (times <= b)
    if not mask.any():
        return (a + b) / 2
    indices = np.flatnonzero(mask)
    directed = gradient[indices] if direction == "up" else -gradient[indices]
    # If evidence has no transition in the expected direction, abstain to the
    # parameter-free midpoint instead of selecting an adversarial edge.
    best = int(np.argmax(directed))
    if directed[best] <= 0:
        return (a + b) / 2
    return float(times[indices[best]])


def arbitrate(curve, duration, closure, native):
    start = transition_time(curve, duration,
                            min(closure[0], native[0]),
                            max(closure[0], native[0]), "up")
    end = transition_time(curve, duration,
                          min(closure[1], native[1]),
                          max(closure[1], native[1]), "down")
    return [start, end] if end > start else midpoint(closure, native)


def interval_contrast(curve, duration, candidate):
    """Counterfactual sufficiency contrast: retained versus removed evidence."""
    values = np.asarray(curve, dtype=float)
    values = np.log(np.clip(values, 1e-6, 1 - 1e-6) /
                    np.clip(1 - values, 1e-6, 1 - 1e-6))
    times = (np.arange(len(values), dtype=float) + 0.5) * duration / len(values)
    inside = (times >= candidate[0]) & (times <= candidate[1])
    if not inside.any() or inside.all():
        return 0.0
    # Standard errors make evidence from clips of different duration comparable.
    a, b = values[inside], values[~inside]
    scale = np.sqrt(a.var() / max(1, len(a)) + b.var() / max(1, len(b)) + 1e-8)
    return float((a.mean() - b.mean()) / scale)


def evidence_barycenter(curve, duration, closure, native):
    """Posterior-derived convex geometry, with no learned mixing weight."""
    c = interval_contrast(curve, duration, closure)
    n = interval_contrast(curve, duration, native)
    # Logistic model evidence turns the two counterfactual contrasts into a
    # normalized posterior responsibility for closure versus native geometry.
    delta = float(np.clip(c - n, -40, 40))
    closure_weight = 1.0 / (1.0 + np.exp(-delta))
    result = [closure_weight * closure[i] + (1 - closure_weight) * native[i]
              for i in (0, 1)]
    return result, closure_weight, c, n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--curve-method", required=True)
    parser.add_argument("--cca", type=Path, required=True)
    parser.add_argument("--t3al", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")

    curves = load(args.curves, args.curve_method)
    cca = load(args.cca)
    t3al = load(args.t3al)
    keys = sorted(set(curves) & set(cca) & set(t3al))
    geometries = ("native", "closure", "midpoint", "arbitrated", "evidence-barycenter")
    gates = ("none", "cca")
    for geometry in geometries:
        for gate in gates:
            method = f"less_factorial__{args.curve_method}__{geometry}__gate-{gate}"
            for key in keys:
                source = curves[key]
                native = pair(t3al[key])
                closure = pair(cca[key])
                keep = gate == "none" or closure is not None
                selected = None
                if keep and geometry == "native":
                    selected = native
                elif keep and geometry == "closure":
                    selected = closure
                responsibility = None
                contrasts = None
                if keep and native is not None and closure is not None:
                    if geometry == "midpoint":
                        selected = midpoint(closure, native)
                    elif geometry == "arbitrated":
                        selected = arbitrate(source["score_curve"],
                                             float(source["duration"]),
                                             closure, native)
                    elif geometry == "evidence-barycenter":
                        selected, responsibility, c_contrast, n_contrast = evidence_barycenter(
                            source["score_curve"], float(source["duration"]), closure, native)
                        contrasts = {"closure": c_contrast, "native": n_contrast}
                # A geometry cannot hallucinate an event when its required
                # source abstains, even in the no-existence-gate control.
                intervals = []
                if selected is not None and selected[1] > selected[0]:
                    intervals = [Interval(max(0.0, selected[0]),
                                          min(float(source["duration"]), selected[1]), 1.0)]
                append_jsonl(args.out, Prediction(
                    method, key[0], key[1], float(source["duration"]),
                    score_curve=[float(x) for x in source["score_curve"]],
                    intervals=intervals, calls=0,
                    modality_evidence={"curve_source": args.curve_method,
                                       "geometry": geometry,
                                       "existence_gate": gate,
                                       "closure_responsibility": responsibility,
                                       "counterfactual_contrasts": contrasts},
                    raw={"gt_access": False,
                         "boundary_arbitration": geometry == "arbitrated",
                         "fitted_numeric_parameters": 0}))
    print(json.dumps({"n": len(keys), "methods": len(geometries) * len(gates)}))


if __name__ == "__main__":
    main()
