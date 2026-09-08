#!/usr/bin/env python3
"""Hierarchical LESS decomposition into video propensity and temporal residual."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def load(path: Path, method: str):
    return {(row["dataset"], row["video_id"]): row
            for row in map(json.loads, path.open()) if row["method"] == method}


def logit(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def sigmoid(values):
    values = np.clip(values, -40, 40)
    return 1 / (1 + np.exp(-values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--absolute", type=Path, required=True)
    parser.add_argument("--absolute-method", required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--residual-method", required=True)
    parser.add_argument("--cca", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    absolute = load(args.absolute, args.absolute_method)
    residual = load(args.residual, args.residual_method)
    cca = {(row["dataset"], row["video_id"]): row
           for row in map(json.loads, args.cca.open())}
    keys = sorted(set(absolute) & set(residual) & set(cca))
    for key in keys:
        absolute_logit = logit(absolute[key]["score_curve"])
        residual_logit = logit(residual[key]["score_curve"])
        video_propensity = float(absolute_logit.mean())
        temporal_residual = residual_logit - residual_logit.mean()
        posterior = sigmoid(video_propensity + temporal_residual)
        geometry = cca[key]
        intervals = [Interval(float(value[0]), float(value[1]),
                              float(value[2]) if len(value) > 2 else 1.0)
                     for value in geometry["intervals"]]
        append_jsonl(args.out, Prediction(
            "fact_less_3v_hierarchical_v3", key[0], key[1],
            float(absolute[key]["duration"]), score_curve=posterior.tolist(),
            intervals=intervals, calls=0,
            modality_evidence={
                "video_propensity_logit": video_propensity,
                "absolute_source": args.absolute_method,
                "temporal_residual_source": args.residual_method,
                "temporal_residual_mean": float(temporal_residual.mean()),
                "geometry_source": "cca_v1"},
            raw={"gt_access": False,
                 "hierarchical_equation": "logit_y_vt=video_propensity_v+zero_mean_temporal_residual_vt",
                 "learned_fusion_weights": 0,
                 "factorized_interval_readout": "consensus_gated_proposal_closure"}))
    print(json.dumps({"n": len(keys), "method": "fact_less_3v_hierarchical_v3"}))


if __name__ == "__main__":
    main()
