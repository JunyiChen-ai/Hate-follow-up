#!/usr/bin/env python3
"""Cross-fitted two-level truth discovery for label-free hate localization.

Unlike post-hoc centering, this model presents different observations to two
separately estimated label models: per-video source summaries explain global
hate propensity, while within-video source deviations explain local episodes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.idea_discovery.run_less import VIEW_NAMES, collect, fit_label_model, infer_static
from scripts.label_free_adapt.schema import Prediction, append_jsonl


def log_odds(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def global_observation(evidence):
    """One categorical summary per source, ignoring temporal abstentions."""
    result = []
    for view in range(evidence.shape[1]):
        active = evidence[:, view][evidence[:, view] != 0]
        mean = float(active.mean()) if len(active) else 0.0
        result.append(1 if mean > 0 else (-1 if mean < 0 else 0))
    return np.asarray(result, dtype=np.int8)[None, :]


def local_observation(evidence):
    """Source-wise deviations from each video's own temporal baseline."""
    local = np.zeros_like(evidence, dtype=np.int8)
    for view in range(evidence.shape[1]):
        values = evidence[:, view].astype(float)
        # The mean (including abstentions as zero evidence) preserves both
        # positive and negative deviations for every non-constant source.  A
        # median would collapse the majority state to abstention for binary
        # sources and destroy the local negative channel.
        center = float(values.mean())
        delta = values - center
        local[:, view] = np.where(delta > 0, 1, np.where(delta < 0, -1, 0))
    return local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a10", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--visual-mode", choices=("majority", "a10", "unanimous"), default="majority")
    parser.add_argument("--fit-scope", choices=("global", "dataset"), default="global")
    parser.add_argument("--text-shift", choices=("aligned", "half"), default="aligned")
    parser.add_argument("--audio-mode", choices=("absolute", "within_video"), default="absolute")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # collect() expects this newer option but two-level inference does not use it.
    args.synchrony_calibration = "none"
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    data = collect(args)
    transformed = {key: {"global": global_observation(row["evidence"]),
                         "local": local_observation(row["evidence"])}
                   for key, row in data.items()}
    scopes = sorted({key[0] for key in data}) if args.fit_scope == "dataset" else ["GLOBAL"]
    models = {}
    audit = {"gt_access": False, "n": len(data), "models": {}}
    for scope in scopes:
        for target_fold in (0, 1):
            keys = [key for key, row in data.items() if row["fold"] != target_fold and
                    (scope == "GLOBAL" or key[0] == scope)]
            global_model = fit_label_model([transformed[key]["global"] for key in keys])
            local_model = fit_label_model([transformed[key]["local"] for key in keys])
            models[(scope, target_fold)] = global_model, local_model
            audit["models"][f"{scope}:target_fold_{target_fold}"] = {
                "global_reliability": dict(zip(VIEW_NAMES, global_model["reliability"])),
                "local_reliability": dict(zip(VIEW_NAMES, local_model["reliability"])),
                "global_prior": global_model["prior"], "local_prior": local_model["prior"]}
    method = ("less_twolevel_" + args.text_shift + "_" + args.audio_mode + "_v2")
    for key, row in data.items():
        scope = key[0] if args.fit_scope == "dataset" else "GLOBAL"
        global_model, local_model = models[(scope, row["fold"])]
        global_p = infer_static(transformed[key]["global"], global_model["prior"],
                                global_model["theta"])[0]
        local_p = infer_static(transformed[key]["local"], local_model["prior"],
                               local_model["theta"])
        residual = log_odds(local_p) - log_odds(local_p).mean()
        posterior = sigmoid(float(log_odds([global_p])[0]) + residual)
        append_jsonl(args.out, Prediction(
            method, key[0], key[1], row["duration"], score_curve=posterior.tolist(),
            intervals=[], calls=0,
            modality_evidence={"fold": row["fold"], "global_posterior": float(global_p),
                               "global_reliability": dict(zip(VIEW_NAMES, global_model["reliability"])),
                               "local_reliability": dict(zip(VIEW_NAMES, local_model["reliability"])),
                               "local_residual_mean": float(residual.mean())},
            raw={"gt_access": False, "fit": "opposite_hash_fold",
                 "model": "two_level_categorical_truth_discovery",
                 "global_observation": "nonabstain_source_majority_per_video",
                 "local_observation": "sourcewise_within_video_mean_deviation"}))
    args.out.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"n": len(data), "method": method}, indent=2))


if __name__ == "__main__":
    main()
