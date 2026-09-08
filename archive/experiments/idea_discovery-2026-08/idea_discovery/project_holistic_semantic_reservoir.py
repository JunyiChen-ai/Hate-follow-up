#!/usr/bin/env python3
"""Add an optional whole-video MLLM semantic reservoir to a dense timeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def logmeanexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.mean(np.exp(np.clip(values - maximum, -60, 0)))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-method", required=True)
    parser.add_argument("--scores", action="append", nargs=2,
                        metavar=("DATASET", "JSONL"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    reservoir = {}
    for dataset, path in args.scores:
        for row in map(json.loads, Path(path).open()):
            value = row.get("z")
            if isinstance(value, (int, float)) and np.isfinite(value):
                reservoir[(dataset, str(row["video_id"]))] = float(value)
    used = total = 0
    with args.out.open("w") as handle:
        for row in map(json.loads, args.base.open()):
            if row.get("method") != args.base_method:
                continue
            score = np.clip(np.asarray(row["score_curve"], float),
                            np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
            logits = np.log(score / (1 - score))
            key = (str(row["dataset"]), str(row["video_id"]))
            holistic = reservoir.get(key)
            if holistic is None:
                propensity = float(logits.mean())
            else:
                propensity = logmeanexp(np.asarray([float(logits.mean()), holistic]))
                used += 1
            # Keep the final timeline in logit space.  A probability writeback
            # can round many large values to exactly one and silently destroy
            # within-video ranks even though this module only applies an offset.
            updated_score = propensity + logits - logits.mean()
            output = dict(row)
            output["method"] = "holistic_semantic_reservoir_v1"
            output["score_curve"] = updated_score.tolist()
            output["raw"] = {**output.get("raw", {}), "gt_access": False,
                             "module": "optional_whole_video_mllm_semantic_reservoir",
                             "holistic_log_odds": holistic,
                             "pooling": "equal_mass_logmeanexp_when_available",
                             "missing_action": "identity",
                             "centered_localization_residual_preserved": True,
                             "score_domain": "unbounded_logit_to_avoid_sigmoid_saturation",
                             "dataset_parameters": 0, "label_selected_parameters": 0}
            handle.write(json.dumps(output, separators=(",", ":")) + "\n")
            total += 1
    summary = {"n": total, "holistic_reservoir_available": used}
    args.out.with_suffix(".audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
