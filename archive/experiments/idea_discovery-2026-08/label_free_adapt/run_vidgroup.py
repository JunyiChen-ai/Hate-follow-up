#!/usr/bin/env python3
"""A10 Vid-Group zero-shot checkpoint on custom CLIP-L/14 features."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from .mechanisms import GENERIC_QUERY
from .run_numpro import load_done
from .schema import Prediction, append_jsonl, intervals_to_curve, normalize_intervals


def configure(config: dict) -> tuple[dict, list]:
    cfg = copy.deepcopy(config)
    sim = cfg["model"]["config"]["simbase"]
    ratios = np.linspace(sim["anchor_meta"]["scale_ratios_start"], 1.0,
                         sim["anchor_meta"]["scale_ratios_num"]).tolist()
    lengths, anchors, length = [], [], cfg["dataset"]["num_sample_clips"] // 4
    while length > 0:
        lengths.append(length); length //= 2
    sim["anchor"]["feature_map_len"] = lengths
    for level, feature_length in enumerate(lengths, 1):
        sim["anchor"][f"scale_ratios_anchor{level}"] = ratios
        unit = cfg["dataset"]["num_sample_clips"] / feature_length
        anchors.append([[[index * unit + unit / 2 - ratio * unit / 2,
                          index * unit + unit / 2 + ratio * unit / 2]
                         for ratio in ratios] for index in range(feature_length)])
    cfg["model"]["config"]["num_sample_clips"] = cfg["dataset"]["num_sample_clips"]
    return cfg, anchors


def decode_candidates(outputs, anchors, duration: float, bins: int, cfg: dict):
    """Return every regressed anchor in descending raw-overlap order.

    Keeping this step separate from the historical top-1 readout makes it
    possible to audit the proposal ceiling without changing A10 predictions.
    """
    overlaps, regressions = outputs
    candidates = []
    ratio_center = cfg["model"]["config"]["simbase"]["loss"]["ratio_center"]
    ratio_width = cfg["model"]["config"]["simbase"]["loss"]["ratio_width"]
    for level, (overlap, regression) in enumerate(zip(overlaps, regressions)):
        overlap = overlap[0].detach().float().cpu().numpy()
        regression = regression[0].detach().float().cpu().numpy()
        for ratio_id in range(overlap.shape[0]):
            for position in range(overlap.shape[1]):
                left, right = anchors[level][position][ratio_id]
                center, width = (left + right) / 2, right - left
                center += ratio_center * width * regression[2 * ratio_id, position]
                width *= math.exp(ratio_width * regression[2 * ratio_id + 1, position])
                score = float(overlap[ratio_id, position])
                candidates.append((score, max(0, center - width / 2),
                                   min(bins, center + width / 2)))
    return sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))


def decode(outputs, anchors, duration: float, bins: int, cfg: dict):
    candidates = decode_candidates(outputs, anchors, duration, bins, cfg)
    score, left, right = candidates[0]
    confidence = float(1 / (1 + math.exp(-max(-30.0, min(30.0, score)))))
    return normalize_intervals([[left / bins * duration, right / bins * duration,
                                 confidence]], duration), score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo", default="third_party/label_free_adapt/Vid-Group")
    parser.add_argument("--checkpoint", default="ckpt/charades/zero_shot.ckpt")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    config = json.loads((repo / "experiment/charades/recorrect_eval_configs_on_ZeroShot+Unsup+Full.json").read_text())
    config, anchors = configure(config)
    sys.path.insert(0, str(repo / "lib"))
    from models.simbase import SimBase
    # The released code assumes the older CLIP loader returned fp32 weights;
    # current OpenAI CLIP keeps CUDA weights in fp16 while TextEncoder forces
    # fp32 activations, so normalize the whole released model to fp32.
    model = SimBase(config["model"]["config"]).cuda().float().eval()
    model.load_non_clip_parameters(str(repo / args.checkpoint))
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()
            if line.strip()]
    method = "A10_VidGroupZeroShot"
    done = load_done(Path(args.out))
    for row in rows:
        if (row["dataset"], row["video_id"], method) in done:
            continue
        prediction = Prediction(method, row["dataset"], row["video_id"],
                                float(row["duration"]))
        try:
            feature_path = Path(args.features) / row["dataset"] / f"{row['video_id']}.npy"
            features = torch.from_numpy(np.load(feature_path)).unsqueeze(0).cuda().float()
            with torch.inference_mode():
                outputs = model([GENERIC_QUERY], features, None)
            prediction.intervals, raw_score = decode(
                (outputs["pred"]["pred_overlap"], outputs["pred"]["pred_reg"]),
                anchors, prediction.duration, 128, config)
            prediction.score_curve = intervals_to_curve(prediction.intervals,
                                                         prediction.duration)
            prediction.raw = {"proposal_logit": raw_score, "checkpoint": args.checkpoint}
        except Exception as exc:
            prediction.error = f"{type(exc).__name__}: {exc}"
        append_jsonl(Path(args.out), prediction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
