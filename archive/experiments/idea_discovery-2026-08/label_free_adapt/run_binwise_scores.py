#!/usr/bin/env python3
"""Independent-window multimodal scoring to prevent video-level score copying."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .mechanisms import parse_json_object
from .run_numpro import load_done, sample_numbered_frames
from .run_reasoning_adapters import infer
from .schema import FPS, Prediction, append_jsonl, curve_to_intervals


def score_response(text: str) -> tuple[float, dict]:
    row = parse_json_object(text)
    values = [float(row[name]) / 4.0
              for name in ("target", "harm", "endorsement", "visual_text")]
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("component outside 0-4 range")
    return min(values[0], values[1]) * (0.5 + .25 * values[2] + .25 * values[3]), row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--asr", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--nbins", type=int, default=8)
    parser.add_argument("--frames-per-bin", type=int, default=4)
    parser.add_argument("--per-dataset", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20250819)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()
            if line.strip()]
    selected, counts = [], defaultdict(int)
    for row in rows:
        if counts[row["dataset"]] < args.per_dataset:
            selected.append(row); counts[row["dataset"]] += 1
    asr = {}
    for line in Path(args.asr).read_text().splitlines():
        item = json.loads(line); asr[(item["dataset"], item["video_id"])] = item

    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.manual_seed(args.seed)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(
        args.model, max_pixels=448 * 448, local_files_only=True)
    method = "A01v4_BinwiseTimedEvidence"
    done = load_done(Path(args.out))
    for row in selected:
        if (row["dataset"], row["video_id"], method) in done:
            continue
        prediction = Prediction(method, row["dataset"], row["video_id"],
                                float(row["duration"]), seed=args.seed)
        raw = []
        try:
            frame_count = args.nbins * args.frames_per_bin
            frames, times = sample_numbered_frames(row["video_path"], frame_count)
            segments = asr.get((row["dataset"], row["video_id"]), {}).get("segments", [])
            scores = []
            for index in range(args.nbins):
                start = index * prediction.duration / args.nbins
                end = (index + 1) * prediction.duration / args.nbins
                bin_frames = [frame for frame, timestamp in zip(frames, times)
                              if start <= timestamp < end]
                if not bin_frames:
                    bin_frames = [frames[min(index * len(frames) // args.nbins,
                                             len(frames) - 1)]]
                speech = " ".join(segment["text"] for segment in segments
                                  if float(segment["end"]) > start and
                                  float(segment["start"]) < end)
                prompt = (
                    f"This is only the video window [{start:.2f},{end:.2f}) seconds. "
                    f"Timestamp-aligned speech in this window: {json.dumps(speech[:1500], ensure_ascii=False)}. "
                    "Score this window independently on integers 0-4. target: a person/group is "
                    "referenced; harm: attack/dehumanization/threat/exclusion/stereotype; endorsement: "
                    "the harmful position is endorsed rather than quoted/reported/condemned; visual_text: "
                    "imagery or visible text supports it. Uncertainty means 1 or 2, never abstain. Return "
                    'only {"target":int,"harm":int,"endorsement":int,"visual_text":int}.')
                response = infer(bin_frames, prompt, processor, model, 128)
                score, components = score_response(response)
                scores.append(score)
                raw.append({"start": start, "end": end, "speech": speech,
                            "response": response, "components": components})
            centers = (np.arange(args.nbins) + .5) * prediction.duration / args.nbins
            grid = np.arange(max(1, int(np.floor(prediction.duration * FPS)))) / FPS
            prediction.score_curve = np.interp(
                grid, centers, scores, left=scores[0], right=scores[-1]).tolist()
            prediction.intervals = curve_to_intervals(
                prediction.score_curve, prediction.duration, threshold=.25)
            prediction.modality_evidence = {"visual": True, "timestamped_speech": True}
            prediction.raw = raw
        except Exception as exc:
            prediction.error = f"{type(exc).__name__}: {exc}"
            prediction.raw = raw
        append_jsonl(Path(args.out), prediction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
