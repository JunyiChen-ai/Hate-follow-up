#!/usr/bin/env python3
"""Non-abstaining frame-evidence pilot on a local Qwen3-VL backbone."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .mechanisms import dense_evidence_prompt, parse_dense_scores
from .run_numpro import load_done, sample_numbered_frames
from .run_reasoning_adapters import infer
from .schema import FPS, Prediction, append_jsonl, curve_to_intervals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--nframes", type=int, default=64)
    parser.add_argument("--nbins", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20250819)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--asr", help="Optional timestamped ASR JSONL")
    args = parser.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.manual_seed(args.seed)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(
        args.model, max_pixels=448 * 448, local_files_only=True)
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()
            if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    asr = {}
    if args.asr:
        for line in Path(args.asr).read_text().splitlines():
            item = json.loads(line)
            asr[(item["dataset"], item["video_id"])] = item
    method = "A01v3_TimedDenseEvidence" if args.asr else "A01v2_DenseEvidence"
    done = load_done(Path(args.out))
    for row in rows:
        if (row["dataset"], row["video_id"], method) in done:
            continue
        prediction = Prediction(method, row["dataset"], row["video_id"],
                                float(row["duration"]), seed=args.seed)
        try:
            frames, times = sample_numbered_frames(row["video_path"], args.nframes)
            timed_bins = None
            if args.asr:
                item = asr.get((row["dataset"], row["video_id"]), {})
                timed_bins = []
                width = prediction.duration / args.nbins
                for index in range(args.nbins):
                    start, end = index * width, (index + 1) * width
                    snippets = [segment["text"] for segment in item.get("segments", [])
                                if float(segment["end"]) > start and
                                float(segment["start"]) < end]
                    timed_bins.append(" ".join(snippets)[:1000])
            prompt = dense_evidence_prompt(
                range(len(frames)), prediction.duration, row.get("transcript", ""),
                args.nbins, timed_bins=timed_bins)
            response = infer(frames, prompt, processor, model, args.max_new_tokens)
            sparse_scores = parse_dense_scores(response, args.nbins)
            bin_times = (np.arange(args.nbins) + 0.5) * prediction.duration / args.nbins
            grid_times = np.arange(max(1, int(np.floor(prediction.duration * FPS)))) / FPS
            prediction.score_curve = np.interp(
                grid_times, bin_times, np.asarray(sparse_scores),
                left=sparse_scores[0], right=sparse_scores[-1]).tolist()
            # Fixed before GT access; only affects interval metrics, never ROC/PR.
            prediction.intervals = curve_to_intervals(
                prediction.score_curve, prediction.duration, threshold=0.25)
            prediction.modality_evidence = {
                "visual_frames": len(frames), "transcript_timestamp_aligned": False,
            }
            prediction.raw = {"response": response, "frame_times": times,
                              "sparse_scores": sparse_scores}
        except Exception as exc:
            prediction.error = f"{type(exc).__name__}: {exc}"
        append_jsonl(Path(args.out), prediction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
