#!/usr/bin/env python3
"""A12 official TimeLens checkpoint on the sanitized hateful-event query."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

from .mechanisms import GENERIC_QUERY
from .run_numpro import load_done
from .schema import Prediction, append_jsonl, intervals_to_curve, normalize_intervals


def extract_times(text: str, duration: float) -> list:
    patterns = [r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)"]
    values = []
    for pattern in patterns:
        values = [[float(a), float(b), 1.0] for a, b in re.findall(pattern, text)]
        if values:
            break
    return normalize_intervals(values, duration)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="TencentARC/TimeLens-8B")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20250819)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()
            if line.strip()]
    if args.limit:
        rows = rows[:args.limit]

    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.manual_seed(args.seed)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(
        args.model, padding_side="left", do_resize=False,
        trust_remote_code=True, local_files_only=True)
    manifest_path = Path(args.manifest).resolve()
    runner_path = Path(__file__).resolve()
    provenance = {
        "model_id": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(runner_path.read_bytes()).hexdigest(),
        "seed": args.seed,
        "official_video_fps": args.fps,
    }
    method = "A12_TimeLens8B"
    done = load_done(Path(args.out))
    for row in rows:
        if (row["dataset"], row["video_id"], method) in done:
            continue
        prediction = Prediction(method, row["dataset"], row["video_id"],
                                float(row["duration"]), seed=args.seed)
        try:
            prompt = (
                "Please find every visual event described by the sentence below, determining "
                "its starting and ending times. If there are multiple disjoint events, list all "
                "start-end pairs in seconds. Sentence: " + GENERIC_QUERY)
            messages = [{"role": "user", "content": [
                {"type": "video", "video": row["video_path"],
                 "min_pixels": 16 * 32 * 32, "total_pixels": 3584 * 32 * 32,
                 "fps": args.fps},
                {"type": "text", "text": prompt},
            ]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(
                messages, image_patch_size=16, return_video_kwargs=True,
                return_video_metadata=True)
            videos, metadata = zip(*videos)
            inputs = processor(text=[text], images=images, videos=list(videos),
                               video_metadata=list(metadata), padding=True,
                               return_tensors="pt", **video_kwargs).to(model.device)
            prediction.calls = 1
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=512)
            answer = processor.batch_decode(
                output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            prediction.intervals = extract_times(answer, prediction.duration)
            prediction.score_curve = intervals_to_curve(
                prediction.intervals, prediction.duration)
            prediction.raw = {"answer": answer, **provenance}
        except Exception as exc:
            prediction.error = f"{type(exc).__name__}: {exc}"
        append_jsonl(Path(args.out), prediction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
