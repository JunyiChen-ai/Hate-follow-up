#!/usr/bin/env python3
"""Resumable VERA inference with decoder reuse and optional ``batch_chat``.

This is deliberately separate from the frozen reproduction adapter.  Batch size
one retains its model call and frame-index formula; larger batches are enabled
only after an exact output A/B check for the selected model/backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

import vera_adapter as legacy
from hate_common import data as hdata


def frame_indices(start: float, window: float, count: int, fps: float,
                  duration: float, frame_count: int) -> np.ndarray:
    """Exactly the index calculation used by ``vera_adapter.read_frames``."""
    end = min(duration, start + window)
    times = np.linspace(start, max(start, end - 1 / max(fps, 1)), count)
    return np.clip(np.rint(times * fps).astype(int), 0, frame_count - 1)


class ReusableVideoReader:
    def __init__(self, path: Path, num_threads: int = 2):
        import decord
        self.reader = decord.VideoReader(str(path), num_threads=num_threads)
        self.fps = float(self.reader.get_avg_fps())
        self.frame_count = len(self.reader)
        self.duration = self.frame_count / max(self.fps, 1e-6)

    def frames(self, start: float, window: float, count: int = 8):
        indices = frame_indices(start, window, count, self.fps,
                                self.duration, self.frame_count)
        return [Image.fromarray(x) for x in self.reader.get_batch(indices).asnumpy()]


def predict_batch(model, tokenizer, image_batches, prompts, batch_size: int):
    """Return predictions in input order; batch=1 uses the frozen call verbatim."""
    if batch_size == 1:
        return [legacy.predict(model, tokenizer, images, prompts)
                for images in image_batches]

    import torch
    pixels = torch.cat([legacy.transform_images(images) for images in image_batches])
    pixels = pixels.to("cuda", dtype=torch.bfloat16)
    questions = [legacy.question(prompts, len(images)) for images in image_batches]
    config = dict(num_beams=1, max_new_tokens=256, do_sample=False)
    with torch.inference_mode():
        responses = model.batch_chat(
            tokenizer, pixels, questions, config,
            num_patches_list=[len(images) for images in image_batches],
        )
    output = []
    for response in responses:
        hits = legacy.re.findall(r"Output\s*:\s*([01])", response,
                                 flags=legacy.re.I)
        score = (int(hits[-1]) if hits else
                 int("yes" in response.lower() and
                     "no, there" not in response.lower()))
        output.append((score, response))
    return output


def infer(args):
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(args.prompt_json).read_text())
    model, tokenizer, _ = legacy.load_model(selection["attention_backend"])
    prompts = selection["prompts"]
    gt = hdata.gt_arrays(args.corpus, args.split)
    ids = [vid for vid in hdata.load_split(args.corpus, args.split) if vid in gt]
    if args.only_video:
        if args.only_video not in ids:
            raise ValueError(f"{args.only_video!r} is not in {args.corpus}/{args.split}")
        ids = [args.only_video]

    for vi, vid in enumerate(ids, 1):
        out = root / f"{vid}.json"
        starts = np.arange(0, len(gt[vid]), args.stride)
        if legacy.valid_raw_result(out, vid, len(starts)):
            print(f"[{vi}/{len(ids)}] {vid}: already complete", flush=True)
            continue

        reader = ReusableVideoReader(legacy.video_path(args.corpus, vid))
        records = []
        for offset in range(0, len(starts), args.batch_size):
            batch_starts = starts[offset:offset + args.batch_size]
            image_batches = [reader.frames(float(start), args.window, 8)
                             for start in batch_starts]
            predictions = predict_batch(model, tokenizer, image_batches, prompts,
                                        args.batch_size)
            for start, (score, response) in zip(batch_starts, predictions):
                records.append({
                    "start": float(start),
                    "end": min(reader.duration, start + args.window),
                    "score": score,
                    "response": response,
                })

        payload = {"video_id": vid, "duration": reader.duration,
                   "segments": records}
        temporary = out.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False))
        os.replace(temporary, out)
        print(f"[{vi}/{len(ids)}] {vid}: {len(records)} windows", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, choices=hdata.CORPORA)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--stride", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--only-video")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    infer(args)


if __name__ == "__main__":
    main()
