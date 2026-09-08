#!/usr/bin/env python3
"""A09: batch the released GroundingGPT video-grounding checkpoint."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .run_numpro import load_done
from .schema import Prediction, append_jsonl, intervals_to_curve, normalize_intervals


def parse_normalized_intervals(text: str, duration: float):
    pairs = re.findall(
        r"[\[({]\s*(0(?:\.\d+)?|1(?:\.0+)?)\s*[,\-]\s*"
        r"(0(?:\.\d+)?|1(?:\.0+)?)\s*[\])}]", text
    )
    return normalize_intervals(
        [[float(a) * duration, float(b) * duration, 1.0] for a, b in pairs], duration
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20250819)
    args = ap.parse_args()

    # Imports stay here so schema/CLI checks do not require the vendor environment.
    from lego import conversation as conversation_lib
    from lego.constants import (DEFAULT_VIDEO_END_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN,
                                DEFAULT_VIDEO_START_TOKEN, IMAGE_TOKEN_INDEX)
    from lego.mm_utils import tokenizer_image_token
    from lego.model.builder import CONFIG, load_pretrained_model
    from video_llama.processors.video_processor import load_video

    torch.manual_seed(args.seed)
    model, tokenizer, _, video_transform, _ = load_pretrained_model(args.model_path)
    model.eval()
    rows = [json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
    if args.limit:
        rows = rows[:args.limit]
    method = "A09_GroundingGPT"
    done = load_done(Path(args.out))
    prefix = (DEFAULT_VIDEO_START_TOKEN
              + DEFAULT_VIDEO_PATCH_TOKEN * CONFIG.video_token_len
              + DEFAULT_VIDEO_END_TOKEN)
    # Match the released Charades-STA grounding instruction distribution.
    question = (
        "Tell me the time when hateful or demeaning content targeting a person or protected group "
        "takes place in the footage."
    )

    for row in rows:
        key = (row["dataset"], row["video_id"], method)
        if key in done:
            continue
        pred = Prediction(method, row["dataset"], row["video_id"],
                          float(row["duration"]), seed=args.seed)
        try:
            video = load_video(video_path=row["video_path"], n_frms=model.config.max_frame,
                               height=224, width=224, sampling="uniform", return_msg=False)
            tensor = video_transform(video).unsqueeze(0).to(CONFIG.device, dtype=torch.bfloat16)
            conv = conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], prefix + "\n" + question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                        return_tensors="pt").unsqueeze(0).to(CONFIG.device)
            with torch.inference_mode():
                out = model.generate(ids, videos=tensor, do_sample=False,
                                     max_new_tokens=args.max_new_tokens, use_cache=True)
            answer = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
            pred.intervals = parse_normalized_intervals(answer, pred.duration)
            pred.score_curve = intervals_to_curve(pred.intervals, pred.duration)
            pred.raw = {"answer": answer, "official_uniform_frames": model.config.max_frame,
                        "modality": "video"}
        except Exception as exc:
            pred.error = f"{type(exc).__name__}: {exc}"
        append_jsonl(Path(args.out), pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
