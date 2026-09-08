#!/usr/bin/env python3
"""One-video GroundingGPT smoke test using the released inference path."""

import argparse
import torch

from lego import conversation as conversation_lib
from lego.constants import (
    DEFAULT_VIDEO_END_TOKEN,
    DEFAULT_VIDEO_PATCH_TOKEN,
    DEFAULT_VIDEO_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from lego.mm_utils import tokenizer_image_token
from lego.model.builder import CONFIG, load_pretrained_model
from video_llama.processors.video_processor import load_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    model, tokenizer, _, video_transform, _ = load_pretrained_model(args.model_path)
    model.eval()
    video = load_video(
        video_path=args.video,
        n_frms=model.config.max_frame,
        height=224,
        width=224,
        sampling="uniform",
        return_msg=False,
    )
    video_tensor = video_transform(video).unsqueeze(0).to(CONFIG.device, dtype=torch.bfloat16)
    question = (
        "Temporally locate every segment containing hateful or demeaning content toward a person "
        "or protected group. Consider both visible actions and spoken content. Return normalized "
        "start and end times between 0 and 1 as [start, end]. If none exists, answer none."
    )
    prefix = (
        DEFAULT_VIDEO_START_TOKEN
        + DEFAULT_VIDEO_PATCH_TOKEN * CONFIG.video_token_len
        + DEFAULT_VIDEO_END_TOKEN
    )
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], prefix + "\n" + question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(CONFIG.device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            videos=video_tensor,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    print(tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
