#!/usr/bin/env python3
"""Native visual core-vs-shell witness for PACT boundary edits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from scripts.idea_discovery.run_cwa_visual_warrants import uniform_times
from scripts.idea_discovery.run_melt import MLLM
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times

PROMPT = (
    "The TOP row contains ordered frames from a proposed hateful-event CORE. "
    "The BOTTOM row contains equally many ordered frames from the temporal SHELL "
    "that would be removed. Judge only visible evidence, including on-screen text, "
    "symbols, gestures, and interactions. Is visible evidence of an attack, threat, "
    "dehumanization, exclusion, or hateful stereotype toward a person or social group "
    "more concentrated in CORE than in SHELL? Do not infer unheard speech. "
    "Answer Yes or No only:"
)


def load(path: Path, method: str | None = None) -> dict[tuple[str, str], dict]:
    output = {}
    for row in map(json.loads, path.open()):
        if method is None or row.get("method") == method:
            output[(row["dataset"], row["video_id"])] = row
    return output


def shell(base: list[float], core: list[float]) -> list[tuple[float, float]]:
    output = []
    if core[0] > base[0]:
        output.append((base[0], core[0]))
    if core[1] < base[1]:
        output.append((core[1], base[1]))
    return output


def paired_canvas(core_images: list[Image.Image], shell_images: list[Image.Image]) -> Image.Image:
    tile_w, tile_h = 112, 112
    output = Image.new("RGB", (8 * tile_w, 2 * tile_h + 36), "white")
    draw = ImageDraw.Draw(output)
    draw.text((4, 2), "CORE (retained)", fill=(0, 100, 0))
    draw.text((4, tile_h + 20), "SHELL (removed)", fill=(160, 0, 0))
    for index, image in enumerate(core_images):
        output.paste(image.resize((tile_w, tile_h)), (index * tile_w, 18))
    for index, image in enumerate(shell_images):
        output.paste(image.resize((tile_w, tile_h)), (index * tile_w, tile_h + 34))
    return output


@torch.inference_mode()
def binary_score(model: MLLM, image: Image.Image) -> float:
    message = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    text = model.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = model.processor(text=[text], images=[image], return_tensors="pt").to(model.model.device)
    logits = model.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    model.calls += 1
    values = logits[:, model.binary_token_ids]
    return float((values[:, 0] - values[:, 1]).item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    manifest, base = load(args.manifest), load(args.base)
    candidate = load(args.candidate, args.method)
    keys = sorted(set(manifest) & set(base) & set(candidate))
    keys = [key for key in keys if candidate[key].get("modality_evidence", {}).get("effective_edit")]
    model = MLLM(args.model)
    config = {"model": args.model, "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
              "offsets": [0.25, 0.75], "frames_per_view": 8,
              "padding_side": model.processor.tokenizer.padding_side, "gt_access": False,
              "transcript_supplied": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        for index, key in enumerate(keys, 1):
            base_interval = [float(x) for x in base[key]["intervals"][0][:2]]
            core_interval = [float(x) for x in candidate[key]["intervals"][0][:2]]
            shell_intervals = shell(base_interval, core_interval)
            scores, times = [], []
            for offset in (0.25, 0.75):
                core_times = uniform_times([tuple(core_interval)], 8, offset)
                shell_times = uniform_times(shell_intervals, 8, offset)
                core_images, _, _ = frames_at_times(Path(manifest[key]["video_path"]), core_times)
                shell_images, _, _ = frames_at_times(Path(manifest[key]["video_path"]), shell_times)
                scores.append(binary_score(model, paired_canvas(core_images, shell_images)))
                times.append({"core": core_times.tolist(), "shell": shell_times.tolist()})
            passed = all(value > 0 for value in scores)
            row = {"dataset": key[0], "video_id": key[1], "base": base_interval,
                   "core": core_interval, "scores": scores, "times": times,
                   "visual_warrant": passed, "config": config}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"i": index, "n": len(keys), "key": key,
                              "scores": scores, "pass": passed}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
