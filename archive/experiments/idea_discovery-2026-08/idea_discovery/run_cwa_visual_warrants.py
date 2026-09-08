#!/usr/bin/env python3
"""CWA Stage-B visual warrant on frozen VASTA transcript-veto cases.

The intervention uses only native frames.  Two offset samplings compare the
visual proposal core with the chronological complement at an identical frame
count.  No black frame, blur, mask seam, label, or transcript is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.idea_discovery.run_melt import MLLM
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times

PROMPT = (
    "These are ordered frames sampled from one part of a video. Judge only visually "
    "observable evidence. Do the frames themselves show an attack, dehumanization, "
    "threat, exclusion, or hateful stereotype directed at a person or social/protected "
    "group? Visible written text, symbols, and gestures count; do not imagine unheard "
    "speech or unseen context. Answer Yes or No only:"
)


def load(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row
            for row in map(json.loads, path.open())}


def uniform_times(intervals: list[tuple[float, float]], n: int, offset: float) -> np.ndarray:
    lengths = np.asarray([max(0.0, b - a) for a, b in intervals], float)
    total = float(lengths.sum())
    if total <= 0:
        raise ValueError("empty native view")
    positions = (np.arange(n, dtype=float) + offset) / n * total
    cumulative = np.cumsum(lengths)
    out = []
    for position in positions:
        index = int(np.searchsorted(cumulative, position, side="right"))
        index = min(index, len(intervals) - 1)
        before = 0.0 if index == 0 else cumulative[index - 1]
        out.append(intervals[index][0] + position - before)
    return np.asarray(out)


def canvas(images: list[Image.Image]) -> Image.Image:
    tile_w, tile_h, cols = 224, 126, 4
    rows = int(np.ceil(len(images) / cols))
    output = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    for index, image in enumerate(images):
        output.paste(image.resize((tile_w, tile_h)), ((index % cols) * tile_w, (index // cols) * tile_h))
    return output


@torch.inference_mode()
def binary_score(model: MLLM, image: Image.Image) -> float:
    msg = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    text = model.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    inputs = model.processor(text=[text], images=[image], return_tensors="pt").to(model.model.device)
    logits = model.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    model.calls += 1
    values = logits[:, model.binary_token_ids]
    return float((values[:, 0] - values[:, 1]).item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--a08", type=Path, required=True)
    parser.add_argument("--a12", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()

    manifest, core = load(args.manifest), load(args.core)
    a08, a12, text = load(args.a08), load(args.a12), load(args.text)
    keys = sorted(set(manifest) & set(core) & set(a08) & set(a12) & set(text))
    anchors = {float(text[key]["config"]["null_log_odds"]) for key in keys}
    if len(anchors) != 1:
        raise RuntimeError(f"expected one text null, got {anchors}")
    anchor = anchors.pop()
    selected = [key for key in keys
                if bool(a08[key].get("intervals")) != bool(a12[key].get("intervals"))
                and float(text[key]["log_odds"]) < anchor and core[key].get("intervals")]

    prior = load(args.out) if args.out.exists() else {}
    model = MLLM(args.model)
    config = {
        "method": "cwa_visual_native_matched_v1", "model": args.model,
        "frames_per_view": args.frames, "offsets": [0.25, 0.75],
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "gt_access": False, "transcript_supplied_to_visual_witness": False,
        "padding_side": model.processor.tokenizer.padding_side,
    }
    for index, key in enumerate(selected, 1):
        if key in prior:
            continue
        row, proposal = manifest[key], core[key]["intervals"]
        start = min(float(value[0]) for value in proposal)
        end = max(float(value[1]) for value in proposal)
        duration = float(row["duration"])
        inside_intervals = [(start, end)]
        outside_intervals = [(0.0, start), (end, duration)]
        scores = {"inside": [], "outside": []}
        times = {"inside": [], "outside": []}
        for offset in (0.25, 0.75):
            for name, intervals in (("inside", inside_intervals), ("outside", outside_intervals)):
                sample_times = uniform_times(intervals, args.frames, offset)
                images, _, _ = frames_at_times(Path(row["video_path"]), sample_times)
                scores[name].append(binary_score(model, canvas(images)))
                times[name].append(sample_times.tolist())
        margins = [a - b for a, b in zip(scores["inside"], scores["outside"])]
        support = all(value > 0 for value in margins)
        output = {
            "dataset": key[0], "video_id": key[1], "proposal": [start, end],
            "scores": scores, "margins": margins,
            "warrant": "support" if support else "abstain", "times": times,
            "config": config,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
        print(json.dumps({"i": index, "n": len(selected), "key": key,
                          "margins": margins, "warrant": output["warrant"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
