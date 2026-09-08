#!/usr/bin/env python3
"""Label-blind pilot for Null-Competitive Localization (NCL).

This runner emits proposal-level evidence only.  It never reads ground truth and
does not make the final EMPTY/non-empty projection.  Keeping evidence collection
separate lets us freeze and hash the semantic observations before evaluating any
decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_melt import MLLM, sanitized_cohort
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times, valid_chunks


POLICY = (
    "Hateful content is an asserted or endorsed attack, dehumanization, threat, "
    "exclusion, or hateful stereotype directed at a person or protected/social "
    "group. Mere profanity is insufficient. Quotation, neutral reporting, "
    "counterspeech, satire, and condemnation are not hateful unless the speaker "
    "endorses the hostility."
)
PHASES = (0.25, 0.75)
ARMS = ("null", "visual", "text", "joint")
QUESTIONS = {
    "support": (
        "Does the TARGET interval contain sufficient evidence of one complete "
        "asserted or endorsed hateful event? Answer Yes or No only:"
    ),
    "explained": (
        "Does the evidence affirmatively establish that the apparent hostile "
        "material in the TARGET is quotation, neutral reporting, counterspeech, "
        "satire, or condemnation rather than endorsed hostility? Absence of hate "
        "evidence is not enough. Answer Yes or No only:"
    ),
}


def load(path: Path) -> dict[tuple[str, str], dict]:
    return {(row["dataset"], row["video_id"]): row
            for row in map(json.loads, path.open())}


def overlap_text(chunks: list[dict], start: float, end: float, cap: int = 1600) -> str:
    values = [str(row.get("text", "")) for row in chunks
              if max(start, float(row["span"][0])) < min(end, float(row["span"][1]))]
    return " ".join(values)[:cap]


def complement_intervals(start: float, end: float, duration: float) -> list[tuple[float, float]]:
    return [(a, b) for a, b in ((0.0, start), (end, duration)) if b - a > 1e-6]


def uniform_times(intervals: list[tuple[float, float]], n: int, phase: float) -> np.ndarray:
    lengths = np.asarray([b - a for a, b in intervals], float)
    if not len(lengths) or lengths.sum() <= 0:
        raise ValueError("cannot sample empty interval collection")
    positions = (np.arange(n) + phase) / n * lengths.sum()
    cumulative = np.cumsum(lengths)
    output = []
    for position in positions:
        index = min(len(intervals) - 1, int(np.searchsorted(cumulative, position, side="right")))
        prior = 0.0 if index == 0 else cumulative[index - 1]
        output.append(intervals[index][0] + position - prior)
    return np.asarray(output)


def carrier(target: list[Image.Image], context: list[Image.Image],
            target_times: np.ndarray, context_times: np.ndarray) -> Image.Image:
    images = context[:2] + target + context[2:]
    times = list(context_times[:2]) + list(target_times) + list(context_times[2:])
    out = Image.new("RGB", (len(images) * 112, 140), "white")
    draw = ImageDraw.Draw(out)
    for index, (image, when) in enumerate(zip(images, times)):
        x = index * 112
        out.paste(image.convert("RGB").resize((112, 112)), (x, 0))
        is_target = 2 <= index < 2 + len(target)
        draw.text((x + 3, 3), f"{'T' if is_target else 'C'} {when:.1f}s",
                  fill="yellow", stroke_width=2, stroke_fill="black")
        if is_target:
            draw.rectangle((x + 2, 2, x + 109, 109), outline="red", width=4)
    draw.text((4, 116), "T=TARGET evidence; C=chronological context", fill="black")
    return out


def prompt(arm: str, target_text: str, context_text: str, question: str) -> str:
    if arm == "null":
        media = (
            "No frames or transcript are supplied. This is the explicit no-media "
            "null hypothesis; do not imagine any event."
        )
    elif arm == "visual":
        media = "No transcript is supplied. Judge the shown visual evidence only."
    elif arm == "text":
        media = (f"TARGET transcript={json.dumps(target_text, ensure_ascii=False)}. "
                 f"Context transcript={json.dumps(context_text, ensure_ascii=False)}. "
                 "No frames are supplied. Judge the transcript evidence only.")
    else:
        media = (f"TARGET transcript={json.dumps(target_text, ensure_ascii=False)}. "
                 f"Context transcript={json.dumps(context_text, ensure_ascii=False)}. "
                 "Use timestamp-matched frames and transcript jointly.")
    return f"{POLICY} {media} {question}"


@torch.inference_mode()
def binary_logits(model: MLLM, images: list[Image.Image] | None,
                  prompts: list[str], batch_size: int) -> np.ndarray:
    output = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        batch_images = None if images is None else images[start:start + batch_size]
        texts = []
        for index, value in enumerate(batch_prompts):
            content = []
            if batch_images is not None:
                content.append({"type": "image", "image": batch_images[index]})
            content.append({"type": "text", "text": value})
            texts.append(model.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True))
        kwargs = {"text": texts, "padding": True, "return_tensors": "pt"}
        if batch_images is not None:
            kwargs["images"] = batch_images
        inputs = model.processor(**kwargs).to(model.model.device)
        logits = model.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        pair = logits[:, model.binary_token_ids]
        output.extend((pair[:, 0] - pair[:, 1]).cpu().tolist())
        model.calls += 1
    return np.asarray(output, float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--proposals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.out.exists() or args.out.with_suffix(args.out.suffix + ".partial").exists():
        raise RuntimeError(f"refusing existing output/partial: {args.out}")

    cohort = sanitized_cohort(args.cohort)
    if args.limit:
        cohort = cohort[:args.limit]
    proposals = load(args.proposals)
    chunks = transcript_rows()
    model = MLLM(args.model)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "version": "ncl_tribunal_v1", "model": args.model,
        "phases": list(PHASES), "arms": list(ARMS), "questions": QUESTIONS,
        "cohort_sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
        "proposals_sha256": hashlib.sha256(args.proposals.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gt_access": False,
    }
    try:
        with partial.open("w", encoding="utf-8") as handle:
            for video_index, row in enumerate(cohort, 1):
                key = (row["dataset"], row["video_id"])
                bank = proposals[key]["proposals"]
                duration = float(row["duration"])
                clean, rejected = valid_chunks(chunks.get(key, []), duration)
                observations = []
                for proposal in bank:
                    start, end = float(proposal["start"]), float(proposal["end"])
                    outside = complement_intervals(start, end, duration)
                    if not outside:
                        # A full-video proposal has no natural complement. Use the
                        # two endpoint neighborhoods as context, without synthetic masks.
                        width = max(duration / 16, 1e-3)
                        outside = [(0.0, min(width, duration)),
                                   (max(0.0, duration - width), duration)]
                    target_text = overlap_text(clean, start, end)
                    context_text = " ".join(
                        overlap_text(clean, a, b, 800) for a, b in outside)[:1600]
                    for phase in PHASES:
                        target_times = uniform_times([(start, end)], 4, phase)
                        context_times = uniform_times(outside, 4, phase)
                        target_frames, _, target_fallback = frames_at_times(
                            Path(row["video_path"]), target_times)
                        context_frames, _, context_fallback = frames_at_times(
                            Path(row["video_path"]), context_times)
                        image = carrier(target_frames, context_frames,
                                        target_times, context_times)
                        scores = {}
                        for arm in ARMS:
                            prompts = [prompt(arm, target_text, context_text, QUESTIONS[name])
                                       for name in QUESTIONS]
                            values = binary_logits(
                                model, None if arm in {"null", "text"} else [image] * len(prompts),
                                prompts, args.batch_size)
                            scores[arm] = dict(zip(QUESTIONS, map(float, values)))
                        observations.append({
                            "rank": int(proposal["rank"]), "start": start, "end": end,
                            "proposal_logit": float(proposal["logit"]), "phase": phase,
                            "scores": scores, "target_text_chars": len(target_text),
                            "context_text_chars": len(context_text),
                            "ffmpeg_fallback_frames": target_fallback + context_fallback,
                        })
                record = {
                    "dataset": key[0], "video_id": key[1], "duration": duration,
                    "observations": observations, "invalid_asr_spans_rejected": rejected,
                    "config": config,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps({"i": video_index, "n": len(cohort), "key": key,
                                  "observations": len(observations),
                                  "batch_forwards": model.calls}), flush=True)
        partial.replace(args.out)
    except Exception:
        raise
    print(json.dumps({"videos": len(cohort), "out": str(args.out),
                      "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                      "batch_forwards": model.calls}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
