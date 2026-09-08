#!/usr/bin/env python3
"""Training-free pilots for A02--A07 on the common local MLLM backbone."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .mechanisms import (GENERIC_QUERY, QUERY_FACETS, boundary_distributions,
                         consistency_curve, numbered_timeline_prompt,
                         parse_numbered_intervals)
from .run_numpro import load_done, sample_numbered_frames
from .schema import Prediction, append_jsonl, intervals_to_curve

STRATEGIES = ("A02", "A03", "A04", "A05", "A06", "A07")


def strategy_prompt(strategy: str, nframes: int, duration: float, transcript: str,
                    suffix: str = "") -> str:
    base = numbered_timeline_prompt(range(nframes), duration, transcript=transcript)
    additions = {
        "A02": " Use a visual-time chain of thought internally: scan the progress bar, highlight candidates, then verify their boundaries.",
        "A03": " Treat this as one consistency probe. Do not infer evidence absent from this input.",
        "A04": " This is the initial prediction before a separate reflection pass.",
        "A05": " Return every disjoint relevant segment; never bridge neutral gaps merely to produce one interval.",
        "A06": " Express boundary uncertainty through calibrated confidence; ambiguous edges should receive lower confidence.",
        "A07": " Check all query facets before grounding: " + "; ".join(QUERY_FACETS) + ".",
    }
    return base + additions[strategy] + suffix


@torch.no_grad()
def infer(frames, prompt, processor, model, max_new_tokens: int) -> str:
    content = [{"type": "image", "image": image} for image in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(output[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()


def run_strategy(strategy, row, frames, times, processor, model, max_new_tokens):
    duration = float(row["duration"]); transcript = row.get("transcript", "")
    raw, samples = [], []
    if strategy == "A03":
        probes = [
            (frames, times, ""),
            (frames, times, " Re-evaluate using semantically equivalent wording: locate targeted group-directed harmful expression."),
            (frames[::2], times[::2], " This input is temporally sparse; avoid filling unseen gaps."),
        ]
    elif strategy == "A06":
        probes = [
            (frames, times, " Prefer the tightest evidence-supported boundaries."),
            (frames, times, " Include uncertain boundary context but lower its confidence."),
            (frames[1::2] or frames, times[1::2] or times, " Estimate boundaries from an offset temporal sample."),
        ]
    else:
        probes = [(frames, times, "")]
    for probe_frames, probe_times, suffix in probes:
        prompt = strategy_prompt(strategy, len(probe_frames), duration, transcript, suffix)
        response = infer(probe_frames, prompt, processor, model, max_new_tokens)
        intervals, evidence = parse_numbered_intervals(response, probe_times, duration)
        raw.append({"response": response, "frame_times": probe_times, "evidence": evidence})
        samples.append(intervals)

    if strategy == "A04":
        initial = samples[0]
        correction = strategy_prompt("A04", len(frames), duration, transcript,
            " Audit this initial prediction: " + json.dumps([x.as_list() for x in initial]) +
            ". Inspect evidence immediately inside and outside every edge, recover missed disjoint events, and return corrected JSON.")
        response = infer(frames, correction, processor, model, max_new_tokens)
        corrected, evidence = parse_numbered_intervals(response, times, duration)
        raw.append({"response": response, "frame_times": times, "evidence": evidence})
        samples.append(corrected)
        final_curve = intervals_to_curve(corrected, duration)
        final_intervals = corrected
    elif strategy == "A03":
        curves = [intervals_to_curve(x, duration) for x in samples]
        final_curve = consistency_curve(curves)
        final_intervals = samples[0]
    elif strategy == "A06":
        curves = [intervals_to_curve(x, duration) for x in samples]
        final_curve = [sum(values) / len(values) for values in zip(*curves)]
        final_intervals = samples[0]
        starts, ends = boundary_distributions(samples, duration)
        raw.append({"start_distribution": starts, "end_distribution": ends})
    else:
        final_intervals = samples[0]
        final_curve = intervals_to_curve(final_intervals, duration)
    evidence = {"probes": len(samples), "facets": list(QUERY_FACETS) if strategy == "A07" else []}
    return final_intervals, final_curve, evidence, raw


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=STRATEGIES + ("all",))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--nframes", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20250819)
    args = ap.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.manual_seed(args.seed)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", local_files_only=True).eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=448 * 448,
                                               local_files_only=True)
    rows = [json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
    strategies = STRATEGIES if args.strategy == "all" else (args.strategy,)
    if args.limit:
        rows = rows[:args.limit]
    done = load_done(Path(args.out))
    for row in rows:
        frames = times = None
        for strategy in strategies:
            method = strategy
            if (row["dataset"], row["video_id"], method) in done:
                continue
            prediction = Prediction(method, row["dataset"], row["video_id"],
                                    float(row["duration"]), seed=args.seed)
            try:
                if frames is None:
                    frames, times = sample_numbered_frames(row["video_path"], args.nframes)
                (prediction.intervals, prediction.score_curve,
                 prediction.modality_evidence, prediction.raw) = run_strategy(
                    strategy, row, frames, times, processor, model, args.max_new_tokens)
            except Exception as exc:
                prediction.error = f"{type(exc).__name__}: {exc}"
            append_jsonl(Path(args.out), prediction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
