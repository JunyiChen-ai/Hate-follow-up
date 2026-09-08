#!/usr/bin/env python3
"""Training-free MLLM semantic-state evidence on a visual temporal canvas."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.idea_discovery.run_counterfactual_evidence import rows, transcript_rows, ecdf
from scripts.idea_discovery.run_visual_temporal_canvas import (
    dense_bins, infer, make_canvas, timed_text, valid_chunks, video_frames)
from scripts.label_free_adapt.mechanisms import parse_json_object
from scripts.label_free_adapt.schema import Prediction, append_jsonl, curve_to_intervals

COHORT = ROOT / "results/idea_discovery/paradigm_adapt/stage_b_32_clean.jsonl"
T3AL = ROOT / "results/idea_discovery/paradigm_adapt/stage_b_t3al_clean"
STATES = {
    "ASSERTED_HATE", "IMPLIED_HATE", "MENTIONED_HATE", "COUNTERSPEECH",
    "OTHER_HOSTILITY", "BENIGN", "UNCLEAR"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--t3al-curves", type=Path, default=T3AL)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prompt-mode", choices=("binwise", "structure-first"),
                        default="binwise")
    args = parser.parse_args()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True,
                                               max_pixels=896 * 800)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="cuda:0", local_files_only=True,
        attn_implementation="sdpa").eval()
    chunks = transcript_rows()
    cohort = rows(args.cohort)
    if args.limit:
        cohort = cohort[:args.limit]
    prior = rows(args.out) if args.out.exists() else []
    done = {(r["dataset"], r["video_id"]) for r in prior}
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for row in cohort:
        dataset, video_id = row["dataset"], row["video_id"]
        if (dataset, video_id) in done:
            continue
        duration = float(row["duration"])
        base = np.load(args.t3al_curves / dataset / f"{video_id}.npy")
        video_chunks, rejected = valid_chunks(chunks.get((dataset, video_id), []), duration)
        speech = timed_text(video_chunks, duration)
        speech_records = [{"bin": i, "start": round(i * duration / 16, 2),
                           "end": round((i + 1) * duration / 16, 2), "text": text}
                          for i, text in enumerate(speech) if text]
        images, times, fallback = video_frames(row["video_path"], duration)
        canvas = make_canvas(images, times, base, "T3AL", 16)
        common = (
            "The image is one chronologically ordered 16-bin video canvas. The blue line is a noisy "
            "label-free visual proposal, never ground truth. Timestamp-aligned speech is "
            + json.dumps(speech_records, ensure_ascii=False) + ". "
            "For every bin infer the AUTHOR/SPEAKER COMMUNICATIVE STATE using visual content, speech, "
            "and neighboring bins. Allowed states: ASSERTED_HATE (endorsed attack, dehumanization, threat, "
            "exclusion or hateful stereotype toward a protected person/group), IMPLIED_HATE (the modalities "
            "jointly convey such endorsed hate indirectly), MENTIONED_HATE (quoted/reported hateful material "
            "without endorsement), COUNTERSPEECH (condemnation or rebuttal), OTHER_HOSTILITY (hostile but not "
            "protected-target hate), BENIGN, UNCLEAR. Also give a graded 0..100 evidence score for content that "
            "the video itself asserts or endorses as protected-target hate. Quotation and counterspeech must be "
            "lower than otherwise identical asserted hate, but do not force scores to zero when authorship is "
            "genuinely ambiguous. Each bin must be judged at its own timestamp. ")
        if args.prompt_mode == "binwise":
            prompt = common + ("Return compact JSON only: {\"states\":[16 state strings in bin order],"
                "\"scores\":[16 integers in bin order]}. Both arrays must have exactly 16 entries.")
        else:
            prompt = common + (
                "First partition the complete 0..15 timeline wherever speaker, scene, topic, communicative "
                "stance, or the presence of the target claim changes. Only after fixing this event structure, "
                "assign one state and score to each segment. Segments must be contiguous, non-overlapping, "
                "cover every bin exactly once, start at 0 and end at 15. Do not merge bins merely because the "
                "video has one overall label. Return compact JSON only: {\"segments\":[[0,3,\"BENIGN\",0],"
                "...]}. Each entry is [inclusive_start,inclusive_end,state,score].")
        attempts = []
        parsed = None
        for attempt in range(2):
            suffix = "" if attempt == 0 else " Previous JSON was invalid; output exactly the required schema."
            raw = infer(canvas, prompt + suffix, processor, model)
            attempts.append(raw)
            try:
                obj = parse_json_object(raw)
                if args.prompt_mode == "binwise":
                    states = obj.get("states", [])
                    scores = obj.get("scores", [])
                else:
                    segments = obj.get("segments", [])
                    starts = [int(segment[0]) for segment in segments]
                    if not starts or starts[0] != 0 or any(
                            right <= left for left, right in zip(starts, starts[1:])):
                        raise ValueError("invalid changepoints")
                    states, scores = [], []
                    for index, segment in enumerate(segments):
                        _, _, state, score = segment
                        stop = starts[index + 1] if index + 1 < len(starts) else 16
                        if stop > 16:
                            raise ValueError("changepoint beyond timeline")
                        width = stop - starts[index]
                        states.extend([state] * width)
                        scores.extend([score] * width)
                valid = (len(states) == 16 and len(scores) == 16 and
                         all(state in STATES for state in states) and
                         all(isinstance(score, (int, float)) and math.isfinite(float(score))
                             and 0 <= float(score) <= 100 for score in scores))
                if valid:
                    parsed = states, np.asarray(scores, dtype=float) / 100
                    break
            except (KeyError, TypeError, ValueError):
                pass
        if parsed is None:
            raise RuntimeError(f"invalid state output for {dataset}/{video_id}: {attempts!r}")
        states, bin_scores = parsed
        curve = dense_bins(bin_scores, len(base))
        append_jsonl(args.out, Prediction(
            f"canvas_evidence_states_{args.prompt_mode}_v2", dataset, video_id, duration,
            score_curve=curve.tolist(), intervals=curve_to_intervals(curve, duration, 0.5),
            calls=len(attempts), modality_evidence={"states": states,
                "bin_scores": bin_scores.tolist(), "invalid_asr_spans_rejected": rejected,
                "ffmpeg_fallback_frames": fallback},
            raw={"gt_access": False, "responses": attempts, "code_sha256": code_hash,
                 "state_vocabulary": sorted(STATES), "prompt_mode": args.prompt_mode}))
        print(json.dumps({"dataset": dataset, "video_id": video_id,
                          "states": states, "scores": bin_scores.tolist()}), flush=True)
    expected = {(r["dataset"], r["video_id"]) for r in cohort}
    observed = {(r["dataset"], r["video_id"]) for r in rows(args.out)}
    if not expected <= observed:
        raise RuntimeError(f"incomplete output: {len(expected - observed)} missing")


if __name__ == "__main__":
    main()
