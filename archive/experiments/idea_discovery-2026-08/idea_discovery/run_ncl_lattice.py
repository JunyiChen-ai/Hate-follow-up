#!/usr/bin/env python3
"""Single-space evidence-role lattice for Null-Competitive Localization."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_melt import MLLM, sanitized_cohort
from scripts.idea_discovery.run_ncl_tribunal import (
    POLICY, PHASES, carrier, complement_intervals, load, overlap_text, uniform_times,
)
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times, valid_chunks


LABELS = {
    "A": "EMPTY_OR_INSUFFICIENT",
    "B": "VISUAL_SUFFICIENT",
    "C": "TEXT_SUFFICIENT",
    "D": "COMPOSITIONAL_JOINT",
    "E": "BENIGN_EXPLAINED",
}


def lattice_prompt(target_text: str, context_text: str) -> str:
    return (
        f"{POLICY} Red T cells and TARGET transcript describe the candidate interval; "
        "C cells and context transcript only help determine stance. "
        f"TARGET transcript={json.dumps(target_text, ensure_ascii=False)}. "
        f"Context transcript={json.dumps(context_text, ensure_ascii=False)}. "
        "Select the single best hypothesis about the TARGET interval. "
        "A=EMPTY_OR_INSUFFICIENT: no complete asserted/endorsed hateful event is supported. "
        "B=VISUAL_SUFFICIENT: visual evidence alone establishes such an event. "
        "C=TEXT_SUFFICIENT: target speech alone establishes such an event. "
        "D=COMPOSITIONAL_JOINT: neither channel alone is sufficient but their aligned "
        "combination establishes the event. "
        "E=BENIGN_EXPLAINED: apparent hostility is affirmatively quotation, reporting, "
        "counterspeech, satire, or condemnation. Do not use E for mere missing evidence. "
        "Choose exactly one next letter A, B, C, D, or E. Answer one letter only:"
    )


@torch.inference_mode()
def score(model: MLLM, images, prompts, batch_size: int) -> np.ndarray:
    output = []
    for start in range(0, len(images), batch_size):
        ims = images[start:start + batch_size]
        prs = prompts[start:start + batch_size]
        texts = []
        for image, prompt in zip(ims, prs):
            message = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            texts.append(model.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True))
        inputs = model.processor(text=texts, images=ims, padding=True,
                                 return_tensors="pt").to(model.model.device)
        logits = model.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        choice = logits[:, model.phase_token_ids]
        output.extend(choice.cpu().tolist())
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
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    if args.out.exists() or partial.exists():
        raise RuntimeError(f"refusing existing output/partial: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cohort = sanitized_cohort(args.cohort)
    if args.limit:
        cohort = cohort[:args.limit]
    proposals, all_chunks = load(args.proposals), transcript_rows()
    model = MLLM(args.model)
    config = {
        "version": "ncl_single_space_lattice_v1", "model": args.model,
        "labels": LABELS, "phases": list(PHASES),
        "cohort_sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
        "proposals_sha256": hashlib.sha256(args.proposals.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gt_access": False,
    }
    with partial.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(cohort, 1):
            key = (row["dataset"], row["video_id"]); duration = float(row["duration"])
            clean, rejected = valid_chunks(all_chunks.get(key, []), duration)
            meta, images, prompts = [], [], []
            for proposal in proposals[key]["proposals"]:
                a, b = float(proposal["start"]), float(proposal["end"])
                outside = complement_intervals(a, b, duration)
                if not outside:
                    width = max(duration / 16, 1e-3)
                    outside = [(0.0, min(width, duration)),
                               (max(0.0, duration - width), duration)]
                target_text = overlap_text(clean, a, b)
                context_text = " ".join(overlap_text(clean, x, y, 800)
                                        for x, y in outside)[:1600]
                for phase in PHASES:
                    tt = uniform_times([(a, b)], 4, phase)
                    ct = uniform_times(outside, 4, phase)
                    tf, _, t_fb = frames_at_times(Path(row["video_path"]), tt)
                    cf, _, c_fb = frames_at_times(Path(row["video_path"]), ct)
                    images.append(carrier(tf, cf, tt, ct))
                    prompts.append(lattice_prompt(target_text, context_text))
                    meta.append({"rank": int(proposal["rank"]), "start": a, "end": b,
                                 "proposal_logit": float(proposal["logit"]), "phase": phase,
                                 "target_text_chars": len(target_text),
                                 "context_text_chars": len(context_text),
                                 "ffmpeg_fallback_frames": t_fb + c_fb})
            values = score(model, images, prompts, args.batch_size)
            observations = []
            for item, logits in zip(meta, values):
                winner = "ABCDE"[int(np.argmax(logits))]
                observations.append({**item, "logits": dict(zip("ABCDE", map(float, logits))),
                                     "winner": winner, "hypothesis": LABELS[winner]})
            record = {"dataset": key[0], "video_id": key[1], "duration": duration,
                      "observations": observations,
                      "invalid_asr_spans_rejected": rejected, "config": config}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush()
            print(json.dumps({"i": index, "n": len(cohort), "key": key,
                              "winners": {label: sum(x["winner"] == label for x in observations)
                                          for label in "ABCDE"},
                              "batch_forwards": model.calls}), flush=True)
    partial.replace(args.out)
    print(json.dumps({"videos": len(cohort), "out": str(args.out),
                      "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                      "batch_forwards": model.calls}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
