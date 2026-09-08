#!/usr/bin/env python3
"""Counterfactual Role-Binding Localization from a cached global relation.

M1 outputs are shared with MELT.  This pilot replaces decorative lifecycle
tokens with task-specific competing explanations of the same local evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import ecdf, transcript_rows
from scripts.idea_discovery.run_melt import MLLM, dense_bins, sanitized_cohort
from scripts.idea_discovery.run_visual_temporal_canvas import (
    frames_at_times, make_canvas, timed_text, valid_chunks)
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl

CHOICES = ("FULL", "STANCE_FLIP", "ROLE_MISMATCH", "ABSENT", "UNKNOWN")
GENERIC_CHOICES = ("HATE", "OUT", "UNKNOWN")
BINARY_RELATION_CHOICES = ("FULL", "NOT_FULL", "UNKNOWN")


@torch.inference_mode()
def binding_logits(model: MLLM, canvas, speech_records, relation, nbins=16,
                   modality="joint"):
    texts, images = [], []
    policy = (
        "Hate requires an asserted or endorsed attack, dehumanization, threat, exclusion, "
        "or hateful stereotype directed at a protected or socially salient group. Generic "
        "violence, personal disputes, reporting, quotation, satire, counterspeech and "
        "condemnation are not the same hateful relation."
    )
    for i in range(nbins):
        marked = (canvas.copy() if modality != "transcript" else
                  Image.new("RGB", canvas.size, (127, 127, 127)))
        draw = ImageDraw.Draw(marked)
        x, y = (i % 4) * 224, (i // 4) * 156
        draw.rectangle((x + 2, y + 2, x + 221, y + 153), outline="red", width=6)
        records = speech_records
        if modality == "shifted":
            records = [{**r, "bin": (int(r["bin"]) + 8) % nbins} for r in speech_records]
        local = ([] if modality == "visual" else
                 [r for r in records if abs(int(r["bin"]) - i) <= 1])
        prompt = (
            f"{policy} Fixed global relation={json.dumps(relation, ensure_ascii=False)}. "
            f"Aligned local speech={json.dumps(local, ensure_ascii=False)}. The red bin is "
            "the only time unit to classify; other frames are stance context. Which competing "
            "explanation best fits the red bin? A=FULL: this exact source→hostile-act→target "
            "relation is locally present and asserted/endorsed. B=STANCE_FLIP: the relation "
            "is mentioned but quoted, reported, satirized, countered, or condemned. "
            "C=ROLE_MISMATCH: harmful content exists but source, act, protected target, or "
            "directedness differs. D=ABSENT: this relation is absent. E=UNKNOWN: evidence is "
            "insufficient or modalities conflict. Choose one letter only:"
        )
        msg = [{"role": "user", "content": [
            {"type": "image", "image": marked}, {"type": "text", "text": prompt}]}]
        texts.append(model.processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True))
        images.append(marked)
    inp = model.processor(text=texts, images=images, padding=True,
                          return_tensors="pt").to(model.model.device)
    logits = model.model(**inp, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    model.calls += 1
    return torch.softmax(logits[:, model.phase_token_ids], dim=-1).cpu().numpy()


@torch.inference_mode()
def generic_logits(model: MLLM, canvas, speech_records, nbins=16):
    """Compute-matched generic hate control without a fixed event identity."""
    texts, images = [], []
    for i in range(nbins):
        marked = canvas.copy()
        draw = ImageDraw.Draw(marked)
        x, y = (i % 4) * 224, (i // 4) * 156
        draw.rectangle((x + 2, y + 2, x + 221, y + 153), outline="red", width=6)
        local = [r for r in speech_records if abs(int(r["bin"]) - i) <= 1]
        prompt = (
            "Classify only the red video bin using its visual evidence and aligned local "
            f"speech={json.dumps(local, ensure_ascii=False)}. HATE means asserted or endorsed "
            "hostility against a protected or social group; quotation, reporting, satire, "
            "counterspeech and condemnation are OUT. Choose exactly one: A=HATE, B=OUT, "
            "C=UNKNOWN. Answer one letter only:"
        )
        msg = [{"role": "user", "content": [
            {"type": "image", "image": marked}, {"type": "text", "text": prompt}]}]
        texts.append(model.processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True))
        images.append(marked)
    inp = model.processor(text=texts, images=images, padding=True,
                          return_tensors="pt").to(model.model.device)
    logits = model.model(**inp, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    model.calls += 1
    return torch.softmax(logits[:, model.phase_token_ids[:3]], dim=-1).cpu().numpy()


@torch.inference_mode()
def binary_relation_logits(model: MLLM, canvas, speech_records, relation, nbins=16):
    """Compute-matched control collapsing all typed relation failures."""
    texts, images = [], []
    for i in range(nbins):
        marked=canvas.copy(); draw=ImageDraw.Draw(marked); x,y=(i%4)*224,(i//4)*156
        draw.rectangle((x+2,y+2,x+221,y+153),outline="red",width=6)
        local=[r for r in speech_records if abs(int(r["bin"])-i)<=1]
        prompt=(f"Fixed relation={json.dumps(relation,ensure_ascii=False)}. Aligned local "
          f"speech={json.dumps(local,ensure_ascii=False)}. For only the red bin, choose A=FULL "
          "if this exact source-act-target-stance relation is completely supported; B=NOT_FULL "
          "if it is absent or any role/stance is wrong; C=UNKNOWN if evidence is insufficient. "
          "Answer one letter only:")
        msg=[{"role":"user","content":[{"type":"image","image":marked},{"type":"text","text":prompt}]}]
        texts.append(model.processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True));images.append(marked)
    inp=model.processor(text=texts,images=images,padding=True,return_tensors="pt").to(model.model.device)
    logits=model.model(**inp,use_cache=False,logits_to_keep=1).logits[:,-1].float();model.calls+=1
    return torch.softmax(logits[:,model.phase_token_ids[:3]],dim=-1).cpu().numpy()


def binding_decode(scores: np.ndarray, duration: float):
    eps = 1e-8
    margin = np.log(np.clip(scores[:, 0], eps, 1)) - np.log(
        np.clip(scores[:, 1:].max(1), eps, 1))
    active = margin > 0
    bounds = np.flatnonzero(np.diff(np.r_[False, active, False])).reshape(-1, 2)
    intervals = [Interval(a / len(active) * duration, b / len(active) * duration,
                          float(scores[a:b, 0].mean())) for a, b in bounds]
    return intervals, margin, [[int(a), int(b)] for a, b in bounds]


def generic_decode(scores: np.ndarray, duration: float):
    eps = 1e-8
    margin = np.log(np.clip(scores[:, 0], eps, 1)) - np.log(
        np.clip(scores[:, 1:].max(1), eps, 1))
    active = margin > 0
    bounds = np.flatnonzero(np.diff(np.r_[False, active, False])).reshape(-1, 2)
    intervals = [Interval(a / len(active) * duration, b / len(active) * duration,
                          float(scores[a:b, 0].mean())) for a, b in bounds]
    return intervals, margin, [[int(a), int(b)] for a, b in bounds]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--cached-m1", type=Path, required=True)
    ap.add_argument("--t3al-curves", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source-method", default="melt_adaptive")
    ap.add_argument("--binding-space", choices=("role", "generic", "binary_relation"), default="role")
    ap.add_argument("--modality", choices=("joint", "visual", "transcript", "shifted"),
                    default="joint")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    cohort = {(r["dataset"], r["video_id"]): r for r in sanitized_cohort(args.cohort)}
    cached = {}
    for line in args.cached_m1.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("method") == args.source_method:
            cached[(row["dataset"], row["video_id"])] = row
    asr = transcript_rows()
    model = MLLM(args.model)
    for key, prior in cached.items():
        row = cohort[key]
        duration = float(row["duration"])
        if prior.get("error"):
            error_method = {"role":"role_binding","generic":"generic_hate_field",
                            "binary_relation":"binary_relation_field"}[args.binding_space]
            append_jsonl(args.out, Prediction(
                error_method, key[0], key[1], duration,
                error=f"cached_m1_failure: {prior['error']}",
                raw={"cached_m1": str(args.cached_m1)}))
            continue
        chunks, invalid = valid_chunks(asr.get(key, []), duration)
        speech = timed_text(chunks, duration)
        records = [{"bin": i, "start": round(i * duration / 16, 3),
                    "end": round((i + 1) * duration / 16, 3), "text": text}
                   for i, text in enumerate(speech) if text]
        curve = np.load(args.t3al_curves / key[0] / f"{key[1]}.npy")
        try:
            images, times, fallback = frames_at_times(
                row["video_path"], (np.arange(16) + .5) / 16 * duration)
        except Exception as exc:
            error_method = {"role":"role_binding","generic":"generic_hate_field",
                            "binary_relation":"binary_relation_field"}[args.binding_space]
            append_jsonl(args.out, Prediction(
                error_method, key[0], key[1], duration,
                error=f"media_decode_failure: {type(exc).__name__}: {exc}"))
            continue
        canvas = make_canvas(images, times, curve, "T3AL")
        relation = prior["modality_evidence"]["event_relation"]
        before = model.calls
        if args.binding_space == "role":
            scores = binding_logits(model, canvas, records, relation,
                                    modality=args.modality)
            intervals, margin, paths = binding_decode(scores, duration)
            choices, method = CHOICES, "role_binding"
        elif args.binding_space == "generic":
            scores = generic_logits(model, canvas, records)
            intervals, margin, paths = generic_decode(scores, duration)
            choices, method = GENERIC_CHOICES, "generic_hate_field"
        else:
            scores = binary_relation_logits(model, canvas, records, relation)
            intervals, margin, paths = generic_decode(scores, duration)
            choices, method = BINARY_RELATION_CHOICES, "binary_relation_field"
        if args.binding_space in {"role","binary_relation"} and relation["stance"] not in {"endorsement", "ambiguous"}:
            scores[:] = 0
            if args.binding_space=="role":
                scores[:, CHOICES.index("ABSENT")]=1;intervals,margin,paths=binding_decode(scores,duration)
            else:
                scores[:, BINARY_RELATION_CHOICES.index("NOT_FULL")]=1;intervals,margin,paths=generic_decode(scores,duration)
        dense = dense_bins(scores[:, 0], len(curve))
        append_jsonl(args.out, Prediction(
            method, key[0], key[1], duration, score_curve=dense.tolist(),
            intervals=intervals, calls=model.calls - before,
            modality_evidence={"event_relation": relation,
                               "binding_choices": choices,
                               "binding_scores": scores.tolist(),
                               "binding_margin": margin.tolist(),
                               "paths": paths,
                               "invalid_asr_spans_rejected": invalid,
                               "ffmpeg_fallback_frames": fallback},
            raw={"cached_m1": str(args.cached_m1),
                 "m2": {"role":"typed_relation_completion_v1",
                        "generic":"generic_hate_control_v1",
                        "binary_relation":"binary_relation_control_v1"}[args.binding_space],
                 "modality": args.modality}))
        print(json.dumps({"dataset": key[0], "video_id": key[1], "paths": paths}),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
