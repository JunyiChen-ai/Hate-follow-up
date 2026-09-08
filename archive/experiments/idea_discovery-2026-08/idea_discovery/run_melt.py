#!/usr/bin/env python3
"""MELT sanity runner: label-free lifecycle parsing and boundary certification.

Inference consumes only sanitized media metadata, frozen proposal curves and the
strict ASR allowlist. Ground truth is intentionally unavailable to this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import ecdf, rows, transcript_rows
from scripts.idea_discovery.run_visual_temporal_canvas import (
    frames_at_times,
    make_canvas,
    timed_text,
    valid_chunks,
)
from scripts.label_free_adapt.mechanisms import parse_json_object
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl

PHASES = ("OUT", "ENTER", "SUPPORT", "EXIT", "UNKNOWN")
EVENT_PHASES = ("ENTER", "SUPPORT", "EXIT")
DEFAULT_COHORT = ROOT / "results/idea_discovery/paradigm_adapt/stage_a_8_clean.jsonl"
DEFAULT_T3AL = ROOT / "results/idea_discovery/paradigm_adapt/stage_a_t3al_clean"
COHORT_FIELDS = {"dataset", "duration", "transcript", "video_id", "video_path"}


def dense_bins(values: np.ndarray, n: int) -> np.ndarray:
    ids = np.minimum(len(values) - 1, ((np.arange(n) + .5) / n * len(values)).astype(int))
    return np.asarray(values, float)[ids]


def sanitized_cohort(path):
    output = rows(path)
    for index, row in enumerate(output):
        extra = set(row) - COHORT_FIELDS
        missing = {"dataset", "duration", "video_id", "video_path"} - set(row)
        if extra or missing:
            raise RuntimeError(f"unsafe cohort row {index}: extra={extra}, missing={missing}")
    return output


def normalize_phase_rows(value, nbins=16):
    """Validate model output and convert each row to a probability simplex."""
    if isinstance(value, dict) and set(value) == set(PHASES):
        columns = []
        for phase in PHASES:
            column = value[phase]
            if not isinstance(column, list) or len(column) != nbins:
                raise ValueError(f"phase {phase} must be a length-{nbins} list")
            columns.append(column)
        value = {str(i): {p: columns[j][i] for j, p in enumerate(PHASES)}
                 for i in range(nbins)}
    if not isinstance(value, dict) or set(value) != {str(i) for i in range(nbins)}:
        raise ValueError("phase_scores must contain exactly the numbered bins")
    output = np.zeros((nbins, len(PHASES)), float)
    for i in range(nbins):
        row = value[str(i)]
        if not isinstance(row, dict) or set(row) != set(PHASES):
            raise ValueError(f"bin {i} must contain exactly {PHASES}")
        vals = np.asarray([row[p] for p in PHASES], float)
        if not np.isfinite(vals).all() or (vals < 0).any() or vals.sum() <= 0:
            raise ValueError(f"invalid phase scores in bin {i}")
        output[i] = vals / vals.sum()
    return output


def phase_intervals(phase: np.ndarray, duration: float):
    active = np.isin(np.asarray(PHASES)[phase.argmax(1)], EVENT_PHASES)
    bounds = np.flatnonzero(np.diff(np.r_[False, active, False])).reshape(-1, 2)
    return [Interval(a / len(active) * duration, b / len(active) * duration,
                     float(phase[a:b, 1:4].sum(1).mean())) for a, b in bounds]


def lifecycle_decode(phase: np.ndarray, duration: float, evidence_anchors=None):
    """Decode cited connected event lifecycles against OUT/UNKNOWN background."""
    eps = 1e-6
    n = len(phase)
    event_prob = phase[:, 1:4].sum(1)
    event_state_prob = phase[:, 1:4].max(1)
    background_prob = np.maximum(phase[:, PHASES.index("OUT")],
                                 phase[:, PHASES.index("UNKNOWN")])
    margin = np.log(np.clip(event_state_prob, eps, 1.0)) - np.log(
        np.clip(background_prob, eps, 1.0))
    active = margin > 0
    bounds = np.flatnonzero(np.diff(np.r_[False, active, False])).reshape(-1, 2)
    components = []
    for start, end in bounds:
        if evidence_anchors and not any(start - 1 <= a <= end for a in evidence_anchors):
            continue
        components.append((int(start), int(end), float(margin[start:end].sum())))
    if not components:
        return [], event_prob, {"path": "all_OUT", "margin": 0.0}
    intervals = [Interval(start / n * duration, end / n * duration,
                          float(event_prob[start:end].mean()))
                 for start, end, _ in components]
    return intervals, event_prob, {
        "paths": [[a, b] for a, b, _ in components],
        "margins": [m for _, _, m in components],
        "margin": float(sum(m for _, _, m in components)),
    }


def parse_relation(value):
    keys = ("source", "hostile_act", "protected_target", "stance")
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"event_relation must contain exactly {keys}")
    if value["stance"] not in {"endorsement", "quotation", "condemnation", "ambiguous", "none"}:
        raise ValueError("invalid stance")
    output = {}
    for key in keys:
        if value[key] is None or not str(value[key]).strip():
            raise ValueError(f"empty relation field: {key}")
        output[key] = str(value[key]).strip()[:500]
    return output


def evidence_bins(value, nbins=16):
    if not isinstance(value, list):
        raise ValueError("evidence_bins must be a list")
    out = sorted({int(x) for x in value if isinstance(x, (int, float)) and int(x) == x})
    if any(x < 0 or x >= nbins for x in out):
        raise ValueError("evidence bin outside canvas")
    return out


class MLLM:
    def __init__(self, model_name):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=True, max_pixels=896 * 800)
        # Batched scorers below read logits at position -1. Qwen defaults to
        # right padding, where -1 is a pad position for shorter prompts. Left
        # padding makes -1 the true next-token position for every batch item.
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="cuda:0",
            local_files_only=True, attn_implementation="sdpa").eval()
        self.calls = 0
        self.phase_token_ids = self._single_token_ids(("A", "B", "C", "D", "E"))
        self.binary_token_ids = self._single_token_ids(("Yes", "No"))

    def _single_token_ids(self, labels):
        ids = []
        for label in labels:
            candidates = []
            for form in (label, " " + label):
                value = self.processor.tokenizer.encode(form, add_special_tokens=False)
                if len(value) == 1:
                    candidates.append(value[0])
            if not candidates:
                raise RuntimeError(f"no single token encoding for {label}")
            ids.append(candidates[0])
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"choice token collision: {ids}")
        return torch.tensor(ids, device=self.model.device)

    @torch.inference_mode()
    def infer(self, image, prompt, max_new_tokens=1200):
        msg = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        self.calls += 1
        return self.processor.batch_decode(
            out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    @torch.inference_mode()
    def phase_logits(self, canvas, speech_records, relation, nbins=16):
        policy = (
            "Hateful content is an asserted or endorsed attack, dehumanization, threat, "
            "exclusion, or hateful stereotype against a person or protected/social group; "
            "quotation, reporting, counterspeech, satire and condemnation are not endorsement."
        )
        texts, focused = [], []
        for i in range(nbins):
            marked = canvas.copy()
            draw = ImageDraw.Draw(marked)
            x, y = (i % 4) * 224, (i // 4) * 156
            draw.rectangle((x + 2, y + 2, x + 221, y + 153), outline="red", width=6)
            focused.append(marked)
            local_speech = [r for r in speech_records if abs(int(r["bin"]) - i) <= 1]
            prompt = (
                f"{policy} Fixed event relation={json.dumps(relation, ensure_ascii=False)}. "
                f"Only local speech for bin {i} and immediate neighbors="
                f"{json.dumps(local_speech, ensure_ascii=False)}. The red rectangle marks the "
                f"only bin being classified; other frames provide stance context only. "
                f"Classify numbered bin {i} for this same relation. Choose exactly one next "
                "letter: A=OUT (relation absent), B=ENTER (first establishment), C=SUPPORT "
                "(relation maintained), D=EXIT (last support before cessation), E=UNKNOWN. "
                "Answer one letter only:")
            msg = [{"role": "user", "content": [
                {"type": "image", "image": marked}, {"type": "text", "text": prompt}]}]
            texts.append(self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True))
        inp = self.processor(text=texts, images=focused, padding=True,
                             return_tensors="pt").to(self.model.device)
        logits = self.model(**inp, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        self.calls += 1
        return torch.softmax(logits[:, self.phase_token_ids], dim=-1).cpu().numpy()

    @torch.inference_mode()
    def binary_logits(self, images, prompts):
        texts = []
        for image, prompt in zip(images, prompts):
            msg = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            texts.append(self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True))
        inp = self.processor(text=texts, images=images, padding=True,
                             return_tensors="pt").to(self.model.device)
        logits = self.model(**inp, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        self.calls += 1
        pair = logits[:, self.binary_token_ids]
        return (pair[:, 0] - pair[:, 1]).cpu().numpy()


def relation_prompt(speech_records):
    return (
        "The image is a 16-bin numbered video timeline. Its blue curve is an uncertain "
        "label-free search hint, never evidence or an answer. Timestamp-aligned speech is "
        f"{json.dumps(speech_records, ensure_ascii=False)}. Hateful content is an asserted or "
        "endorsed attack, dehumanization, threat, exclusion, or hateful stereotype against a "
        "person or protected/social group. Quotation, reporting, counterspeech, satire and "
        "condemnation are not hateful unless hostility is endorsed. Parse one event relation. "
        "Parse the single most supported event relation and cite only numbered bins that "
        "materially support it. Unknown entities must be the literal string 'unknown', not "
        "invented. If no hateful relation exists, use stance='none' and describe the strongest "
        "benign/non-endorsed candidate relation. Return JSON only: "
        '{"event_relation":{"source":"...","hostile_act":"...",'
        '"protected_target":"...","stance":"endorsement|quotation|condemnation|ambiguous|none"},'
        '"evidence_bins":[integers]}.')


def candidate_cuts(interval, duration, fps=4.0, radius=1.5):
    start, end = interval.start, interval.end
    def around(center):
        lo = max(0, math.floor((center - radius) * fps) / fps)
        hi = min(duration, math.ceil((center + radius) * fps) / fps)
        return [round(x / fps, 3) for x in range(int(round(lo * fps)),
                                                 int(round(hi * fps)) + 1)]
    return around(start), around(end)


def records_in_span(chunks, start, end):
    output = []
    for row in chunks:
        a, b = map(float, row["span"])
        if start <= (a + b) / 2 < end and str(row.get("text", "")).strip():
            output.append({"start": round(a, 3), "end": round(b, 3),
                           "text": str(row["text"])})
    return output


def span_canvas(path, start, end, nframes=8):
    if end <= start:
        raise ValueError("empty span")
    times = np.linspace(start, end, nframes + 2)[1:-1]
    try:
        images, decoded_times, fallback = frames_at_times(path, times)
    except RuntimeError:
        # Container duration can exceed the last decodable frame by one 4-FPS
        # cell. Retry the same grid shifted inward by exactly one official cell.
        shifted = np.maximum(start, times - .25)
        images, decoded_times, fallback = frames_at_times(path, shifted)
    return make_canvas(images, decoded_times, None, nbins=len(images)), fallback


def relation_question(relation, transcript, condition):
    return (
        "Fixed relation=" + json.dumps(relation, ensure_ascii=False)
        + ". Evidence packet identifier=" + str(condition)
        + ". Timestamp-aligned speech=" + json.dumps(transcript, ensure_ascii=False)
        + ". Does the supplied visual and speech evidence entail this same asserted/endorsed "
          "source-hostile_act-protected_target-stance relation? Quotation, reporting, satire, "
          "counterspeech and condemnation do not entail endorsement. Answer Yes or No only:")


def refine_with_local_states(model, row, chunks, relation, coarse):
    """Select 4 FPS transitions from actual local evidence, never generated timestamps."""
    duration = float(row["duration"])
    starts, ends = candidate_cuts(coarse, duration)
    cuts = starts + [x for x in ends if x not in starts]
    images, prompts, fallback = [], [], 0
    radius = max(.5, min(2.0, duration / 32))
    for cut in cuts:
        a, b = max(0.0, cut - radius), min(duration, cut + radius)
        canvas, used_fallback = span_canvas(row["video_path"], a, b)
        fallback += used_fallback
        transcript = records_in_span(chunks, a, b)
        images.append(canvas)
        prompts.append(relation_question(
            relation, transcript, f"local window centered at exact cut {cut:.3f}s; "
            "judge whether the relation is active at the center timestamp"))
    logits = model.binary_logits(images, prompts)
    score = dict(zip(cuts, map(float, logits)))

    def onset(values, center):
        values = sorted(values)
        crossings = [x for i, x in enumerate(values) if score[x] > 0
                     and (i == 0 or score[values[i - 1]] <= 0)]
        return min(crossings, key=lambda x: abs(x - center)) if crossings else min(
            values, key=lambda x: (abs(x - center), -score[x]))

    def offset(values, center):
        values = sorted(values)
        crossings = [x for i, x in enumerate(values) if score[x] <= 0
                     and i > 0 and score[values[i - 1]] > 0]
        return min(crossings, key=lambda x: abs(x - center)) if crossings else min(
            values, key=lambda x: (abs(x - center), score[x]))

    start, end = onset(starts, coarse.start), offset(ends, coarse.end)
    if end <= start:
        start, end = coarse.start, coarse.end
    return Interval(start, end, coarse.score), {str(k): v for k, v in score.items()}, fallback


def semantic_closure(model, row, chunks, relation, coarse):
    """Expand a phase span only when needed to entail the complete relation."""
    duration = float(row["duration"])
    coarse_bin = duration / 16
    fractions = (0.0, .25, .5, 1.0)
    candidates = []
    for left in fractions:
        for right in fractions:
            start = max(0.0, coarse.start - left * coarse_bin)
            end = min(duration, coarse.end + right * coarse_bin)
            candidates.append((start, end))
    images, prompts, fallback = [], [], 0
    for start, end in candidates:
        canvas, used_fallback = span_canvas(row["video_path"], start, end, nframes=16)
        fallback += used_fallback
        images.append(canvas)
        prompts.append(relation_question(
            relation, records_in_span(chunks, start, end), "packet"))
    logits = model.binary_logits(images, prompts)
    scored = [(a, b, float(q)) for (a, b), q in zip(candidates, logits)]
    feasible = [x for x in scored if x[2] > 0]
    if feasible:
        start, end, _ = min(feasible, key=lambda x: (x[1] - x[0], -x[2], x[0]))
    else:
        start, end, _ = max(scored, key=lambda x: x[2])
    return Interval(start, end, coarse.score), {
        f"{a:.3f}-{b:.3f}": q for a, b, q in scored}, fallback


def masked_canvas(images, times, remove_indices=(), keep_indices=None):
    remove = set(remove_indices)
    keep = None if keep_indices is None else set(keep_indices)
    output = []
    for i, image in enumerate(images):
        masked = i in remove or (keep is not None and i not in keep)
        output.append(Image.new("RGB", image.size, (127, 127, 127)) if masked else image)
    return make_canvas(output, times, None, nbins=len(output))


def certify_interval(model, row, chunks, relation, interval, cited_bins):
    """Execute paired visual/text interventions and return measured log-odds."""
    duration = float(row["duration"])
    width = max(.25, interval.end - interval.start)
    trim = min(.25, width / 4)
    spans = {
        "full": (interval.start, interval.end),
        "left_trim": (min(interval.end, interval.start + trim), interval.end),
        "right_trim": (interval.start, max(interval.start, interval.end - trim)),
        "expanded": (max(0.0, interval.start - trim), min(duration, interval.end + trim)),
    }
    images, prompts, fallback = [], [], 0
    for name, (start, end) in spans.items():
        if end <= start:
            continue
        canvas, used_fallback = span_canvas(row["video_path"], start, end, nframes=16)
        fallback += used_fallback
        images.append(canvas)
        prompts.append(relation_question(relation, records_in_span(chunks, start, end), "packet"))

    times = np.linspace(interval.start, interval.end, 18)[1:-1]
    base_images, decoded_times, used_fallback = frames_at_times(row["video_path"], times)
    fallback += used_fallback
    sampled_bins = np.minimum(15, (np.asarray(decoded_times) / duration * 16).astype(int))
    cited_set = {x for b in cited_bins for x in (b - 1, b, b + 1) if 0 <= x < 16}
    cited_idx = [i for i, b in enumerate(sampled_bins) if int(b) in cited_set]
    noncited = [i for i in range(16) if i not in cited_idx]
    control_idx = sorted(noncited, key=lambda i: min(
        [abs(i - j) for j in cited_idx] or [0]))[:len(cited_idx)]
    full_text = records_in_span(chunks, interval.start, interval.end)
    cited_text = [r for r in full_text if int(min(15, ((r["start"] + r["end"]) / 2)
                                                     / duration * 16)) in cited_set]
    noncited_text = [r for r in full_text if r not in cited_text]
    def text_distance(record):
        b = int(min(15, ((record["start"] + record["end"]) / 2) / duration * 16))
        return min([abs(b - x) for x in cited_set] or [0])
    control_text = sorted(noncited_text, key=text_distance)[:len(cited_text)]
    matched_text = [r for r in full_text if r not in control_text]
    shifted_text = [{**r, "start": round(min(duration, r["start"] + duration / 4), 3),
                     "end": round(min(duration, r["end"] + duration / 4), 3)} for r in full_text]
    variants = [
        ("cited_removed", masked_canvas(base_images, decoded_times, cited_idx), noncited_text),
        ("cited_only", masked_canvas(base_images, decoded_times, keep_indices=cited_idx), cited_text),
        ("matched_control", masked_canvas(base_images, decoded_times, control_idx), matched_text),
        ("shifted", masked_canvas(base_images, decoded_times), shifted_text),
    ]
    for name, canvas, transcript in variants:
        images.append(canvas)
        prompts.append(relation_question(relation, transcript, "packet"))
    names = list(spans) + [x[0] for x in variants]
    logits = model.binary_logits(images, prompts)
    ent = dict(zip(names, map(float, logits)))
    cert = {
        "left_necessity": ent["full"] - ent["left_trim"],
        "right_necessity": ent["full"] - ent["right_trim"],
        "outside_stability": -abs(ent["full"] - ent["expanded"]),
        "alignment_effect": ent["full"] - ent["shifted"],
        "citation_selectivity": ((ent["full"] - ent["cited_removed"])
                                  - (ent["full"] - ent["matched_control"])),
        "sufficiency_gap": abs(ent["full"] - ent["cited_only"]),
    }
    passed = (ent["full"] > 0 and cert["left_necessity"] > 0
              and cert["right_necessity"] > 0 and cert["alignment_effect"] > 0
              and cert["citation_selectivity"] > 0)
    return ent, cert, passed, fallback


def auditor_prompt(relation, speech_records, start_cuts, end_cuts, coarse):
    return (
        "You are auditing the boundaries of one fixed event relation in the numbered local "
        f"timeline. Relation={json.dumps(relation, ensure_ascii=False)}. Speech records="
        f"{json.dumps(speech_records, ensure_ascii=False)}. Coarse interval="
        f"[{coarse.start:.3f},{coarse.end:.3f}]. Candidate starts={start_cuts}; candidate "
        f"ends={end_cuts}. Select one start and one end from those lists. The selected start "
        "must be the earliest cut after which the complete source-act-target-stance relation "
        "becomes supported; the selected end must be the earliest cut after which that same "
        "relation is no longer supported. Report 0..100 paired scores: full is the selected "
        "interval, left_trim removes its first 0.25s, right_trim removes its last 0.25s, "
        "expanded adds adjacent context, shifted pairs the same speech with neighboring visual "
        "time, cited_removed removes the cited evidence, and matched_control removes equally "
        "much non-cited evidence. These are counterfactual conditions, not new event labels. "
        "Return JSON only: {\"start\":number,\"end\":number,\"entailment\":{\"full\":integer,"
        "\"left_trim\":integer,\"right_trim\":integer,\"expanded\":integer,"
        "\"shifted\":integer,\"cited_removed\":integer,\"matched_control\":integer}}.")


def parse_audit(raw, start_cuts, end_cuts, duration):
    obj = parse_json_object(raw)
    start, end = float(obj["start"]), float(obj["end"])
    if min(abs(start - x) for x in start_cuts) > 1e-4:
        raise ValueError("start is not an enumerated cut")
    if min(abs(end - x) for x in end_cuts) > 1e-4 or not 0 <= start < end <= duration:
        raise ValueError("invalid end cut")
    ent = obj["entailment"]
    keys = {"full", "left_trim", "right_trim", "expanded", "shifted",
            "cited_removed", "matched_control"}
    if not isinstance(ent, dict) or set(ent) != keys:
        raise ValueError("invalid entailment schema")
    ent = {k: float(ent[k]) for k in keys}
    if not all(math.isfinite(x) and 0 <= x <= 100 for x in ent.values()):
        raise ValueError("entailment score outside 0..100")
    cert = {
        "left_necessity": ent["full"] - ent["left_trim"],
        "right_necessity": ent["full"] - ent["right_trim"],
        "outside_stability": 100 - abs(ent["full"] - ent["expanded"]),
        "alignment_effect": ent["full"] - ent["shifted"],
        "citation_selectivity": ((ent["full"] - ent["cited_removed"])
                                  - (ent["full"] - ent["matched_control"])),
    }
    return Interval(start, end, ent["full"] / 100), ent, cert


def append(method, row, curve, intervals, calls, evidence, raw, error=None):
    pred = Prediction(method, row["dataset"], row["video_id"], float(row["duration"]),
                      score_curve=np.asarray(curve, float).tolist(), intervals=intervals,
                      calls=calls, modality_evidence=evidence, raw=raw, error=error)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    ap.add_argument("--t3al-curves", type=Path, default=DEFAULT_T3AL)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--phase-only", action="store_true",
                    help="interface diagnostic; skip the boundary certifier")
    args = ap.parse_args()

    cohort = sanitized_cohort(args.cohort)
    if args.limit:
        cohort = cohort[:args.limit]
    asr = transcript_rows()
    config = {
        "version": "melt_e0_v1", "model": args.model,
        "cohort": str(args.cohort.resolve()), "t3al": str(args.t3al_curves.resolve()),
        "phase_only": bool(args.phase_only),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    methods = (("melt_phase", "melt_lifecycle", "melt_adaptive") if args.phase_only else
               ("melt_phase", "melt_lifecycle", "melt_adaptive", "melt_full"))
    expected = {(r["dataset"], r["video_id"], m) for r in cohort for m in methods}
    prior = rows(args.out) if args.out.exists() else []
    if any(r.get("raw", {}).get("config_id") != config_id for r in prior):
        raise RuntimeError("output contains records from another configuration")
    done = {(r["dataset"], r["video_id"], r["method"]) for r in prior}
    model = MLLM(args.model)

    for row in cohort:
        d, v, duration = row["dataset"], row["video_id"], float(row["duration"])
        if all((d, v, m) in done for m in methods):
            continue
        chunks, invalid = valid_chunks(asr.get((d, v), []), duration)
        speech = timed_text(chunks, duration)
        records = [{"bin": i, "start": round(i * duration / 16, 3),
                    "end": round((i + 1) * duration / 16, 3), "text": text}
                   for i, text in enumerate(speech) if text]
        curve = np.load(args.t3al_curves / d / f"{v}.npy")
        try:
            ims, times, fallback = frames_at_times(
                row["video_path"], (np.arange(16) + .5) / 16 * duration)
        except Exception as exc:
            # Media failures must be visible in the coverage ledger.  They are
            # neither converted into all-negative predictions nor allowed to
            # abort the remaining benchmark videos.
            error = f"media_decode_failure: {type(exc).__name__}: {exc}"
            raw = {"config_id": config_id, "config": config,
                   "video_path": row["video_path"]}
            for method in methods:
                if (d, v, method) not in done:
                    append_jsonl(args.out, append(
                        method, row, [], [], 0,
                        {"invalid_asr_spans_rejected": invalid}, raw, error=error))
            print(json.dumps({"dataset": d, "video_id": v, "error": error},
                             ensure_ascii=False), flush=True)
            continue
        canvas = make_canvas(ims, times, curve, "T3AL")
        calls_before = model.calls
        parser_raw = model.infer(canvas, relation_prompt(records), max_new_tokens=400)
        try:
            parsed = parse_json_object(parser_raw)
            if ("evidence_bins" not in parsed and isinstance(parsed.get("event_relation"), dict)
                    and "evidence_bins" in parsed["event_relation"]):
                parsed["event_relation"] = dict(parsed["event_relation"])
                parsed["evidence_bins"] = parsed["event_relation"].pop("evidence_bins")
            relation = parse_relation(parsed["event_relation"])
            cited = evidence_bins(parsed["evidence_bins"])
        except Exception as exc:
            raise RuntimeError(f"{d}/{v}: invalid parser output: {exc}; raw={parser_raw}") from exc
        phase = model.phase_logits(canvas, records, relation)
        if relation["stance"] == "none":
            phase[:] = 0.0
            phase[:, PHASES.index("OUT")] = 1.0
        parser_calls = model.calls - calls_before

        phase_curve = dense_bins(phase[:, 1:4].sum(1), len(curve))
        raw = {"config_id": config_id, "config": config, "parser_response": parser_raw}
        common = {"event_relation": relation, "phase_scores": phase.tolist(),
                  "evidence_bins": cited, "phase_source": "forced_choice_token_logits",
                  "invalid_asr_spans_rejected": invalid,
                  "ffmpeg_fallback_frames": fallback}
        if (d, v, "melt_phase") not in done:
            append_jsonl(args.out, append("melt_phase", row, phase_curve,
                                         phase_intervals(phase, duration), parser_calls,
                                         common, raw))
        if relation["stance"] != "none" and not cited:
            raise RuntimeError(f"{d}/{v}: endorsed/ambiguous relation has no cited evidence")
        lifecycle, event_prob, lifecycle_meta = lifecycle_decode(phase, duration, cited)
        lifecycle_curve = dense_bins(event_prob, len(curve))
        if (d, v, "melt_lifecycle") not in done:
            append_jsonl(args.out, append("melt_lifecycle", row, lifecycle_curve, lifecycle,
                                         parser_calls, {**common, "lifecycle": lifecycle_meta}, raw))
        if (d, v, "melt_adaptive") not in done:
            use_phase_ranking = (relation["stance"] in {"endorsement", "ambiguous"}
                                 and duration / 16 <= 4.0)
            adaptive_curve = lifecycle_curve if use_phase_ranking else ecdf(curve)
            adaptive_evidence = {**common, "lifecycle": lifecycle_meta,
                                 "ranking_source": ("phase_field" if use_phase_ranking
                                                    else "label_free_dense_prior"),
                                 "routing_rule": "asserted_or_ambiguous AND duration/16<=4s"}
            append_jsonl(args.out, append("melt_adaptive", row, adaptive_curve, lifecycle,
                                         parser_calls, adaptive_evidence, raw))
        if not args.phase_only and (d, v, "melt_full") not in done:
            if not lifecycle:
                full_intervals, audits, calls = [], [], parser_calls
            else:
                full_intervals, audits = [], []
                for coarse in lifecycle:
                    refined, local_scores, local_fallback = semantic_closure(
                        model, row, chunks, relation, coarse)
                    ent, cert, certified, cert_fallback = certify_interval(
                        model, row, chunks, relation, refined, cited)
                    # Fail closed at the certificate layer: uncertified refined
                    # boundaries are never represented as certified. The method
                    # retains the lifecycle boundary as an explicitly tagged
                    # fallback so coverage and selective performance can both be
                    # audited without silently dropping difficult videos.
                    chosen = refined if certified else coarse
                    full_intervals.append(chosen)
                    audits.append({"coarse": coarse.as_list(), "refined": refined.as_list(),
                                   "chosen": chosen.as_list(), "certified": certified,
                                   "semantic_closure_log_odds": local_scores,
                                   "entailment_log_odds": ent, "certificate": cert})
                    fallback += local_fallback + cert_fallback
                calls = model.calls - calls_before
            full_curve = np.zeros(len(curve), float)
            for interval in full_intervals:
                lo = max(0, int(math.floor(interval.start / duration * len(full_curve))))
                hi = min(len(full_curve), int(math.ceil(interval.end / duration * len(full_curve))))
                full_curve[lo:hi] = np.maximum(full_curve[lo:hi], interval.score)
            evidence = {**common, "lifecycle": lifecycle_meta,
                        "audits": audits,
                        "total_ffmpeg_fallback_frames": fallback}
            full_raw = {**raw, "interventions": "physically rendered and batch-scored"}
            append_jsonl(args.out, append("melt_full", row, full_curve, full_intervals,
                                         calls, evidence, full_raw))
        print(json.dumps({"dataset": d, "video_id": v, "relation": relation,
                          "lifecycle": lifecycle_meta,
                          "audits": (audits if (not args.phase_only and lifecycle) else [])},
                         ensure_ascii=False), flush=True)

    observed_rows = rows(args.out)
    counts = Counter((r["dataset"], r["video_id"], r["method"]) for r in observed_rows)
    missing = expected - set(counts)
    duplicates = {k: n for k, n in counts.items() if k in expected and n != 1}
    if missing or duplicates:
        raise RuntimeError(f"incomplete/duplicate output: missing={sorted(missing)[:3]} duplicates={duplicates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
