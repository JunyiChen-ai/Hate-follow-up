#!/usr/bin/env python3
"""MARS: modality-specialized attribution of left/right temporal shells.

The frozen MLLM assigns each proposed shell one of four semantic states:
same-relation evidence, context/repetition, another event, or uncertainty.
Visual and transcript chambers are evaluated separately. Transcript judgments
include three circular-shift negative controls. No temporal ground truth is read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_melt import MLLM, sanitized_cohort
from scripts.idea_discovery.run_visual_temporal_canvas import (
    frames_at_times, make_canvas, timed_text, valid_chunks,
)


METHODS = {
    "tight": "fact_less_t3al_dualgeo_shorter_v5",
    "midpoint": "fact_less_t3al_dualgeo_midpoint_v5",
    "broad": "fact_less_t3al_dualgeo_union_v5",
}


@torch.inference_mode()
def choices4(model, images, prompts, batch_size):
    output = []
    for start in range(0, len(images), batch_size):
        ims, ps = images[start:start + batch_size], prompts[start:start + batch_size]
        texts = []
        for image, text in zip(ims, ps):
            message = [{"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": text}]}]
            texts.append(model.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True))
        inputs = model.processor(text=texts, images=ims, padding=True,
                                 return_tensors="pt").to(model.model.device)
        logits = model.model(**inputs, use_cache=False,
                             logits_to_keep=1).logits[:, -1].float()
        output.extend(logits[:, model.phase_token_ids[:4]].cpu().tolist())
        model.calls += 1
    return np.asarray(output)


def load_predictions(path):
    return {(r["dataset"], r["video_id"], r["method"]): r
            for r in map(json.loads, Path(path).open())}


def load_relations(path):
    output = {}
    for row in map(json.loads, Path(path).open()):
        relation = row.get("modality_evidence", {}).get("event_relation")
        if relation:
            output[(row["dataset"], row["video_id"])] = relation
    return output


def first_interval(row):
    values = row.get("intervals", []) if row else []
    return tuple(map(float, values[0][:2])) if values else None


def records(speech, duration):
    n = len(speech)
    return [{"bin": i, "start": round(i * duration / n, 3),
             "end": round((i + 1) * duration / n, 3), "text": text}
            for i, text in enumerate(speech) if text]


def in_span(rows, span):
    a, b = span
    return [r for r in rows if max(float(r["start"]), a) < min(float(r["end"]), b)]


def shifted(rows, amount, duration, n=16):
    output = []
    for row in rows:
        item = dict(row); item["bin"] = (int(item["bin"]) + amount) % n
        item["start"] = round(item["bin"] * duration / n, 3)
        item["end"] = round((item["bin"] + 1) * duration / n, 3)
        output.append(item)
    return output


def shell_canvas(images, times, core, shell, visual=True):
    blank = Image.new("RGB", images[0].size, (127, 127, 127))
    visible = []
    for image, time in zip(images, times):
        keep = core[0] <= time < core[1] or shell[0] <= time < shell[1]
        visible.append(image if visual and keep else blank)
    canvas = make_canvas(visible, times, None, "", nbins=len(images))
    draw = ImageDraw.Draw(canvas); cell_w, cell_h = 224, 156
    for i, time in enumerate(times):
        x, y = (i % 4) * cell_w, (i // 4) * cell_h
        if core[0] <= time < core[1]:
            draw.rectangle((x + 3, y + 3, x + 220, y + 152), outline="green", width=5)
        elif shell[0] <= time < shell[1]:
            draw.rectangle((x + 3, y + 3, x + 220, y + 152), outline="orange", width=5)
    return canvas


def prompt(relation, core, shell, chamber, core_text=None, shell_text=None):
    specialization = (
        "Judge only visually grounded source, protected target, depicted act, and scene continuity. "
        "Do not guess spoken stance or words."
        if chamber == "visual" else
        "Judge only the asserted hostile act, protected target reference, and speaker stance in the transcript. "
        "The gray image contains no usable visual evidence."
    )
    evidence = "" if chamber == "visual" else (
        f"Core transcript={json.dumps(core_text, ensure_ascii=False)}. "
        f"Shell transcript={json.dumps(shell_text, ensure_ascii=False)}. ")
    return (
        f"Fixed hateful-event hypothesis={json.dumps(relation, ensure_ascii=False)}. "
        f"Core interval={list(map(lambda x: round(x,3), core))}; candidate shell="
        f"{list(map(lambda x: round(x,3), shell))}. Green cells are core and orange cells are shell. "
        f"{specialization} {evidence}Classify what the SHELL adds relative to the core for this exact relation. "
        "Choose exactly one next letter: A=necessary evidence belonging to the same event relation; "
        "B=background, explanation, repetition, or reaction not belonging to the event boundary; "
        "C=evidence of a different event/relation; D=insufficient or ambiguous evidence. Answer one letter only:"
    )


def endpoint_transitions(intervals, side):
    tight = intervals["tight"]
    values = [intervals[name][0 if side == "left" else 1]
              for name in ("tight", "midpoint", "broad")]
    # Expansion is decreasing on the left and increasing on the right.
    endpoints = sorted(set(values), reverse=(side == "left"))
    start = tight[0 if side == "left" else 1]
    if start in endpoints:
        endpoints.remove(start)
    endpoints.insert(0, start)
    return list(zip(endpoints[:-1], endpoints[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--relations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shifts", default="4,8,12")
    args = ap.parse_args()
    partial = args.out.with_name(args.out.name + ".partial")
    if args.out.exists() or partial.exists():
        raise RuntimeError(f"refusing existing output/partial: {args.out}")
    cohort = sanitized_cohort(args.cohort)
    if args.limit: cohort = cohort[:args.limit]
    bank, relations, asr = load_predictions(args.base), load_relations(args.relations), transcript_rows()
    model = MLLM(args.model); shifts = tuple(int(x) for x in args.shifts.split(",") if x)
    written = 0
    try:
        with partial.open("w") as handle:
            for row in cohort:
                key = row["dataset"], row["video_id"]
                iv = {name: first_interval(bank.get((*key, method))) for name, method in METHODS.items()}
                relation = relations.get(key)
                if relation is None or any(value is None for value in iv.values()):
                    continue
                duration = float(row["duration"]); clean, rejected = valid_chunks(asr.get(key, []), duration)
                recs = records(timed_text(clean, duration, nbins=16), duration)
                centers = (np.arange(16) + .5) / 16 * duration
                images, times, fallback = frames_at_times(row["video_path"], centers)
                examples, prompts, metadata = [], [], []
                tight = iv["tight"]
                for side in ("left", "right"):
                    for old, new in endpoint_transitions(iv, side):
                        if abs(old - new) < 1e-8: continue
                        if side == "left": core = (old, tight[1]); shell = (new, old)
                        else: core = (tight[0], old); shell = (old, new)
                        if shell[1] <= shell[0]: continue
                        visual = shell_canvas(images, times, core, shell, visual=True)
                        blank = shell_canvas(images, times, core, shell, visual=False)
                        core_text, shell_text = in_span(recs, core), in_span(recs, shell)
                        examples.append(visual); prompts.append(prompt(relation, core, shell, "visual"))
                        metadata.append((side, old, new, "visual"))
                        examples.append(blank); prompts.append(prompt(relation, core, shell, "text", core_text, shell_text))
                        metadata.append((side, old, new, "text_aligned"))
                        for shift in shifts:
                            shifted_rows = shifted(recs, shift, duration)
                            examples.append(blank); prompts.append(prompt(
                                relation, core, shell, "text", in_span(shifted_rows, core),
                                in_span(shifted_rows, shell)))
                            metadata.append((side, old, new, f"text_shift{shift}"))
                logits = choices4(model, examples, prompts, args.batch_size)
                grouped = {}
                for meta, values in zip(metadata, logits):
                    side, old, new, chamber = meta
                    grouped.setdefault((side, old, new), {})[chamber] = dict(zip("ABCD", map(float, values[:4])))
                decisions = []
                for (side, old, new), chambers in grouped.items():
                    visual = chambers["visual"]; text = chambers["text_aligned"]
                    vwinner = max(visual, key=visual.get); twinner = max(text, key=text.get)
                    tmargin = text["A"] - max(text["B"], text["C"])
                    null = []
                    for shift in shifts:
                        z = chambers[f"text_shift{shift}"]
                        null.append(z["A"] - max(z["B"], z["C"]))
                    corrected = tmargin - float(np.median(null))
                    vstate = "extend" if vwinner == "A" else "abstain" if vwinner == "D" else "reject"
                    tstate = ("extend" if twinner == "A" and corrected > 0 else
                              "abstain" if twinner == "D" else "reject")
                    extend = "reject" not in (vstate, tstate) and "extend" in (vstate, tstate)
                    decisions.append({"side": side, "from": old, "to": new,
                        "visual_logits": visual, "text_logits": text,
                        "text_shift_logits": {str(s): chambers[f"text_shift{s}"] for s in shifts},
                        "visual_state": vstate, "text_state": tstate,
                        "text_null_corrected_margin": corrected, "extend": extend})
                result = {"method": "mars_shell_tribunal_v1", "dataset": key[0],
                    "video_id": key[1], "duration": duration, "relation": relation,
                    "interval_lattice": iv, "decisions": decisions,
                    "calls": int(np.ceil(len(examples) / args.batch_size)),
                    "modality_evidence": {"invalid_asr_spans_rejected": rejected,
                                          "ffmpeg_fallback_frames": fallback},
                    "raw": {"gt_access": False, "shifts": list(shifts),
                            "states": ["same_relation", "context", "new_event", "uncertain"]}}
                handle.write(json.dumps(result, ensure_ascii=False) + "\n"); handle.flush()
                written += 1; print(json.dumps({"dataset": key[0], "video_id": key[1],
                    "shells": len(decisions), "calls": result["calls"]}), flush=True)
        if not written: raise RuntimeError("no eligible videos")
        os.replace(partial, args.out)
    except Exception:
        partial.unlink(missing_ok=True); raise


if __name__ == "__main__": main()
