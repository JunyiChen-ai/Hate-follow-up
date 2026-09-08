#!/usr/bin/env python3
"""AMBI pilot: active multimodal bilateral boundary interrogation.

The MLLM is not asked to rescore the video.  For each incumbent endpoint it
compares eight local cells with a four-frame event core and estimates same-event
membership.  The largest directed membership transition supplies a continuous
endpoint.  A matched fused-field transition control uses identical samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times, valid_chunks
from scripts.label_free_adapt.mechanisms import parse_json_object


BANK_METHODS = {
    "tight": "fact_less_t3al_dualgeo_shorter_v5",
    "midpoint": "fact_less_t3al_dualgeo_midpoint_v5",
    "broad": "fact_less_t3al_dualgeo_union_v5",
}
COALITION_METHODS = {
    "va": "less_3v_no_language_majority_global_aligned_within_video_v2",
    "vl": "less_3v_no_audio_majority_global_aligned_within_video_v2",
    "la": "less_3v_no_visual_majority_global_aligned_within_video_v2",
}


def load(path: Path, methods: dict[str, str] | None = None):
    if methods is None:
        return {(row["dataset"], row["video_id"]): row for row in map(json.loads, path.open())}
    output = {name: {} for name in methods}
    reverse = {method: name for name, method in methods.items()}
    for row in map(json.loads, path.open()):
        name = reverse.get(row.get("method"))
        if name is not None:
            output[name][(row["dataset"], row["video_id"])] = row
    return output


def logit(values):
    values = np.clip(np.asarray(values, float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def resize(values, length):
    values = np.asarray(values, float)
    index = np.minimum((np.arange(length) * len(values) / length).astype(int), len(values) - 1)
    return values[index]


def modality_fields(rows, length):
    pair = {name: resize(logit(row["score_curve"]), length) for name, row in rows.items()}
    pair = {name: value - value.mean() for name, value in pair.items()}
    va, vl, la = pair["va"], pair["vl"], pair["la"]
    fields = {"V": 0.5 * (va + vl - la), "A": 0.5 * (va + la - vl),
              "T": 0.5 * (vl + la - va)}
    return {name: value - value.mean() for name, value in fields.items()}


def interval(row):
    return tuple(map(float, row["intervals"][0][:2])) if row.get("intervals") else None


def sample(field, times, duration):
    index = np.clip(np.round(np.asarray(times) / max(duration, 1e-9) * (len(field) - 1)).astype(int),
                    0, len(field) - 1)
    values = field[index]
    scale = np.median(np.abs(field - np.median(field))) + 1e-6
    return np.tanh(values / (2 * scale))


def local_times(tight, broad, midpoint, duration, side):
    lo = min(tight[side], broad[side], midpoint[side])
    hi = max(tight[side], broad[side], midpoint[side])
    span = max(1.0, hi - lo)
    lo = max(0.0, lo - span)
    hi = min(duration, hi + span)
    if hi - lo < 1.0:
        lo, hi = max(0.0, midpoint[side] - 2.0), min(duration, midpoint[side] + 2.0)
    return np.linspace(lo, hi, 8)


def core_times(tight, duration):
    start, end = tight
    if end <= start:
        return np.full(4, 0.5 * duration)
    return np.linspace(start, end, 6)[1:-1]


def text_bins(chunks, times, radius):
    answer = []
    for index, time in enumerate(times):
        texts = [str(row.get("text", "")) for row in chunks
                 if float(row["span"][0]) <= time + radius and float(row["span"][1]) >= time - radius]
        answer.append({"cell": index, "time": round(float(time), 2),
                       "transcript": " ".join(texts)[:700]})
    return answer


def canvas(core_images, local_images, times, signals, side):
    width, cell_w, cell_h = 896, 112, 128
    image = Image.new("RGB", (width, 2 * cell_h + 92), "white")
    draw = ImageDraw.Draw(image)
    draw.text((5, 3), f"CORE REFERENCE (top) | {side.upper()} BOUNDARY ZOOM (bottom)", fill="black")
    for index, frame in enumerate(core_images):
        x = index * 224
        image.paste(frame.resize((224, 112)), (x, 18))
        draw.text((x + 3, 116), f"C{index}", fill="black")
    for index, (frame, time) in enumerate(zip(local_images, times)):
        x = index * cell_w
        image.paste(frame.resize((cell_w, 72)), (x, cell_h + 18))
        draw.text((x + 2, cell_h + 91), f"B{index} {time:.1f}s", fill="black")
        for offset, (name, color) in enumerate((("V", "blue"), ("A", "orange"), ("T", "green"))):
            value = float(signals[name][index])
            center = x + cell_w // 2
            y = cell_h + 108 + offset * 9
            draw.line((x + 2, y, x + cell_w - 2, y), fill=(220, 220, 220), width=1)
            draw.line((center, y, int(center + value * (cell_w // 2 - 4)), y), fill=color, width=5)
    return image


def infer(processor, model, image, prompt):
    message = [{"role": "user", "content": [{"type": "image", "image": image},
                                                {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=220, do_sample=False)
    return processor.batch_decode(output[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0]


def boundary_from_membership(times, values, side):
    values = np.asarray(values, float)
    difference = np.diff(values)
    index = int(np.argmax(difference) if side == "left" else np.argmin(difference))
    return float(0.5 * (times[index] + times[index + 1])), index


def control_membership(field, times, duration):
    values = sample(field, times, duration)
    return ((values + 1) * 50).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--coalitions", type=Path, required=True)
    parser.add_argument("--field", type=Path, required=True)
    parser.add_argument("--field-method", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()
    prior = list(map(json.loads, args.out.open())) if args.out.exists() else []
    done = {(row["dataset"], row["video_id"], row["method"]) for row in prior}
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, max_pixels=896 * 400)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True,
        attn_implementation="sdpa").eval()

    cohort = load(args.cohort)
    bank = load(args.bank, BANK_METHODS)
    coalitions = load(args.coalitions, COALITION_METHODS)
    fields = load(args.field, {"field": args.field_method})["field"]
    chunks = transcript_rows()
    config = {"version": "ambi_boundary_zoom_v1", "model": args.model,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "gt_access": False}
    with args.out.open("a") as handle:
        for key, manifest in cohort.items():
            expected = {(key[0], key[1], "ambi_zoom_v1"),
                        (key[0], key[1], "ambi_fused_transition_control_v1")}
            if expected <= done:
                continue
            if (key not in fields or any(key not in rows for rows in bank.values())
                    or any(key not in rows for rows in coalitions.values())):
                print(json.dumps({"dataset": key[0], "video_id": key[1],
                                  "skipped": "missing_common_resource"}), flush=True)
                continue
            bank_rows = {name: rows[key] for name, rows in bank.items()}
            tight, midpoint, broad = (interval(bank_rows[name]) for name in ("tight", "midpoint", "broad"))
            field_row = fields[key]
            if midpoint is None or tight is None or broad is None:
                for method in ("ambi_zoom_v1", "ambi_fused_transition_control_v1"):
                    row = dict(field_row); row["method"] = method; row["intervals"] = []
                    row["raw"] = {**row.get("raw", {}), "config": config, "fallback": "empty"}
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                continue
            duration = float(manifest["duration"])
            length = len(field_row["score_curve"])
            split = modality_fields({name: rows[key] for name, rows in coalitions.items()}, length)
            fused = logit(field_row["score_curve"]); fused -= fused.mean()
            clean_chunks, rejected = valid_chunks(chunks.get(key, []), duration)
            ctimes = core_times(tight, duration)
            cimages, _, _ = frames_at_times(manifest["video_path"], ctimes)
            choices, controls, audits = {}, {}, {}
            for side_index, side in enumerate(("left", "right")):
                times = local_times(tight, broad, midpoint, duration, side_index)
                images, _, fallback = frames_at_times(manifest["video_path"], times)
                signals = {name: sample(value, times, duration) for name, value in split.items()}
                picture = canvas(cimages, images, times, signals, side)
                records = text_bins(clean_chunks, times, max(0.5, (times[-1] - times[0]) / 14))
                prompt = (
                    "The top row is a reference core from one candidate hateful event. The bottom row is an ordered "
                    f"zoom around its {side} boundary. Blue/orange/green bars are label-free visual/acoustic/language "
                    "residuals and are uncertain evidence, not labels. Timestamped transcript records are "
                    + json.dumps(records, ensure_ascii=False) + ". For each bottom cell, estimate 0..100 membership in "
                    "the SAME event as the core. Event identity requires compatible depicted source/target/action and "
                    "compatible speech stance; quoted, condemned, unrelated, or another event is not the same event. "
                    "Use images, transcript, and all three evidence bars without allowing one modality to dominate. "
                    "Return JSON only: {\"membership\":[eight numbers in B0..B7 order]}."
                )
                raw = infer(processor, model, picture, prompt)
                obj = parse_json_object(raw)
                membership = obj.get("membership", [])
                if len(membership) != 8 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in membership):
                    raise RuntimeError(f"invalid membership for {key}/{side}: {raw}")
                choices[side], transition = boundary_from_membership(times, membership, side)
                control_values = control_membership(fused, times, duration)
                controls[side], control_transition = boundary_from_membership(times, control_values, side)
                audits[side] = {"times": times.tolist(), "membership": membership,
                                "control_membership": control_values, "transition": transition,
                                "control_transition": control_transition, "response": raw,
                                "transcript_records": records, "ffmpeg_fallback_frames": fallback}
            for method, endpoints in (("ambi_zoom_v1", choices),
                                      ("ambi_fused_transition_control_v1", controls)):
                start, end = endpoints["left"], endpoints["right"]
                if start >= end:
                    start, end = midpoint
                    fallback = "invalid_geometry"
                else:
                    fallback = None
                row = dict(field_row); row["method"] = method
                row["intervals"] = [[start, end, 1.0]]
                row["raw"] = {**row.get("raw", {}), "gt_access": False, "config": config,
                              "boundary_audit": audits, "fallback": fallback,
                              "invalid_asr_spans_rejected": rejected}
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            print(json.dumps({"dataset": key[0], "video_id": key[1],
                              "zoom": choices, "control": controls}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
