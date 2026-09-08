#!/usr/bin/env python3
"""HYPER pilot: reciprocal MLLM selection over a frozen A10 proposal lattice."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_melt import MLLM, sanitized_cohort
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times, valid_chunks
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl, intervals_to_curve

POLICY = (
    "A hateful event is an asserted or endorsed attack, dehumanization, threat, exclusion, "
    "or hateful stereotype directed at a person or protected/social group. Quotation, neutral "
    "reporting, counterspeech, satire, and condemnation are not endorsement."
)
ARMS = ("hyper_residue", "hyper_whole")


def overlap_parts(a, b):
    common = [(max(a[0], b[0]), min(a[1], b[1]))] if max(a[0], b[0]) < min(a[1], b[1]) else []
    def subtract(x, common_):
        if not common_:
            return [x]
        c = common_[0]; out = []
        if x[0] < c[0]: out.append((x[0], c[0]))
        if c[1] < x[1]: out.append((c[1], x[1]))
        return out
    return common, subtract(a, common), subtract(b, common)


def sample_regions(regions, count):
    total = sum(max(0.0, b - a) for a, b in regions)
    if total <= 1e-8:
        return []
    targets = (np.arange(count) + .5) / count * total
    output = []
    for target in targets:
        offset = 0.0
        for a, b in regions:
            width = b - a
            if target <= offset + width or (a, b) == regions[-1]:
                output.append(a + min(width - 1e-7, max(0.0, target - offset)))
                break
            offset += width
    return output


def text_for(chunks, regions, cap=700):
    pieces = []
    for row in chunks:
        a, b = map(float, row["span"])
        if any(max(a, x) < min(b, y) for x, y in regions):
            pieces.append(f"[{a:.2f}-{b:.2f}] {row.get('text', '')}")
    return " ".join(pieces)[:cap]


def nearest(frames, frame_times, time):
    return frames[int(np.argmin(np.abs(frame_times - time)))]


def canvas(rows, frames, frame_times):
    image = Image.new("RGB", (4 * 196, len(rows) * 132), (118, 118, 118))
    draw = ImageDraw.Draw(image)
    for row_index, (label, regions) in enumerate(rows):
        times = sample_regions(regions, 4)
        for col in range(4):
            x, y = col * 196, row_index * 132
            if col < len(times):
                image.paste(nearest(frames, frame_times, times[col]).resize((196, 110)), (x, y))
                draw.text((x + 3, y + 3), f"{label} {times[col]:.1f}s", fill="yellow",
                          stroke_width=2, stroke_fill="black")
            else:
                draw.text((x + 3, y + 45), f"{label}: EMPTY", fill="white")
        draw.text((4, row_index * 132 + 112),
                  f"{label} regions=" + ",".join(f"[{a:.1f},{b:.1f})" for a, b in regions), fill="white")
    return image


@torch.inference_mode()
def choice_logits(model, images, prompts, batch_size):
    outputs = []
    for start in range(0, len(images), batch_size):
        ims = images[start:start + batch_size]; ps = prompts[start:start + batch_size]
        messages = [[{"role": "user", "content": [{"type": "image", "image": im},
                     {"type": "text", "text": prompt}]}] for im, prompt in zip(ims, ps)]
        texts = [model.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                 for m in messages]
        inputs = model.processor(text=texts, images=ims, padding=True,
                                 return_tensors="pt").to(model.model.device)
        logits = model.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        outputs.append(logits[:, model.phase_token_ids[:3]].cpu().numpy())
        model.calls += 1
    return np.concatenate(outputs)


def select(proposals, comparisons):
    wins = {i: set() for i in range(len(proposals))}
    for result in comparisons:
        if result["winner"] in (result["first"], result["second"]):
            loser = result["second"] if result["winner"] == result["first"] else result["first"]
            wins[result["winner"]].add(loser)
    condorcet = [i for i in wins if len(wins[i]) == len(proposals) - 1]
    return (condorcet[0], False) if len(condorcet) == 1 else (0, True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--proposals", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()
    if args.out.exists(): raise RuntimeError(f"refusing existing output: {args.out}")
    cohort = sanitized_cohort(args.cohort)
    proposal_rows = {(r["dataset"], r["video_id"]): r for r in
                     (json.loads(x) for x in args.proposals.read_text().splitlines() if x.strip())}
    asr = transcript_rows(); model = MLLM(args.model)
    config = {"version": "hyper_tournament_v1", "top_k": args.top_k, "model": args.model,
              "cohort_sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
              "proposal_sha256": hashlib.sha256(args.proposals.read_bytes()).hexdigest()}
    for row in cohort:
        key = (row["dataset"], row["video_id"]); duration = float(row["duration"])
        before = model.calls
        props = [(float(x["start"]), float(x["end"]), float(x["logit"]))
                 for x in proposal_rows[key]["proposals"][:args.top_k]]
        chunks, rejected = valid_chunks(asr.get(key, []), duration)
        frame_times = (np.arange(64) + .5) / 64 * duration
        frames, frame_times, fallbacks = frames_at_times(row["video_path"], frame_times)
        for arm in ARMS:
            images, prompts, meta = [], [], []
            for i, j in combinations(range(len(props)), 2):
                a, b = props[i][:2], props[j][:2]
                common, ra, rb = overlap_parts(a, b)
                for swapped in (False, True):
                    first, second = (rb, ra) if swapped else (ra, rb)
                    first_full, second_full = ([b], [a]) if swapped else ([a], [b])
                    if arm == "hyper_residue" and common:
                        display = [("COMMON", common), ("FIRST-extra", first), ("SECOND-extra", second)]
                        evidence = {"common": text_for(chunks, common),
                                    "first_extra": text_for(chunks, first),
                                    "second_extra": text_for(chunks, second)}
                        question = ("The COMMON row is evidence both hypotheses share. Which extra row better "
                                    "completes the same hateful event while adding less unrelated material?")
                    else:
                        display = [("FIRST", first_full), ("SECOND", second_full)]
                        evidence = {"first": text_for(chunks, first_full),
                                    "second": text_for(chunks, second_full)}
                        question = ("Which complete interval more precisely contains the full hateful event, "
                                    "including target, hostile act, and speaker stance, with less unrelated material?")
                    images.append(canvas(display, frames, frame_times))
                    prompts.append(f"{POLICY} Timestamp-aligned transcript evidence={json.dumps(evidence, ensure_ascii=False)}. "
                                   f"{question} Choose exactly one next letter: A=FIRST, B=SECOND, "
                                   "C=UNCERTAIN. Answer one letter only:")
                    meta.append((i, j, swapped, bool(common)))
            logits = choice_logits(model, images, prompts, args.batch_size)
            comparisons = []
            for pair_index in range(0, len(meta), 2):
                i, j, _, has_common = meta[pair_index]
                forward, reverse = logits[pair_index], logits[pair_index + 1]
                # Reverse maps B to original FIRST and A to original SECOND.
                scores = np.asarray([(forward[0] + reverse[1]) / 2,
                                     (forward[1] + reverse[0]) / 2,
                                     (forward[2] + reverse[2]) / 2])
                winner = i if int(np.argmax(scores)) == 0 else j if int(np.argmax(scores)) == 1 else None
                comparisons.append({"first": i, "second": j, "winner": winner,
                                    "scores": scores.tolist(), "forward_logits": forward.tolist(),
                                    "reverse_logits": reverse.tolist(), "has_common": has_common})
            selected, fallback = select(props, comparisons)
            interval = Interval(props[selected][0], props[selected][1],
                                float(1 / (1 + np.exp(-np.clip(props[selected][2], -30, 30)))))
            prediction = Prediction(arm, key[0], key[1], duration,
                                    score_curve=intervals_to_curve([interval], duration), intervals=[interval],
                                    calls=model.calls - before,
                                    modality_evidence={"comparisons": comparisons,
                                                       "selected_rank": selected + 1,
                                                       "condorcet_fallback": fallback,
                                                       "invalid_asr_spans_rejected": rejected,
                                                       "ffmpeg_fallback_frames": fallbacks},
                                    raw={"gt_access": False, "config": config})
            append_jsonl(args.out, prediction)
            print(json.dumps({"dataset": key[0], "video_id": key[1], "arm": arm,
                              "rank": selected + 1, "fallback": fallback}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
