#!/usr/bin/env python3
"""RCD-Witness: label-free matched real-context deletion localization."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_melt import MLLM, sanitized_cohort
from scripts.idea_discovery.run_tide_u32 import POLICY, score_batches
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times, valid_chunks
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl

METHODS = ("rcd_witness", "rcd_no_necessity", "rcd_hash_zoom",
           "rcd_uniform24", "factual_uniform96")
GRID, COARSE, PARENTS, CHILDREN = 128, 8, 4, 4


def text_in(chunks, start, end, cap=1200):
    pieces = [str(x.get("text", "")) for x in chunks
              if max(float(x["span"][0]), start) < min(float(x["span"][1]), end)]
    return " ".join(pieces)[:cap]


def frame(frames, position):
    index = min(len(frames) - 1, max(0, int(position * len(frames))))
    return frames[index]


def carriers(frames, start, end, duration):
    width = end - start
    target = [start + q * width for q in (.2, .4, .6, .8)]
    context = [max(0., start - .75 * width), max(0., start - .25 * width),
               min(1. - 1e-9, end + .25 * width), min(1. - 1e-9, end + .75 * width)]
    factual_positions = context[:2] + target + context[2:]
    deleted_positions = context[:2] + context + context[2:]

    def render(positions):
        canvas = Image.new("RGB", (896, 140), "white"); draw = ImageDraw.Draw(canvas)
        for i, position in enumerate(positions):
            x = i * 112; canvas.paste(frame(frames, position).convert("RGB").resize((112, 112)), (x, 0))
            role = "T" if 2 <= i <= 5 else "C"
            draw.text((x + 3, 3), f"{role} {position*duration:.2f}s", fill="yellow",
                      stroke_width=2, stroke_fill="black")
            if 2 <= i <= 5:
                draw.rectangle((x + 2, 2, x + 109, 109), outline="red", width=4)
        draw.text((4, 116), "T=target-evidence slot; C=context; labels are true sample times", fill="black")
        return canvas
    return render(factual_positions), render(deleted_positions)


def prompt(start, end, duration, target_speech, context_speech):
    return (
        f"{POLICY} The target interval is [{start*duration:.3f}, {end*duration:.3f}) seconds. "
        "Red T slots are the visual evidence supplied for the target; every cell "
        "shows its true sampling time. C cells are context and must not themselves "
        "be classified. Use only evidence whose timestamp belongs to the target. "
        f"Target-interval speech evidence: {json.dumps(target_speech, ensure_ascii=False)}. "
        f"Neighboring context speech: {json.dumps(context_speech, ensure_ascii=False)}. "
        "Under the evidence shown, does the target interval itself contain the "
        "complete asserted or endorsed hateful event evidence? Answer Yes or No only:"
    )


def four_arm_scores(model, frames, chunks, intervals, duration, batch_size):
    images, prompts = [], []
    for start, end in intervals:
        factual, deleted = carriers(frames, start, end, duration)
        a, b = start * duration, end * duration; width = b - a
        target = text_in(chunks, a, b)
        context = (text_in(chunks, max(0., a - width), a, 600) + " " +
                   text_in(chunks, b, min(duration, b + width), 600)).strip()
        images.extend((factual, deleted, factual, deleted))
        prompts.extend((prompt(start, end, duration, target, context),
                        prompt(start, end, duration, target, context),
                        prompt(start, end, duration, "", context),
                        prompt(start, end, duration, "", context)))
    logits = score_batches(model, images, prompts, batch_size).reshape(-1, 4)
    return {interval: tuple(map(float, values)) for interval, values in zip(intervals, logits)}


def factual_scores(model, frames, chunks, intervals, duration, batch_size):
    images, prompts = [], []
    for start, end in intervals:
        factual, _ = carriers(frames, start, end, duration)
        a, b = start * duration, end * duration; width = b - a
        target = text_in(chunks, a, b)
        context = (text_in(chunks, max(0., a - width), a, 600) + " " +
                   text_in(chunks, b, min(duration, b + width), 600)).strip()
        images.append(factual); prompts.append(prompt(start, end, duration, target, context))
    logits = score_batches(model, images, prompts, batch_size)
    return {interval: float(value) for interval, value in zip(intervals, logits)}


def coarse_intervals():
    return [(i / COARSE, (i + 1) / COARSE) for i in range(COARSE)]


def split(interval):
    a, b = interval; width = (b - a) / CHILDREN
    return [(a + i * width, a + (i + 1) * width) for i in range(CHILDREN)]


def is_witness(values):
    factual, _, _, deleted = values
    return factual > 0 and deleted <= 0


def witness_order(scores):
    def key(interval):
        factual, minus_v, minus_t, deleted = scores[interval]
        disagreement = (minus_v > 0) != (minus_t > 0)
        margin = min(factual, -deleted)
        return (int(is_witness(scores[interval])), int(disagreement),
                -abs(margin), factual, -interval[0])
    return sorted(scores, key=key, reverse=True)


def hash_order(intervals, dataset, video_id):
    return sorted(intervals, key=lambda x: hashlib.sha256(
        f"RCD-HASH-V1\0{dataset}\0{video_id}\0{x[0]:.8f}\0{x[1]:.8f}".encode()).hexdigest())


def tree_partition(parents, selected, scores, necessity=True):
    chosen = set(selected); output = []
    positive = (is_witness if necessity else lambda z: z[0] > 0)
    for parent in parents:
        if parent not in chosen:
            output.append(parent); continue
        children = split(parent)
        if any(positive(scores[x]) for x in children) or not positive(scores[parent]):
            output.extend(children)
        else:
            output.append(parent)
    return output


def values_for(interval, scores, necessity):
    z = scores[interval]
    return min(z[0], -z[3]) if necessity else z[0]


def output_prediction(method, row, partition, scores, necessity, calls,
                      semantic_queries, rejected, fallback, shared_calls,
                      evaluated_windows):
    duration = float(row["duration"]); active = []
    for interval in partition:
        if (is_witness(scores[interval]) if necessity else scores[interval][0] > 0):
            active.append(interval)
    active.sort(); merged = []
    for a, b in active:
        if merged and abs(merged[-1][1] - a) < 1e-10:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    intervals = []
    for a, b in merged:
        covered = [x for x in partition if x[0] >= a and x[1] <= b]
        margin = min(values_for(x, scores, necessity) for x in covered)
        intervals.append(Interval(a * duration, b * duration,
                                  float(1 / (1 + np.exp(-np.clip(margin, -30, 30))))))
    n = max(1, math.floor(duration * 4)); t = (np.arange(n) + .5) / (4 * duration)
    curve = np.zeros(n, float)
    for interval in partition:
        mask = (t >= interval[0]) & (t < interval[1])
        margin = values_for(interval, scores, necessity)
        curve[mask] = 1 / (1 + np.exp(-np.clip(margin, -30, 30)))
    provenance = []
    for interval in partition:
        factual, minus_v, minus_t, deleted = scores[interval]
        d_v, d_t = factual - minus_v, factual - minus_t
        if d_v > 0 and d_t > 0:
            kind = "synergistic" if factual - minus_v - minus_t + deleted > 0 else "redundant"
        elif d_v > 0:
            kind = "visual_led"
        elif d_t > 0:
            kind = "text_led"
        else:
            kind = "unstable"
        provenance.append(kind)
    return Prediction(method, row["dataset"], row["video_id"], duration,
                      score_curve=curve.tolist(), intervals=intervals, calls=calls,
                      modality_evidence={
                          "leaf_windows": [list(x) for x in partition],
                          "factual_log_odds": [scores[x][0] for x in partition],
                          "minus_visual_log_odds": [scores[x][1] for x in partition],
                          "minus_text_log_odds": [scores[x][2] for x in partition],
                          "joint_deleted_log_odds": [scores[x][3] for x in partition],
                          "visual_delta": [scores[x][0] - scores[x][1] for x in partition],
                          "text_delta": [scores[x][0] - scores[x][2] for x in partition],
                          "interaction": [scores[x][0] - scores[x][1] - scores[x][2] + scores[x][3]
                                          for x in partition],
                          "evaluated_windows": [list(x) for x in evaluated_windows],
                          "evaluated_four_arm_log_odds": [list(scores[x]) for x in evaluated_windows],
                          "provenance": provenance,
                          "necessity_required": necessity,
                          "invalid_asr_spans_rejected": rejected,
                          "ffmpeg_fallback_frames": fallback,
                      }, raw={"gt_access": False, "semantic_queries": semantic_queries,
                              "shared_runner_batch_forwards": shared_calls,
                              "decision_threshold": 0.0})


def factual_prediction(row, partition, scores, calls, rejected, fallback, shared_calls):
    duration = float(row["duration"]); n = max(1, math.floor(duration * 4))
    t = (np.arange(n) + .5) / (4 * duration); curve = np.zeros(n); active = []
    for interval in partition:
        z = scores[interval]; curve[(t >= interval[0]) & (t < interval[1])] = 1/(1+np.exp(-np.clip(z,-30,30)))
        if z > 0: active.append(interval)
    merged=[]
    for a,b in active:
        if merged and abs(merged[-1][1]-a)<1e-10: merged[-1][1]=b
        else: merged.append([a,b])
    objects=[Interval(a*duration,b*duration,float(1/(1+np.exp(-np.clip(min(scores[x] for x in partition if x[0]>=a and x[1]<=b),-30,30))))) for a,b in merged]
    return Prediction("factual_uniform96",row["dataset"],row["video_id"],duration,
                      score_curve=curve.tolist(),intervals=objects,calls=calls,
                      modality_evidence={"leaf_windows":[list(x) for x in partition],
                                         "factual_log_odds":[scores[x] for x in partition],
                                         "invalid_asr_spans_rejected":rejected,
                                         "ffmpeg_fallback_frames":fallback},
                      raw={"gt_access":False,"semantic_queries":96,
                           "shared_runner_batch_forwards":shared_calls,"decision_threshold":0.0})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cohort",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True); ap.add_argument("--model",default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--batch-size",type=int,default=8); ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--methods",default=",".join(METHODS)); args=ap.parse_args()
    methods=tuple(x.strip() for x in args.methods.split(",") if x.strip())
    if not methods or len(set(methods))!=len(methods) or not set(methods)<=set(METHODS): raise ValueError(methods)
    partial=args.out.with_name(args.out.name+".partial")
    if args.out.exists() or partial.exists(): raise RuntimeError(f"refusing existing output/partial: {args.out}")
    cohort=sanitized_cohort(args.cohort); cohort=cohort[:args.limit] if args.limit else cohort
    asr=transcript_rows(); model=MLLM(args.model); expected=len(cohort)*len(methods)
    try:
        for row in cohort:
            before=model.calls; duration=float(row["duration"])
            chunks,rejected=valid_chunks(asr.get((row["dataset"],row["video_id"]),[]),duration)
            centers=(np.arange(GRID)+.5)/GRID*duration; frames,_,fallback=frames_at_times(row["video_path"],centers)
            if len(frames)!=GRID: raise RuntimeError(f"decoded {len(frames)}/{GRID}")
            needs_tree=any(x in methods for x in ("rcd_witness","rcd_no_necessity","rcd_hash_zoom"))
            coarse=coarse_intervals(); scores={}; selected=[]; hash_selected=[]
            if needs_tree or "rcd_uniform24" in methods:
                if needs_tree:
                    scores=four_arm_scores(model,frames,chunks,coarse,duration,args.batch_size)
                    selected=witness_order(scores)[:PARENTS]
                    hash_selected=hash_order(coarse,row["dataset"],row["video_id"])[:PARENTS]
                    requested=[]
                    if {"rcd_witness","rcd_no_necessity"}&set(methods): requested+=selected
                    if "rcd_hash_zoom" in methods: requested+=hash_selected
                    children=sorted(set(sum((split(x) for x in requested),[])))
                    if children: scores.update(four_arm_scores(model,frames,chunks,children,duration,args.batch_size))
                else: children=[]
            uniform24=[(i/24,(i+1)/24) for i in range(24)]; uniform_scores={}
            if "rcd_uniform24" in methods:
                uniform_scores=four_arm_scores(model,frames,chunks,uniform24,duration,args.batch_size)
            uniform96=[(i/96,(i+1)/96) for i in range(96)]; factual={}
            if "factual_uniform96" in methods:
                factual=factual_scores(model,frames,chunks,uniform96,duration,args.batch_size)
            shared=model.calls-before; pending=[]
            adaptive_calls=math.ceil(32/args.batch_size)+math.ceil(64/args.batch_size)
            for name in methods:
                if name=="rcd_witness":
                    evaluated=coarse+sum((split(x) for x in selected),[])
                    part=tree_partition(coarse,selected,scores,True); pending.append(output_prediction(name,row,part,scores,True,adaptive_calls,96,rejected,fallback,shared,evaluated))
                elif name=="rcd_no_necessity":
                    evaluated=coarse+sum((split(x) for x in selected),[])
                    part=tree_partition(coarse,selected,scores,False); pending.append(output_prediction(name,row,part,scores,False,adaptive_calls,96,rejected,fallback,shared,evaluated))
                elif name=="rcd_hash_zoom":
                    evaluated=coarse+sum((split(x) for x in hash_selected),[])
                    part=tree_partition(coarse,hash_selected,scores,True); pending.append(output_prediction(name,row,part,scores,True,adaptive_calls,96,rejected,fallback,shared,evaluated))
                elif name=="rcd_uniform24":
                    pending.append(output_prediction(name,row,uniform24,uniform_scores,True,math.ceil(96/args.batch_size),96,rejected,fallback,shared,uniform24))
                else:
                    pending.append(factual_prediction(row,uniform96,factual,math.ceil(96/args.batch_size),rejected,fallback,shared))
            for prediction in pending: append_jsonl(partial,prediction)
            print(json.dumps({"dataset":row["dataset"],"video_id":row["video_id"],"methods":methods,"batched_calls":shared}),flush=True)
        rows=[json.loads(x) for x in partial.open()]; keys=[(r["dataset"],r["video_id"],r["method"]) for r in rows]
        if len(rows)!=expected or len(set(keys))!=expected: raise RuntimeError(f"output audit failed rows={len(rows)} unique={len(set(keys))} expected={expected}")
        os.replace(partial,args.out)
    except Exception:
        partial.unlink(missing_ok=True); raise


if __name__=="__main__": main()
