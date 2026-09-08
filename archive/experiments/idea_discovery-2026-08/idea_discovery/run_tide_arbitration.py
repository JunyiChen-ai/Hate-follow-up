#!/usr/bin/env python3
"""Disagreement-triggered local multimodal arbitration for TIDE U32."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_melt import MLLM, sanitized_cohort
from scripts.idea_discovery.run_tide_u32 import (POLICY, bins_to_curve, carrier,
                                                 positive_intervals, score_batches)
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times, timed_text, valid_chunks
from scripts.label_free_adapt.schema import Prediction, append_jsonl


def arbitration_prompt(index, records, votes):
    return (
        f"{POLICY} The red temporal cell is the only cell to decide. Immediate "
        f"neighbors provide context. Timestamp-aligned local speech records="
        f"{json.dumps(records, ensure_ascii=False)}. Three independently isolated "
        f"views disagreed: joint={votes[0]}, visual={votes[1]}, transcript={votes[2]}. "
        "Treat these votes as fallible hints. Determine whether the visual evidence "
        "corroborates, contradicts, or is irrelevant to the speech, and whether the "
        f"speaker asserts or endorses the hostility. Does cell {index} itself contain "
        "a complete hateful event? Answer Yes or No only:"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--fields", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.out.exists():
        raise RuntimeError(f"refusing existing output: {args.out}")
    partial = Path(str(args.out) + ".partial")
    if partial.exists():
        raise RuntimeError(f"refusing stale partial output: {partial}")
    cohort = {(r["dataset"], r["video_id"]): r for r in sanitized_cohort(args.cohort)}
    fields = {}
    for line in args.fields.open():
        r = json.loads(line); key = (r["dataset"], r["video_id"]); method = r["method"]
        if method in fields.setdefault(key, {}):
            raise RuntimeError(f"duplicate cached field: {(key, method)}")
        fields[key][method] = r
    if set(fields) != set(cohort):
        raise RuntimeError(f"field/cohort key mismatch: extra={set(fields)-set(cohort)}, missing={set(cohort)-set(fields)}")
    required = {"u32_joint", "u32_visual", "u32_text"}
    for key, arms in fields.items():
        if not required <= set(arms):
            raise RuntimeError(f"missing cached arms for {key}: {required-set(arms)}")
        for method in required:
            r = arms[method]; values = np.asarray(r["modality_evidence"]["corrected_log_odds"], float)
            if r["dataset"] != key[0] or r["video_id"] != key[1] or abs(float(r["duration"])-float(cohort[key]["duration"])) > 1e-6:
                raise RuntimeError(f"cached identity/duration mismatch for {(key, method)}")
            if values.shape != (32,) or not np.isfinite(values).all():
                raise RuntimeError(f"invalid cached field for {(key, method)}: {values.shape}")
    asr = transcript_rows(); model = MLLM(args.model)
    for item_index, (key, row) in enumerate(cohort.items()):
        if args.limit and item_index >= args.limit:
            break
        arms = fields[key]
        z = np.stack([arms[name]["modality_evidence"]["corrected_log_odds"]
                      for name in ("u32_joint", "u32_visual", "u32_text")])
        signs = z > 0; disputed = np.flatnonzero(np.any(signs, axis=0) & np.any(~signs, axis=0))
        final = np.median(z, axis=0)
        duration = float(row["duration"]); before = model.calls
        chunks, rejected = valid_chunks(asr.get(key, []), duration)
        speech = timed_text(chunks, duration, nbins=32)
        remaining = 12000; budgeted = []
        for text in speech:
            value = text[:remaining] if remaining > 0 else ""; budgeted.append(value); remaining -= len(value)
        speech = budgeted
        records = [{"bin": i, "start": round(i * duration / 32, 3),
                    "end": round((i + 1) * duration / 32, 3), "text": text}
                   for i, text in enumerate(speech) if text]
        centers = (np.arange(32) + .5) / 32 * duration
        frames, times, fallback = frames_at_times(row["video_path"], centers)
        if len(frames) != 32:
            raise RuntimeError(f"decoded {len(frames)}/32 center frames for {key}")
        actual_images=[]; null_images=[]; actual_prompts=[]; null_prompts=[]
        for i in disputed:
            local_ids={max(0,int(i)-1),int(i),min(31,int(i)+1)}
            local_records=[r for r in records if r["bin"] in local_ids]
            vote_words=["Yes" if x else "No" for x in signs[:,i]]
            actual_images.append(carrier(frames,int(i),local_ids));null_images.append(carrier(frames,int(i),set()))
            actual_prompts.append(arbitration_prompt(int(i),local_records,vote_words))
            null_prompts.append(arbitration_prompt(int(i),[],vote_words))
        if len(disputed):
            raw=score_batches(model,actual_images,actual_prompts,args.batch_size)
            null=score_batches(model,null_images,null_prompts,args.batch_size)
            final[disputed]=raw-null
        probability=1/(1+np.exp(-np.clip(final,-30,30)))
        append_jsonl(partial,Prediction(
            "tide_arbitrated",key[0],key[1],duration,
            score_curve=bins_to_curve(probability,duration).tolist(),
            intervals=positive_intervals(final,duration),calls=model.calls-before,
            modality_evidence={"disputed_bins":disputed.tolist(),
                               "final_log_odds":final.tolist(),
                               "invalid_asr_spans_rejected":rejected,
                               "ffmpeg_fallback_frames":fallback},
            raw={"semantic_queries":2*len(disputed),"gt_access":False}))
        print(json.dumps({"dataset":key[0],"video_id":key[1],"disputed":len(disputed),
                          "calls":model.calls-before}),flush=True)
    produced = [json.loads(x) for x in partial.open()]
    expected = min(len(cohort), args.limit) if args.limit else len(cohort)
    output_keys = [(r["dataset"], r["video_id"], r["method"]) for r in produced]
    if len(produced) != expected or len(set(output_keys)) != expected:
        raise RuntimeError(f"invalid final output: rows={len(produced)}, unique={len(set(output_keys))}, expected={expected}")
    partial.replace(args.out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if "--out" in sys.argv:
            index = sys.argv.index("--out")
            if index + 1 < len(sys.argv):
                Path(sys.argv[index + 1] + ".partial").unlink(missing_ok=True)
        raise
