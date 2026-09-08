#!/usr/bin/env python3
"""Hierarchical relation-state transition refinement for frozen MELT outputs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idea_discovery.run_counterfactual_evidence import rows, transcript_rows
from scripts.idea_discovery.run_melt import (
    MLLM, records_in_span, relation_question, span_canvas, valid_chunks,
)
from scripts.label_free_adapt.schema import Interval, Prediction, append_jsonl


def score_cuts(model, row, chunks, relation, cuts):
    duration = float(row["duration"])
    radius = max(.5, min(2.0, duration / 32))
    images, prompts, fallback = [], [], 0
    for cut in cuts:
        start, end = max(0, cut - radius), min(duration, cut + radius)
        canvas, used = span_canvas(row["video_path"], start, end)
        fallback += used
        text = records_in_span(chunks, start, end)
        images.append(canvas)
        prompts.append(relation_question(relation, text, "packet") +
                       f" The exact center timestamp is {cut:.3f}s; judge the relation state "
                       "at that center, not merely anywhere in the packet.")
    logits = model.binary_logits(images, prompts)
    return {float(c): float(q) for c, q in zip(cuts, logits)}, fallback


def crossing(scores, kind, center):
    cuts = sorted(scores)
    if kind == "onset":
        valid = [x for i, x in enumerate(cuts) if i and scores[cuts[i - 1]] <= 0 < scores[x]]
    else:
        valid = [x for i, x in enumerate(cuts) if i and scores[cuts[i - 1]] > 0 >= scores[x]]
    return min(valid, key=lambda x: abs(x - center)) if valid else None


def refine_interval(model, row, chunks, relation, interval):
    duration = float(row["duration"]); width = duration / 16
    coarse_cuts = sorted(set(
        np.linspace(max(0, interval.start - width / 2),
                    min(duration, interval.start + width / 2), 9).tolist()
        + np.linspace(max(0, interval.end - width / 2),
                      min(duration, interval.end + width / 2), 9).tolist()))
    coarse_scores, fallback = score_cuts(model, row, chunks, relation, coarse_cuts)
    onset = crossing(coarse_scores, "onset", interval.start)
    offset = crossing(coarse_scores, "offset", interval.end)
    fine_centers = [x for x in (onset, offset) if x is not None]
    fine_scores = {}
    if fine_centers:
        radius = max(.5, min(2.0, width / 4))
        fine_cuts = sorted({round(t, 3) for center in fine_centers
                            for t in np.arange(max(0, center - radius),
                                               min(duration, center + radius) + 1e-6, .25)})
        fine_scores, used = score_cuts(model, row, chunks, relation, fine_cuts)
        fallback += used
        if onset is not None:
            onset = crossing(fine_scores, "onset", onset) or onset
        if offset is not None:
            offset = crossing(fine_scores, "offset", offset) or offset
    start = interval.start if onset is None else onset
    end = interval.end if offset is None else offset
    overlap = max(0.0, min(end, interval.end) - max(start, interval.start))
    union = max(end, interval.end) - min(start, interval.start)
    trust_iou = overlap / union if union > 0 else 0.0
    if end <= start or trust_iou < .5:
        start, end = interval.start, interval.end
    return Interval(start, end, interval.score), {
        "coarse_log_odds": {str(k): v for k, v in coarse_scores.items()},
        "fine_log_odds": {str(k): v for k, v in fine_scores.items()},
        "onset_found": onset is not None, "offset_found": offset is not None,
        "trust_iou": trust_iou,
    }, fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    args = ap.parse_args()
    source = [r for r in rows(args.predictions) if r["method"] == "melt_adaptive"]
    cohort = {(r["dataset"], r["video_id"]): r for r in rows(args.cohort)}
    asr = transcript_rows(); model = MLLM(args.model)
    config = {"version": "melt_transition_v1", "source": str(args.predictions.resolve()),
              "source_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest()}
    prior = rows(args.out) if args.out.exists() else []
    done = {(r["dataset"], r["video_id"]) for r in prior}
    for base in source:
        key = base["dataset"], base["video_id"]
        if key in done: continue
        row = cohort[key]; chunks, _ = valid_chunks(asr.get(key, []), float(row["duration"]))
        relation = base["modality_evidence"]["event_relation"]
        before = model.calls; intervals = []; audits = []; fallback = 0
        for raw_interval in base["intervals"]:
            interval = Interval(*map(float, raw_interval[:3]))
            refined, audit, used = refine_interval(model, row, chunks, relation, interval)
            intervals.append(refined); audits.append(audit); fallback += used
        pred = Prediction("melt_transition_refined", base["dataset"], base["video_id"],
                          float(base["duration"]), score_curve=base["score_curve"],
                          intervals=intervals, calls=model.calls - before,
                          modality_evidence={**base["modality_evidence"],
                                             "transition_audits": audits,
                                             "refine_ffmpeg_fallback_frames": fallback},
                          raw={"config": config})
        append_jsonl(args.out, pred)
        print(json.dumps({"dataset": key[0], "video_id": key[1],
                          "before": base["intervals"],
                          "after": [x.as_list() for x in intervals]}), flush=True)


if __name__ == "__main__":
    main()
