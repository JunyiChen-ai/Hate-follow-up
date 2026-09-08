"""Regression check: reproduce the frozen HateMM endpoint through the shared module.

The HateMM frame-level endpoint (pooled ROC-AUC 0.7451, PR-AUC 0.5601)
was produced by scripts/duplex/frame_level_eval_hatemm.py, which carried
its own copy of the grid, the gold construction, and the statistics. The
reproduction study replaces all three with
scripts/duplex/frame_eval_common.py plus the released gold array
results/reproduction/gt/hatemm_test.npz. If the new base is faithful,
recomputing the same scores through it must land on the same two
numbers; if it does not, the base is wrong and the gold must not be
adjusted to hide it.

What is recomputed: the locator's per-chunk z_masked
(results/masked_parallel_isolation/per_chunk.jsonl) spread onto the 1 fps
grid, each frame taking the z of the chunk whose [start, end) contains
it, uncovered frames taking (corpus-min chunk z) - 1 exactly as the
original pre-registration froze. Gold comes from the npz, never from the
scorer.

The method scored 212 of the 214 videos in the gold array; the two it
skipped (hate_video_321, non_hate_video_512) have unusable Whisper chunk
spans, a method-side gap, so they carry no scores to evaluate. The check
therefore runs over the intersection, which is the original cohort.

CPU only, no model calls. Output:
results/reproduction/gt/regression_hatemm.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _THIS)

import frame_eval_common as fec  # noqa: E402
from sentinel_localization_pilot import usable_spans  # noqa: E402

PER_CHUNK = os.path.join(PROJECT_ROOT, "results",
                         "masked_parallel_isolation", "per_chunk.jsonl")
CHUNKS_JSONL = os.path.join(PROJECT_ROOT, "results", "hatemm_localization",
                            "timestamped_chunks.jsonl")
GT_NPZ = os.path.join(PROJECT_ROOT, "results", "reproduction", "gt",
                      "hatemm_test.npz")
OUT_JSON = os.path.join(PROJECT_ROOT, "results", "reproduction", "gt",
                        "regression_hatemm.json")

TARGET_ROC = 0.7451
TARGET_PR = 0.5601
TOL = 0.0005

SCORE_KEY = "z_masked"


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", default=GT_NPZ)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()

    rows = read_jsonl(PER_CHUNK)
    chunk_recs = {r["video_id"]: r for r in read_jsonl(CHUNKS_JSONL)}
    gt = np.load(args.gt)
    gt_ids = set(gt.files)

    by_video = {}
    labels = {}
    for r in rows:
        by_video.setdefault(r["video_id"], {})[r["chunk_index"]] = r
        labels[r["video_id"]] = r["video_label"]

    floor = min(float(r[SCORE_KEY]) for r in rows) - 1.0

    scored_ids = sorted(by_video)
    evaluated, skipped_no_gt = [], []
    per_video = {}
    n_uncovered = 0
    for vid in scored_ids:
        if vid not in gt_ids:
            skipped_no_gt.append(vid)
            continue
        rec = chunk_recs.get(vid)
        if rec is None:
            raise SystemExit("no timestamped chunk record for %s" % vid)
        spans = usable_spans(rec)
        if spans is None:
            raise SystemExit("unusable chunk spans for %s" % vid)
        dur = float(rec["wav_duration"])
        scored = by_video[vid]
        # Non-overlapping Whisper segments; first covering scored chunk wins.
        cover = sorted((float(spans[k][0]), float(spans[k][1]), k)
                       for k in scored if k < len(spans))
        span_list = [(s, e) for s, e, _k in cover]
        values = [float(scored[k][SCORE_KEY]) for _s, _e, k in cover]
        scores, filled = fec.spans_to_frame_scores(
            span_list, values, dur, fps=fec.FPS_DEFAULT, uncovered=floor)
        n_uncovered += int((~filled).sum())
        y = gt[vid]
        if len(y) != len(scores):
            raise SystemExit("%s: gold %d frames vs scores %d frames"
                             % (vid, len(y), len(scores)))
        per_video[vid] = (scores, y)
        evaluated.append(vid)

    hate_ids = {v for v in evaluated if labels[v] == "hate"}
    res = fec.evaluate(per_video, macro_over=hate_ids)

    gt_only = sorted(gt_ids - set(evaluated))
    d_roc = abs(res["roc_auc"] - TARGET_ROC)
    d_pr = abs(res["pr_auc"] - TARGET_PR)
    passed = d_roc <= TOL and d_pr <= TOL

    print("gold array         : %s" % os.path.relpath(args.gt, PROJECT_ROOT))
    print("gold videos        : %d" % len(gt_ids))
    print("scored videos      : %d" % len(scored_ids))
    print("evaluated          : %d" % len(evaluated))
    print("in gold, unscored  : %d  %s" % (len(gt_only), gt_only))
    print("scored, not in gold: %d  %s" % (len(skipped_no_gt), skipped_no_gt))
    print("frames             : %d (%d pos / %d neg), uncovered %d, floor %.2f"
          % (res["n_frames"], res["n_pos"], res["n_neg"], n_uncovered, floor))
    print("")
    print("pooled ROC-AUC     : %.16f   target %.4f   diff %.2e"
          % (res["roc_auc"], TARGET_ROC, d_roc))
    print("pooled PR-AUC      : %.16f   target %.4f   diff %.2e"
          % (res["pr_auc"], TARGET_PR, d_pr))
    print("within-hate macro  : %.4f over %d videos"
          % (res["per_video"]["macro_auc"],
             res["per_video"]["n_videos_both_classes"]))
    print("")
    print("regression %s (tolerance %.4f)"
          % ("PASSED" if passed else "FAILED", TOL))

    report = {
        "protocol": "docs/duplex/FRAME_EVAL_PROTOCOL.md",
        "gold": os.path.relpath(args.gt, PROJECT_ROOT),
        "scores": os.path.relpath(PER_CHUNK, PROJECT_ROOT),
        "score_key": SCORE_KEY,
        "uncovered_floor": floor,
        "n_gold_videos": len(gt_ids),
        "n_scored_videos": len(scored_ids),
        "n_evaluated_videos": len(evaluated),
        "gold_videos_without_scores": gt_only,
        "scored_videos_without_gold": skipped_no_gt,
        "n_uncovered_frames": n_uncovered,
        "pooled": {k: res[k] for k in
                   ("n_frames", "n_pos", "n_neg", "positive_rate",
                    "roc_auc", "pr_auc")},
        "within_hate_video_macro": {
            k: res["per_video"][k] for k in
            ("macro_auc", "macro_auc_sd", "macro_auc_median",
             "n_videos_both_classes")},
        "targets": {"roc_auc": TARGET_ROC, "pr_auc": TARGET_PR,
                    "tolerance": TOL},
        "diffs": {"roc_auc": d_roc, "pr_auc": d_pr},
        "verdict": "PASS" if passed else "FAIL",
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("wrote %s" % os.path.relpath(args.out, PROJECT_ROOT))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
