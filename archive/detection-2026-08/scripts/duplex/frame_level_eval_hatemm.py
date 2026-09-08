"""Frame-level evaluation of the locator's per-chunk scores on HateMM.

Pre-registration: docs/duplex/PREREG_frame_level_evaluation_hatemm.md
(frozen 2026-08-18, commit febf147). Nothing in that protocol is
re-decided here.

CPU only. No model calls. This is a deterministic transformation of
scores already on disk: the locator's per-chunk z values
(results/masked_parallel_isolation/per_chunk.jsonl) are spread onto a
1 fps frame grid and scored against the upstream hate spans, under the
LAVAD pooling convention that the published comparison target (LELA)
reports against.

Protocol, as frozen:

- Cohort: the 212 pilot videos (84 hate / 128 non-hate), i.e. exactly
  the videos present in the pilot's per_chunk.jsonl.
- Frame grid: one frame per second over [0, wav_duration) per video,
  frame timestamps t = 0, 1, 2, ... while t < wav_duration.
- Frame score: the z of the scored chunk whose [start, end) contains
  the frame's timestamp. Chunk spans come from the same usable_spans
  helper the pilot used, indexed by the pilot's chunk_index.
- Uncovered frames (no scored chunk spans them: silence, music, or a
  chunk the pilot dropped): score = (corpus-wide min chunk z) - 1,
  computed per score column. A transcript-only locator has no evidence
  there. Sensitivity row: covered frames only.
- Gold: a frame is positive iff its timestamp lies inside a hate span
  (span_gold.json, half-open [s, e)). Frames of hate videos outside
  spans are negative; all frames of non-hate videos are negative.
- Statistics: pooled frame ROC-AUC (primary), pooled PR-AUC / average
  precision (secondary, primary row). Honesty section: within-hate-video
  frame-level macro AUC over hate videos that carry both classes.

Rows: primary z_masked (all frames); sensitivity z_reference (all
frames); sensitivity z_masked (covered frames only). The same table is
also computed with z_causal, purely descriptively: it shows what the
unmasked arm would score and is NOT a decision row.

Frozen decision rule: endpoint 1 PASSES iff the primary pooled frame
ROC-AUC >= 0.65.

Output: results/frame_level_eval/{report.json, run.log}. No transcript
text is written anywhere.
"""

import argparse
import json
import logging
import math
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _THIS)

from sentinel_localization_pilot import rank_auc, usable_spans  # noqa: E402

PER_CHUNK = os.path.join(PROJECT_ROOT, "results",
                         "masked_parallel_isolation", "per_chunk.jsonl")
LOC_DIR = os.path.join(PROJECT_ROOT, "results", "hatemm_localization")
CHUNKS_JSONL = os.path.join(LOC_DIR, "timestamped_chunks.jsonl")
SPAN_GOLD = os.path.join(LOC_DIR, "span_gold.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "frame_level_eval")

FRAME_RATE_HZ = 1.0
PASS_BAR = 0.65

SCORE_KEYS = ["z_masked", "z_reference", "z_causal"]


# ------------------------------------------------------------- statistics
def average_precision(pos, neg):
    """Rank-based average precision (step-wise AP, ties collapsed).

    Scores are sorted descending; tied scores form one group, and the
    precision/recall pair is read after each complete group, so the
    result does not depend on the input order within a tie.
    """
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    labels = labels[order]
    n_pos = float(len(pos))
    ap = 0.0
    prev_recall = 0.0
    tp = 0.0
    seen = 0.0
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j < n and scores[j] == scores[i]:
            j += 1
        tp += float(labels[i:j].sum())
        seen += float(j - i)
        recall = tp / n_pos
        precision = tp / seen
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return float(ap)


def describe_macro(values):
    v = [x for x in values if x is not None]
    if not v:
        return {"macro_auc": None, "macro_auc_sd": None,
                "macro_auc_median": None, "n_videos_both_classes": 0}
    a = np.asarray(v, float)
    return {
        "macro_auc": float(a.mean()),
        "macro_auc_sd": float(a.std(ddof=1)) if len(a) > 1 else None,
        "macro_auc_median": float(np.median(a)),
        "n_videos_both_classes": int(len(a)),
    }


# ------------------------------------------------------------------ inputs
def load_inputs():
    rows = []
    with open(PER_CHUNK) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    chunk_recs = {}
    with open(CHUNKS_JSONL) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                chunk_recs[r["video_id"]] = r

    with open(SPAN_GOLD) as f:
        gold = json.load(f)["spans"]

    return rows, chunk_recs, gold


def build_frames(rows, chunk_recs, gold, log):
    """One frame table per video; frame score columns are chunk indices.

    Returns (frames, counts). Each frame is
    (video_id, video_label, t, gold_positive, chunk_index_or_None).
    """
    by_video = {}
    labels = {}
    for r in rows:
        by_video.setdefault(r["video_id"], {})[r["chunk_index"]] = r
        labels[r["video_id"]] = r["video_label"]

    counts = {
        "videos": 0, "videos_hate": 0, "videos_non_hate": 0,
        "frames_total": 0, "frames_pos": 0, "frames_neg": 0,
        "frames_uncovered": 0, "frames_covered": 0,
        "frames_uncovered_hate_video": 0,
        "frames_uncovered_non_hate_video": 0,
        "seconds_total": 0.0,
        "chunks_scored": len(rows),
        "chunks_unusable_span": 0,
    }

    frames = []
    for vid in sorted(by_video):
        rec = chunk_recs.get(vid)
        if rec is None:
            raise SystemExit("no timestamped chunk record for %s" % vid)
        spans = usable_spans(rec)
        if spans is None:
            counts["chunks_unusable_span"] += 1
            raise SystemExit("unusable chunk spans for %s" % vid)
        dur = rec.get("wav_duration") or rec.get("container_duration")
        if not dur or float(dur) <= 0:
            raise SystemExit("no positive duration for %s" % vid)
        dur = float(dur)

        is_hate = labels[vid] == "hate"
        gspans = gold.get(vid) or []
        if is_hate and not gspans:
            raise SystemExit("hate video without span gold: %s" % vid)

        scored = by_video[vid]
        # Non-overlapping Whisper segments: first covering scored chunk wins.
        cover = sorted(
            (float(spans[k][0]), float(spans[k][1]), k)
            for k in scored
            if k < len(spans)
        )

        counts["videos"] += 1
        counts["videos_hate" if is_hate else "videos_non_hate"] += 1
        counts["seconds_total"] += dur

        n_frames = int(math.ceil(dur / FRAME_RATE_HZ))
        for i in range(n_frames):
            t = float(i) * FRAME_RATE_HZ
            if t >= dur:
                break
            ck = None
            for s, e, k in cover:
                if s <= t < e:
                    ck = k
                    break
            pos = False
            if is_hate:
                for s, e in gspans:
                    if float(s) <= t < float(e):
                        pos = True
                        break
            frames.append((vid, labels[vid], t, pos, ck))
            counts["frames_total"] += 1
            counts["frames_pos" if pos else "frames_neg"] += 1
            if ck is None:
                counts["frames_uncovered"] += 1
                counts["frames_uncovered_hate_video" if is_hate
                       else "frames_uncovered_non_hate_video"] += 1
            else:
                counts["frames_covered"] += 1

    counts["seconds_total"] = round(counts["seconds_total"], 3)
    log("frames built: %d over %d videos (%d hate / %d non-hate)"
        % (counts["frames_total"], counts["videos"],
           counts["videos_hate"], counts["videos_non_hate"]))
    return frames, counts, by_video


# ------------------------------------------------------------------- rows
def score_frames(frames, by_video, key, covered_only):
    """Frame scores and gold for one score column."""
    allz = [r[key] for v in by_video.values() for r in v.values()]
    floor = float(min(allz)) - 1.0
    pos, neg = [], []
    per_video = {}
    for vid, label, _t, gpos, ck in frames:
        if ck is None:
            if covered_only:
                continue
            s = floor
        else:
            s = float(by_video[vid][ck][key])
        (pos if gpos else neg).append(s)
        if label == "hate":
            d = per_video.setdefault(vid, ([], []))
            (d[0] if gpos else d[1]).append(s)
    return pos, neg, floor, per_video


def row_stats(frames, by_video, key, covered_only, with_ap):
    pos, neg, floor, per_video = score_frames(
        frames, by_video, key, covered_only)
    vids = sorted(per_video)
    aucs = [rank_auc(*per_video[v]) for v in vids]
    macro = [a for a in aucs if a is not None]
    macro_used = [v for v, a in zip(vids, aucs) if a is not None]
    out = {
        "score": key,
        "coverage": "covered_only" if covered_only else "all_frames",
        "uncovered_floor": None if covered_only else floor,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "roc_auc": rank_auc(pos, neg),
        "pr_auc": average_precision(pos, neg) if with_ap else None,
        "positive_rate": (len(pos) / float(len(pos) + len(neg))
                          if (pos or neg) else None),
        "within_hate_video": describe_macro(macro),
    }
    out["within_hate_video"]["per_video_auc"] = dict(zip(macro_used, macro))
    return out


def fmt(x, nd=4):
    return "n/a" if x is None else ("%.*f" % (nd, x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(args.out_dir, "run.log"),
                                      mode="w"),
                  logging.StreamHandler(sys.stdout)])
    log = logging.info

    log("frame-level evaluation, prereg "
        "docs/duplex/PREREG_frame_level_evaluation_hatemm.md (febf147)")
    rows, chunk_recs, gold = load_inputs()
    log("loaded %d scored chunks, %d chunk records, %d span-gold videos"
        % (len(rows), len(chunk_recs), len(gold)))

    frames, counts, by_video = build_frames(rows, chunk_recs, gold, log)

    decision_rows = [
        ("primary", "z_masked", False, True),
        ("sensitivity", "z_reference", False, False),
        ("sensitivity", "z_masked", True, False),
    ]
    descriptive_rows = [
        ("descriptive", "z_causal", False, True),
        ("descriptive", "z_causal", True, False),
    ]

    report_rows = []
    for role, key, cov_only, with_ap in decision_rows + descriptive_rows:
        r = row_stats(frames, by_video, key, cov_only, with_ap)
        r["role"] = role
        report_rows.append(r)

    primary = report_rows[0]
    passed = (primary["roc_auc"] is not None
              and primary["roc_auc"] >= PASS_BAR)

    # ------------------------------------------------------------- table
    log("")
    log("frame counts: total=%d pos=%d neg=%d covered=%d uncovered=%d "
        "(hate-video uncovered=%d, non-hate-video uncovered=%d)"
        % (counts["frames_total"], counts["frames_pos"],
           counts["frames_neg"], counts["frames_covered"],
           counts["frames_uncovered"],
           counts["frames_uncovered_hate_video"],
           counts["frames_uncovered_non_hate_video"]))
    log("videos: %d (%d hate / %d non-hate); audio seconds=%.1f; "
        "scored chunks=%d"
        % (counts["videos"], counts["videos_hate"],
           counts["videos_non_hate"], counts["seconds_total"],
           counts["chunks_scored"]))
    log("")
    head = ("%-12s %-12s %-13s %9s %9s %9s %9s %9s %6s"
            % ("role", "score", "coverage", "n_pos", "n_neg",
               "ROC-AUC", "PR-AUC", "macroAUC", "n_vid"))
    log(head)
    log("-" * len(head))
    for r in report_rows:
        w = r["within_hate_video"]
        log("%-12s %-12s %-13s %9d %9d %9s %9s %9s %6d"
            % (r["role"], r["score"], r["coverage"], r["n_pos"],
               r["n_neg"], fmt(r["roc_auc"]), fmt(r["pr_auc"]),
               fmt(w["macro_auc"]), w["n_videos_both_classes"]))
    log("")
    log("honesty section (within-hate-video frame-level macro AUC, "
        "hate videos with both frame classes):")
    for r in report_rows:
        w = r["within_hate_video"]
        log("  %-12s %-13s macro=%s  sd=%s  median=%s  n=%d"
            % (r["score"], r["coverage"], fmt(w["macro_auc"]),
               fmt(w["macro_auc_sd"]), fmt(w["macro_auc_median"]),
               w["n_videos_both_classes"]))
    log("")
    log("frozen bar: primary pooled frame ROC-AUC >= %.2f" % PASS_BAR)
    log("primary pooled frame ROC-AUC = %s -> %s"
        % (fmt(primary["roc_auc"]), "PASS" if passed else "FAIL"))

    report = {
        "preregistered": True,
        "prereg": "docs/duplex/PREREG_frame_level_evaluation_hatemm.md",
        "prereg_commit": "febf147",
        "dataset": "HateMM",
        "split": "test_clean",
        "cpu_only": True,
        "model_calls": 0,
        "inputs": {
            "per_chunk": os.path.relpath(PER_CHUNK, PROJECT_ROOT),
            "timestamped_chunks": os.path.relpath(CHUNKS_JSONL, PROJECT_ROOT),
            "span_gold": os.path.relpath(SPAN_GOLD, PROJECT_ROOT),
        },
        "protocol": {
            "frame_rate_hz": FRAME_RATE_HZ,
            "frame_timestamps": "t = 0, 1, 2, ... while t < wav_duration",
            "span_convention": "half-open [start, end)",
            "uncovered_frame_score": "corpus-min chunk z minus 1, per column",
            "gold": ("positive iff timestamp inside a hate span; hate-video "
                     "non-span frames and all non-hate-video frames negative"),
        },
        "counts": counts,
        "rows": report_rows,
        "decision": {
            "bar": PASS_BAR,
            "primary_roc_auc": primary["roc_auc"],
            "verdict": "PASS" if passed else "FAIL",
        },
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, sort_keys=False)
    log("wrote %s" % os.path.join(args.out_dir, "report.json"))


if __name__ == "__main__":
    main()
