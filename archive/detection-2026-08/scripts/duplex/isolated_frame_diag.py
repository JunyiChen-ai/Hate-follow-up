"""Isolated-frame judgment diagnostic (descriptive, NOT a preregistered test).

Status: this is a DESCRIPTIVE DIAGNOSTIC, the visual counterpart of
scripts/duplex/isolated_chunk_diag.py (commit b925e39) and
scripts/duplex/isolated_window_diag.py (commit 3864116). It has no
pre-registration, no decision bar and no interpretation grid. It cannot confirm
a method; it can only tell the goal loop whether a capability exists on this
channel.

Question. The text side of isolated judgment has already been measured on this
backbone: a single ASR chunk scored on its own separates offensive from normal
material at pooled AUC 0.533, and grouping chunks into >=45 s windows moves the
pooled number to 0.690 largely by making the evaluation unit coarser. The
visual channel has never been tested in isolation. Every localization design
left on the table -- per-frame gating, frame selection, visual sentinels --
assumes the frozen judge can look at one frame, with no transcript and no other
frame, and say something label-relevant about it. This script measures exactly
that assumption and nothing else.

Design. Each of a video's 16 stored frames is scored on its own in one minimal
call: the frozen judge's system message, the frozen union rules block, ONE
image, and a frozen single-frame question. No transcript, no title, no other
frame, no temporal context. The score is the model's own answer margin at the
final position,

    z_frame = logsumexp(logits[Yes ids]) - logsumexp(logits[No ids]),

read from the logits with nothing generated. The pixel budget is the frozen
extractor's (MIN_PIXELS, MAX_PIXELS), so each frame reaches the model at the
same resolution it does inside the frozen judge's own 16-frame call.

Frame gold. Frame i of a video carries the nominal timestamp
(i + 0.5) / n_frames * duration, with duration taken by the window
diagnostic's convention (wav_duration, container_duration as fallback, from
results/interleaved_timeline/hateclipseg/timestamped_chunks.jsonl). The frame's
gold label is the label of the unique gold segment strictly containing that
timestamp: offensive-union or normal-only. Frames whose timestamp lands on a
segment boundary, outside every segment, inside more than one segment, or
inside a segment that is neither offensive-union nor normal-only are dropped,
and each drop reason is counted in the report.

Position control. A per-frame score can look discriminative for a reason that
has nothing to do with what the frame shows: HateClipSeg's offensive material
is not uniformly placed on the timeline, so a model whose margin drifted with
frame index alone would produce an AUC above chance. The report therefore
carries Spearman(z, frame index) computed within each video and summarized
across videos, the same correlation pooled, and the AUC that frame index alone
achieves against the same gold. A frame-index AUC near the measured z AUC means
the visual channel added nothing over position.

These are offline diagnostic calls. The two-call deployment cap in CLAUDE.md
governs methods; this script proposes no method.

Frozen material is imported, never copied: the system message and the binary
token ids come from the frozen judge; the rules block, the sample draw, the
saturation threshold and the summary helper come from isolated_chunk_diag; the
cohort, the gold segments, the timestamped chunks and the rank-AUC helper come
through it from the sentinel localization pilot, so the sample is exactly the
same 80 videos drawn with seed 20260812. The frame paths and the pixel budget
come from the frozen extractor.

Output: results/isolated_frame_diag/{per_frame.jsonl, report.json, STATUS,
DONE}; the caller redirects stdout to run.log.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

_THIS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _THIS)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "duplex"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "our_method"))

from extract_duplex_readout import (  # noqa: E402
    MAX_PIXELS,
    MIN_PIXELS,
    resolve_frames,
)
from score_duplex_probe import (  # noqa: E402
    SYSTEM_MESSAGE,
    build_binary_token_ids,
)
from sentinel_localization_pilot import (  # noqa: E402
    build_cohort,
    is_normal_only,
    is_offensive_union,
    rank_auc,
)
from isolated_chunk_diag import (  # noqa: E402
    DATASET,
    MODEL,
    RULES_BLOCK,
    SAMPLE_SEED,
    SATURATION,
    describe,
    draw_sample,
    sha,
)

NUM_FRAMES = 16
BOUNDARY_EPS = 1e-6   # a timestamp this close to a segment edge counts as on it

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "isolated_frame_diag")

# ---------------------------------------------------------------- frozen text
# The rules block is imported verbatim from the chunk diagnostic (which took it
# from the frozen judge's YOUTUBE_RULES under the frozen judge's own lead-in).
# The question is the frozen judge's binary question narrowed to one frame.

FRAME_QUESTION = ("This is a single frame from a video.\n"
                  "Does this frame violate the rules above? "
                  "Answer \"Yes\" or \"No\".")

FROZEN_TEXT_SHA = {
    "rules_block": sha(RULES_BLOCK),
    "frame_question": sha(FRAME_QUESTION),
    "system_message": sha(SYSTEM_MESSAGE),
}


def build_messages(rules, question):
    """System turn, then one user turn: rules text, one image, question text."""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": [
            {"type": "text", "text": rules},
            {"type": "image"},
            {"type": "text", "text": question},
        ]},
    ]


# ----------------------------------------------------------------- frame gold

def frame_gold_label(t, labels, spans):
    """Gold label of the segment strictly containing timestamp t.

    Returns (label, reason) with label in {"offensive", "normal", None}. The
    reason names the drop when label is None: "boundary" (t sits on a segment
    edge), "outside" (no segment contains t), "multi" (more than one segment
    contains t) or "other_label" (the containing segment is neither
    offensive-union nor normal-only).
    """
    hits = []
    for lab, (gs, ge) in zip(labels, spans):
        gs, ge = float(gs), float(ge)
        if abs(t - gs) <= BOUNDARY_EPS or abs(t - ge) <= BOUNDARY_EPS:
            return None, "boundary"
        if gs < t < ge:
            hits.append(lab)
    if not hits:
        return None, "outside"
    if len(hits) > 1:
        return None, "multi"
    lab = hits[0]
    if is_offensive_union(lab):
        return "offensive", None
    if is_normal_only(lab):
        return "normal", None
    return None, "other_label"


# -------------------------------------------------------------------- scoring

def run_forward(items, per_frame_path, status):
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    done = set()
    if os.path.exists(per_frame_path):
        with open(per_frame_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["video_id"], r["frame_index"]))
    remaining = [it for it in items
                 if (it["video_id"], it["frame_index"]) not in done]
    logging.info("resume: %d frames done, %d remaining"
                 % (len(done), len(remaining)))
    if not remaining:
        return

    processor = AutoProcessor.from_pretrained(MODEL)
    tokenizer = processor.tokenizer
    ids = build_binary_token_ids(tokenizer)
    yes_ids, no_ids = sorted(ids["Yes"]), sorted(ids["No"])
    logging.info("Yes ids %s No ids %s" % (yes_ids, no_ids))

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    yes_t = torch.tensor(yes_ids, device=model.device)
    no_t = torch.tensor(no_ids, device=model.device)

    size_kwarg = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    msgs = build_messages(RULES_BLOCK, FRAME_QUESTION)
    prompt = processor.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    logging.info("prompt sha256 %s" % prompt_sha)

    t0 = time.time()
    fout = open(per_frame_path, "a")
    for i, it in enumerate(remaining):
        if i % 100 == 0:
            el = time.time() - t0
            status("forward %d/%d  %.1f s elapsed  %s"
                   % (i, len(remaining), el, time.strftime("%F %T")))
            logging.info("  %d/%d  %.1f s" % (i, len(remaining), el))
        with Image.open(it["frame_path"]) as im:
            image = im.convert("RGB")
        inputs = processor(text=[prompt], images=[image],
                           return_tensors="pt", size=size_kwarg)
        seq = int(inputs["input_ids"].shape[1])
        inputs = inputs.to(model.device)
        with torch.no_grad():
            out = model(**inputs, use_cache=False, logits_to_keep=1)
            lg = out.logits[0, -1, :].float()
            z = float(torch.logsumexp(lg[yes_t], 0)
                      - torch.logsumexp(lg[no_t], 0))
        image.close()
        rec = dict(it)
        rec["z_frame"] = z
        rec["seq_len"] = seq
        rec["prompt_sha256"] = prompt_sha
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
    fout.close()
    dt = time.time() - t0
    logging.info("forward done: %d frames, %.1f s (%.3f s/frame, 1 call each)"
                 % (len(remaining), dt, dt / max(1, len(remaining))))


# ------------------------------------------------------------------- analysis

def position_control(rows):
    """Is the margin tracking where the frame sits rather than what it shows?"""
    z = np.array([r["z_frame"] for r in rows], dtype=np.float64)
    idx = np.array([r["frame_index"] for r in rows], dtype=np.float64)
    rel = np.array([r["rel_time"] for r in rows], dtype=np.float64)
    rho_pool, p_pool = stats.spearmanr(z, idx)
    rho_rel, p_rel = stats.spearmanr(z, rel)

    per_video = {}
    for r in rows:
        per_video.setdefault(r["video_id"], []).append(r)
    rhos, n_deg = [], 0
    for _v, rs in sorted(per_video.items()):
        if len(rs) < 4:
            continue
        zz = [x["z_frame"] for x in rs]
        ii = [x["frame_index"] for x in rs]
        if len(set(zz)) < 2:
            n_deg += 1
            continue
        rho, _ = stats.spearmanr(zz, ii)
        if np.isfinite(rho):
            rhos.append(float(rho))

    pos_idx = [r["frame_index"] for r in rows if r["gold"] == "offensive"]
    neg_idx = [r["frame_index"] for r in rows if r["gold"] == "normal"]
    pos_rel = [r["rel_time"] for r in rows if r["gold"] == "offensive"]
    neg_rel = [r["rel_time"] for r in rows if r["gold"] == "normal"]

    return {
        "spearman_z_vs_frame_index_pooled": {"rho": float(rho_pool),
                                             "p": float(p_pool)},
        "spearman_z_vs_rel_time_pooled": {"rho": float(rho_rel),
                                          "p": float(p_rel)},
        "per_video_spearman_z_vs_frame_index": describe(rhos),
        "per_video_spearman_n_videos": len(rhos),
        "per_video_constant_z_videos": n_deg,
        "per_video_spearman_frac_positive": (
            float(np.mean([r > 0 for r in rhos])) if rhos else None),
        "frame_index_alone_pooled_auc": rank_auc(pos_idx, neg_idx),
        "rel_time_alone_pooled_auc": rank_auc(pos_rel, neg_rel),
        "note": ("frame_index_alone_pooled_auc is the AUC a pure position "
                 "predictor reaches against the same gold; if it matches the "
                 "z AUC the image channel bought nothing."),
    }


def arm_report(rows, key):
    z = np.array([r[key] for r in rows], dtype=np.float64)
    lab = np.array([r["gold"] for r in rows])
    pos = z[lab == "offensive"]
    neg = z[lab == "normal"]

    per_video = {}
    for r in rows:
        per_video.setdefault(r["video_id"], []).append((r[key], r["gold"]))
    macro, nboth = [], 0
    for _vid, vals in sorted(per_video.items()):
        p = [v for v, g in vals if g == "offensive"]
        n = [v for v, g in vals if g == "normal"]
        if p and n:
            nboth += 1
            macro.append(rank_auc(p, n))

    rng = np.random.default_rng(0)
    boot = [rank_auc(rng.choice(pos, pos.size), rng.choice(neg, neg.size))
            for _ in range(4000)] if pos.size and neg.size else []

    return {
        "pooled_auc": rank_auc(list(pos), list(neg)),
        "pooled_auc_boot95": ([float(np.percentile(boot, 2.5)),
                               float(np.percentile(boot, 97.5))]
                              if boot else None),
        "n_pos": int(pos.size), "n_neg": int(neg.size),
        "macro_auc_per_video": float(np.mean(macro)) if macro else None,
        "macro_auc_sd": float(np.std(macro, ddof=1)) if len(macro) > 1 else None,
        "macro_auc_median": float(np.median(macro)) if macro else None,
        "n_videos_both_classes": nboth,
        "macro_auc_degenerate_fraction": (
            float(np.mean([a in (0.0, 1.0) for a in macro])) if macro else None),
        "score_offensive": describe(pos),
        "score_normal": describe(neg),
        "score_all": describe(z),
        "mean_gap_off_minus_norm": (float(pos.mean() - neg.mean())
                                    if pos.size and neg.size else None),
        "frac_saturated_abs_gt_13": float(np.mean(np.abs(z) > SATURATION)),
        "frac_saturated_pos": float(np.mean(z > SATURATION)),
        "frac_saturated_neg": float(np.mean(z < -SATURATION)),
        "frac_yes_side": float(np.mean(z > 0.0)),
    }


# The text-side results this diagnostic is read against, quoted verbatim from
# the two companion reports so the three granularities sit in one file.
TEXT_SIDE_REFERENCE = {
    "chunk": {
        "source": "results/isolated_chunk_diag/report.json",
        "n_units": 1157, "n_pos": 731, "n_neg": 426,
        "isolated_pooled_auc": 0.5325475424365619,
        "isolated_macro_auc_per_video": 0.5711021840296147,
        "frac_saturated_abs_gt_13": 0.6542783059636992,
    },
    "window": {
        "source": "results/isolated_window_diag/report.json",
        "n_units": 195, "n_pos": 113, "n_neg": 82,
        "isolated_pooled_auc": 0.6901036045758687,
        "isolated_macro_auc_per_video": 0.7623456790123457,
        "frac_saturated_abs_gt_13": 0.558974358974359,
    },
}


def analyze(per_frame_path, counts, geometry, sample):
    rows = []
    with open(per_frame_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows = [r for r in rows if r.get("gold") in ("offensive", "normal")]

    return {
        "diagnostic": True,
        "preregistered": False,
        "model": MODEL,
        "dataset": DATASET,
        "sample_seed": SAMPLE_SEED,
        "sample_size": len(sample),
        "unit": "one stored frame, scored alone",
        "pixel_budget": {"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
        "frame_time_rule": ("(i + 0.5) / n_frames * duration, duration from "
                            "wav_duration (container_duration fallback)"),
        "counts": counts,
        "geometry": geometry,
        "frozen_text_sha256": FROZEN_TEXT_SHA,
        "arm_isolated_frame": arm_report(rows, "z_frame"),
        "position_control": position_control(rows),
        "text_side_reference": TEXT_SIDE_REFERENCE,
        "sample_video_ids": sample,
    }


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--build-only", action="store_true",
                    help="build frame items and print counts, run no model")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    status_path = os.path.join(args.out_dir, "STATUS")
    per_frame_path = os.path.join(args.out_dir, "per_frame.jsonl")
    report_path = os.path.join(args.out_dir, "report.json")

    def status(s):
        with open(status_path, "w") as f:
            f.write(s + "\n")

    status("cohort")
    cohort, _cohort_counts, _excl, gold, chunks = build_cohort()
    sample = draw_sample(cohort)
    if args.limit_videos:
        sample = sample[:args.limit_videos]
    logging.info("sentinel cohort %d videos; sample %d (seed %d)"
                 % (len(cohort), len(sample), SAMPLE_SEED))
    logging.info("frozen text sha256: %s" % json.dumps(FROZEN_TEXT_SHA, indent=2))

    items = []
    counts = {"cohort_videos": len(cohort), "sample_videos": len(sample),
              "videos_without_frames": 0, "videos_with_frames": 0,
              "frames_total": 0, "frames_offensive": 0, "frames_normal": 0,
              "frames_dropped_boundary": 0, "frames_dropped_outside": 0,
              "frames_dropped_multi_segment": 0,
              "frames_dropped_other_label": 0}
    durations, n_frames_per_video, labeled_per_video = [], [], []

    for vid in sample:
        frame_paths = resolve_frames(vid, DATASET, NUM_FRAMES)
        if not frame_paths:
            counts["videos_without_frames"] += 1
            logging.info("  %s: no frames" % vid)
            continue
        counts["videos_with_frames"] += 1
        n_f = len(frame_paths)
        n_frames_per_video.append(n_f)
        rec = chunks[vid]
        dur = rec.get("wav_duration") or rec.get("container_duration")
        dur = float(dur) if dur else None
        if not dur or dur <= 0.0:
            raise SystemExit("%s: no usable duration" % vid)
        durations.append(dur)
        labels, gspans = gold[vid]
        n_lab = 0
        counts["frames_total"] += n_f
        for i, p in enumerate(frame_paths):
            rel = (i + 0.5) / float(n_f)
            t = rel * dur
            lab, reason = frame_gold_label(t, labels, gspans)
            if lab is None:
                counts["frames_dropped_%s" % {
                    "boundary": "boundary", "outside": "outside",
                    "multi": "multi_segment",
                    "other_label": "other_label"}[reason]] += 1
                continue
            n_lab += 1
            counts["frames_offensive" if lab == "offensive"
                   else "frames_normal"] += 1
            items.append({
                "video_id": vid, "frame_index": i, "n_frames": n_f,
                "frame_path": p, "timestamp": t, "rel_time": rel,
                "duration": dur, "gold": lab,
            })
        labeled_per_video.append(n_lab)
    counts["frames_labeled"] = len(items)

    geometry = {
        "duration_s": describe(durations),
        "frames_per_video": describe(n_frames_per_video),
        "labeled_frames_per_video": describe(labeled_per_video),
        "gold_segments_per_video": describe(
            [len(gold[v][0]) for v in sample if v in gold]),
    }
    logging.info("frame counts: %s" % json.dumps(counts, indent=2))
    logging.info("geometry: %s" % json.dumps(geometry, indent=2))

    if args.build_only:
        status("build-only")
        return

    if not args.analyze_only:
        status("forward")
        run_forward(items, per_frame_path, status)

    status("analyze")
    report = analyze(per_frame_path, counts, geometry, sample)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    head = {k: report[k] for k in ("counts", "geometry", "arm_isolated_frame",
                                   "position_control", "text_side_reference")}
    logging.info(json.dumps(head, indent=2))
    status("DONE")
    with open(os.path.join(args.out_dir, "DONE"), "w") as f:
        f.write(time.strftime("%F %T") + "\n")


if __name__ == "__main__":
    main()
