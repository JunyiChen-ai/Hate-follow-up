"""Isolated-chunk judgment diagnostic (descriptive, NOT a preregistered test).

Status: this is a DESCRIPTIVE DIAGNOSTIC. It has no pre-registration, no
decision bar, and no interpretation grid. It cannot confirm a method; it can
only tell the goal loop whether a capability exists at this scale.

Why it exists. Two preregistered localization pilots died with the same
mechanism (docs/duplex/SENTINEL_LOCALIZATION_NOTE.md): with causal attention
over the whole video, every within-video probe returns the model's single
global verdict smeared over the timeline. What that leaves open is whether the
frozen judge can judge a transcript segment at all when the rest of the video
is absent. Every remaining localization direction (block-diagonal isolation,
per-segment calls, any local-evidence method) is gated on the answer.

Design. Each transcript chunk is scored on its own, in one minimal text-only
call: the frozen judge's system message, the frozen union rules block, the
chunk text, and the frozen binary question. No frames, no title, no other
chunk, no surrounding context. The score is the model's own answer margin at
the final position,

    z_chunk = logsumexp(logits[Yes ids]) - logsumexp(logits[No ids]),

read from the model's logits with nothing generated. A second arm repeats the
identical calls with only the video's title prepended, which measures how much
the cheapest possible context moves the score.

These are offline diagnostic calls. The two-call deployment cap in CLAUDE.md
governs methods; this script proposes no method.

Frozen material is imported, never copied: YOUTUBE_RULES, SYSTEM_MESSAGE and
build_binary_token_ids come from the frozen judge; the cohort, the gold
segments, the timestamped chunks and the rank-AUC helper come from the sentinel
localization pilot (committed 1505911), so the sample is a subset of exactly
the 326 videos that pilot ran on.

Output: results/isolated_chunk_diag/{per_chunk.jsonl, report.json, STATUS,
DONE}; the caller redirects stdout to run.log.
"""

import argparse
import hashlib
import json
import logging
import os
import random
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

from score_duplex_probe import (  # noqa: E402
    SYSTEM_MESSAGE,
    YOUTUBE_RULES,
    build_binary_token_ids,
)
from data_utils import load_annotations  # noqa: E402
from sentinel_localization_pilot import (  # noqa: E402
    build_cohort,
    clean_chunk_text,
    is_normal_only,
    is_offensive_union,
    rank_auc,
    usable_spans,
)

DATASET = "HateClipSeg"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SAMPLE_SEED = 20260812
SAMPLE_SIZE = 80
SATURATION = 13.0

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "isolated_chunk_diag")

# ---------------------------------------------------------------- frozen text
# The rules block is the frozen judge's YOUTUBE_RULES verbatim, under the frozen
# judge's own lead-in sentence; the question is the frozen judge's own binary
# question, narrowed from the video to the excerpt.

RULES_LEAD_IN = ("Based on this platform's hate speech policy, the following "
                 "types of content are prohibited:")
RULES_BLOCK = RULES_LEAD_IN + "\n" + YOUTUBE_RULES

QUESTION = ("Does this excerpt violate the rules above?\n\n"
            "Answer \"Yes\" or \"No\".")


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def user_text(chunk_text, title=None):
    """Rules block, then the excerpt (optionally titled), then the question."""
    head = "" if title is None else "Video title: %s\n" % title
    return "%s\n\n%sTranscript excerpt from a video:\n%s\n\n%s" % (
        RULES_BLOCK, head, chunk_text, QUESTION)


FROZEN_TEXT_SHA = {
    "rules_block": sha(RULES_BLOCK),
    "question": sha(QUESTION),
    "system_message": sha(SYSTEM_MESSAGE),
    "user_text_template_notitle": sha(user_text("<CHUNK>")),
    "user_text_template_title": sha(user_text("<CHUNK>", "<TITLE>")),
}


# ----------------------------------------------------------------- chunk gold

def chunk_gold_label(span, labels, spans):
    """Overlap-weighted majority gold label of one chunk.

    Returns (label, detail) where label is "offensive", "normal" or None.
    Every gold segment overlapping the chunk contributes its overlap duration to
    one of three buckets: offensive-union, normal-only, or neither. The chunk is
    offensive if the offensive bucket holds more than half of the total
    overlapped gold time, normal if the normal bucket does, otherwise ambiguous.
    """
    cs, ce = float(span[0]), float(span[1])
    t_off = t_norm = t_other = 0.0
    for lab, (gs, ge) in zip(labels, spans):
        ov = min(ce, float(ge)) - max(cs, float(gs))
        if ov <= 0.0:
            continue
        if is_offensive_union(lab):
            t_off += ov
        elif is_normal_only(lab):
            t_norm += ov
        else:
            t_other += ov
    total = t_off + t_norm + t_other
    detail = {"t_off": t_off, "t_norm": t_norm, "t_other": t_other,
              "t_total": total}
    if total <= 0.0:
        return None, detail
    if t_off > 0.5 * total:
        return "offensive", detail
    if t_norm > 0.5 * total:
        return "normal", detail
    return None, detail


# --------------------------------------------------------------------- sample

def draw_sample(cohort):
    """Fixed random sample of SAMPLE_SIZE videos from the sentinel cohort."""
    pool = sorted(cohort)
    if len(pool) <= SAMPLE_SIZE:
        return pool
    return sorted(random.Random(SAMPLE_SEED).sample(pool, SAMPLE_SIZE))


# -------------------------------------------------------------------- scoring

def run_forward(items, per_chunk_path, status):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    done = set()
    if os.path.exists(per_chunk_path):
        with open(per_chunk_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["video_id"], r["chunk_index"]))
    remaining = [it for it in items if (it["video_id"], it["chunk_index"]) not in done]
    logging.info("resume: %d chunks done, %d remaining" % (len(done), len(remaining)))
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

    def score(text):
        msgs = [{"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt = processor.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, use_cache=False, logits_to_keep=1)
            lg = out.logits[0, -1, :].float()
            z = float(torch.logsumexp(lg[yes_t], 0) - torch.logsumexp(lg[no_t], 0))
        return z, int(enc["input_ids"].shape[1]), hashlib.sha256(
            prompt.encode()).hexdigest()

    t0 = time.time()
    fout = open(per_chunk_path, "a")
    for i, it in enumerate(remaining):
        if i % 50 == 0:
            el = time.time() - t0
            status("forward %d/%d  %.1f s elapsed  %s"
                   % (i, len(remaining), el, time.strftime("%F %T")))
            logging.info("  %d/%d  %.1f s" % (i, len(remaining), el))
        rec = dict(it)
        z_a, n_a, sha_a = score(user_text(it["text"]))
        z_b, n_b, sha_b = score(user_text(it["text"], it["title"]))
        rec["z_isolated"] = z_a
        rec["prompt_tokens_isolated"] = n_a
        rec["prompt_sha256_isolated"] = sha_a
        rec["z_titled"] = z_b
        rec["prompt_tokens_titled"] = n_b
        rec["prompt_sha256_titled"] = sha_b
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
    fout.close()
    logging.info("forward done: %d chunks, %.1f s (%.3f s/chunk, 2 calls each)"
                 % (len(remaining), time.time() - t0,
                    (time.time() - t0) / max(1, len(remaining))))


# ------------------------------------------------------------------- analysis

def describe(vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()),
            "median": float(np.median(a)), "sd": float(a.std(ddof=1))
            if a.size > 1 else None,
            "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)),
            "min": float(a.min()), "max": float(a.max())}


def arm_report(rows, key):
    z = np.array([r[key] for r in rows], dtype=np.float64)
    lab = np.array([r["gold"] for r in rows])
    pos = z[lab == "offensive"]
    neg = z[lab == "normal"]

    per_video = {}
    for r in rows:
        per_video.setdefault(r["video_id"], []).append((r[key], r["gold"]))
    macro, nboth = [], 0
    for vid, vals in sorted(per_video.items()):
        p = [v for v, g in vals if g == "offensive"]
        n = [v for v, g in vals if g == "normal"]
        if p and n:
            nboth += 1
            macro.append(rank_auc(p, n))

    ntok = np.array([r["chunk_tokens"] for r in rows], dtype=np.float64)
    rho, pval = stats.spearmanr(z, ntok)

    return {
        "pooled_auc": rank_auc(list(pos), list(neg)),
        "n_pos": int(pos.size), "n_neg": int(neg.size),
        "macro_auc_per_video": float(np.mean(macro)) if macro else None,
        "macro_auc_sd": float(np.std(macro, ddof=1)) if len(macro) > 1 else None,
        "n_videos_both_classes": nboth,
        "macro_auc_median": float(np.median(macro)) if macro else None,
        "score_offensive": describe(pos),
        "score_normal": describe(neg),
        "score_all": describe(z),
        "frac_saturated_abs_gt_13": float(np.mean(np.abs(z) > SATURATION)),
        "frac_saturated_pos": float(np.mean(z > SATURATION)),
        "frac_saturated_neg": float(np.mean(z < -SATURATION)),
        "frac_yes_side": float(np.mean(z > 0.0)),
        "spearman_z_vs_chunk_tokens": {"rho": float(rho), "p": float(pval)},
    }


def analyze(per_chunk_path, counts, sample):
    rows = []
    with open(per_chunk_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows = [r for r in rows if r.get("gold") in ("offensive", "normal")]

    za = np.array([r["z_isolated"] for r in rows])
    zb = np.array([r["z_titled"] for r in rows])
    agree = float(np.mean(np.sign(za) == np.sign(zb)))
    rho_ab, _ = stats.spearmanr(za, zb)

    return {
        "diagnostic": True,
        "preregistered": False,
        "model": MODEL,
        "sample_seed": SAMPLE_SEED,
        "sample_size": len(sample),
        "counts": counts,
        "frozen_text_sha256": FROZEN_TEXT_SHA,
        "arm_isolated": arm_report(rows, "z_isolated"),
        "arm_titled": arm_report(rows, "z_titled"),
        "arm_agreement": {"sign_agreement": agree,
                          "spearman_rho": float(rho_ab),
                          "mean_shift_titled_minus_isolated":
                              float(np.mean(zb - za)),
                          "mean_abs_shift": float(np.mean(np.abs(zb - za)))},
        "sample_video_ids": sample,
    }


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    status_path = os.path.join(args.out_dir, "STATUS")
    per_chunk_path = os.path.join(args.out_dir, "per_chunk.jsonl")
    report_path = os.path.join(args.out_dir, "report.json")

    def status(s):
        with open(status_path, "w") as f:
            f.write(s + "\n")

    status("cohort")
    cohort, cohort_counts, _excl, gold, chunks = build_cohort()
    sample = draw_sample(cohort)
    if args.limit_videos:
        sample = sample[:args.limit_videos]
    logging.info("sentinel cohort %d videos; sample %d (seed %d)"
                 % (len(cohort), len(sample), SAMPLE_SEED))
    logging.info("frozen text sha256: %s" % json.dumps(FROZEN_TEXT_SHA, indent=2))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    annotations = load_annotations(DATASET)

    items = []
    counts = {"cohort_videos": len(cohort), "sample_videos": len(sample),
              "chunks_total": 0, "chunks_offensive": 0, "chunks_normal": 0,
              "chunks_dropped_ambiguous": 0, "chunks_dropped_no_gold_overlap": 0,
              "chunks_empty_text": 0}
    for vid in sample:
        rec = chunks[vid]
        spans = usable_spans(rec)
        labels, gspans = gold[vid]
        title = (annotations.get(vid) or {}).get("title", "") or ""
        for k, c in enumerate(rec["chunks"]):
            text = clean_chunk_text(c.get("text"))
            counts["chunks_total"] += 1
            if not text:
                counts["chunks_empty_text"] += 1
                continue
            lab, detail = chunk_gold_label(spans[k], labels, gspans)
            if lab is None:
                if detail["t_total"] <= 0.0:
                    counts["chunks_dropped_no_gold_overlap"] += 1
                else:
                    counts["chunks_dropped_ambiguous"] += 1
                continue
            counts["chunks_offensive" if lab == "offensive" else "chunks_normal"] += 1
            items.append({
                "video_id": vid, "chunk_index": k, "span": list(spans[k]),
                "text": text, "title": title, "gold": lab,
                "gold_overlap": detail,
                "chunk_tokens": len(tok(text, add_special_tokens=False)["input_ids"]),
            })
    counts["chunks_labeled"] = len(items)
    logging.info("chunk counts: %s" % json.dumps(counts, indent=2))

    if not args.analyze_only:
        status("forward")
        run_forward(items, per_chunk_path, status)

    status("analyze")
    report = analyze(per_chunk_path, counts, sample)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    head = {k: report[k] for k in ("counts", "arm_isolated", "arm_titled",
                                   "arm_agreement")}
    logging.info(json.dumps(head, indent=2))
    status("DONE")
    with open(os.path.join(args.out_dir, "DONE"), "w") as f:
        f.write(time.strftime("%F %T") + "\n")


if __name__ == "__main__":
    main()
