"""Isolated-chunk judgment diagnostic on HateMM (descriptive, NOT preregistered).

Status: this is a DESCRIPTIVE DIAGNOSTIC. It has no pre-registration, no
decision bar and no interpretation grid. It cannot confirm a method; it can only
tell the goal loop whether a capability exists on this corpus.

Why it exists. The same diagnostic on HateClipSeg
(`scripts/duplex/isolated_chunk_diag.py`, commit b925e39) found no isolated-chunk
judgment capability at all: pooled AUC 0.533 for hate-span chunks against
normal chunks of the same videos. That leaves one obvious alternative reading --
HateClipSeg's segment gold is fine-grained and low-contrast, so perhaps the
capability exists but the corpus cannot show it. HateMM is the contrasting case:
its hate spans are coarse, annotator-marked stretches of clearly hateful speech
(median 13 s, median 70% of the audio), and LELA (arXiv 2602.09637) reports a
frame-level AUC of 0.726 on it as the multi-call ceiling. If isolated-segment
judgment is a real capability, this is the corpus where it should appear.

Design (mirrors the HateClipSeg run exactly). Each Whisper chunk of each test
video is scored on its own in one minimal text-only call: the frozen judge's
system message, the frozen union rules block, the chunk text, the frozen binary
question. No frames, no other chunk, no surrounding context. The score is the
answer margin at the final position, read from logits with nothing generated,

    z_chunk = logsumexp(logits[Yes ids]) - logsumexp(logits[No ids]).

The prompt builder, rules block and question are imported from the HateClipSeg
diagnostic, so the prompts are byte-identical across the two corpora; the run
records the same sha256 fingerprints.

Deviation from the HateClipSeg run: the titled arm is dropped. Every HateMM
record has an empty `Title` field, so a title arm would re-score identical
prompts. One call per chunk here, two there.

Chunk gold. HateMM marks only the hateful stretches; unmarked time is
implicitly non-hate. A chunk of a hate video is positive when more than half of
its own duration falls inside the union of that video's hate spans, negative
when less than half does. Every chunk of a non-hate video is negative. The two
contrasts are reported separately, because they answer different questions:

  * within-hate-video (the localization-relevant number): span chunks against
    non-span chunks of the same hate videos, pooled and macro-averaged per
    video;
  * cross-video: span chunks against chunks of non-hate videos, and the pooled
    all-negatives variant.

A strict sensitivity variant repeats the within-video contrast using only
chunks that are fully inside a span (overlap >= 0.9) against chunks with no
overlap at all, which removes boundary-straddling chunks.

Offline diagnostic calls. The two-call deployment cap in CLAUDE.md governs
methods; this script proposes no method.

Output: results/hatemm_localization/{per_chunk.jsonl, report.json, STATUS,
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

from score_duplex_probe import (  # noqa: E402
    SYSTEM_MESSAGE,
    build_binary_token_ids,
)
from sentinel_localization_pilot import (  # noqa: E402
    clean_chunk_text,
    rank_auc,
    usable_spans,
)
from isolated_chunk_diag import FROZEN_TEXT_SHA, user_text  # noqa: E402
from hatemm_span_gold import load_span_gold, read_split  # noqa: E402

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SATURATION = 13.0
STRICT_POS = 0.9

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "hatemm_localization")
CHUNKS_JSONL = os.path.join(OUT_DIR, "timestamped_chunks.jsonl")


# ----------------------------------------------------------------- chunk gold

def overlap_fraction(span, gold_spans):
    """Fraction of the chunk's own duration that lies inside the hate spans."""
    cs, ce = float(span[0]), float(span[1])
    dur = ce - cs
    if dur <= 0.0:
        return None
    ov = 0.0
    for gs, ge in gold_spans:
        ov += max(0.0, min(ce, float(ge)) - max(cs, float(gs)))
    return min(ov / dur, 1.0)


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
    remaining = [it for it in items
                 if (it["video_id"], it["chunk_index"]) not in done]
    logging.info("resume: %d chunks done, %d remaining"
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
        if i % 100 == 0:
            el = time.time() - t0
            status("forward %d/%d  %.1f s elapsed  %s"
                   % (i, len(remaining), el, time.strftime("%F %T")))
            logging.info("  %d/%d  %.1f s" % (i, len(remaining), el))
        rec = dict(it)
        z, n, sha = score(user_text(it["text"]))
        rec["z_isolated"] = z
        rec["prompt_tokens_isolated"] = n
        rec["prompt_sha256_isolated"] = sha
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
    fout.close()
    logging.info("forward done: %d chunks, %.1f s (%.3f s/chunk, 1 call each)"
                 % (len(remaining), time.time() - t0,
                    (time.time() - t0) / max(1, len(remaining))))


# ------------------------------------------------------------------- analysis

def describe(vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()),
            "median": float(np.median(a)),
            "sd": float(a.std(ddof=1)) if a.size > 1 else None,
            "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)),
            "min": float(a.min()), "max": float(a.max())}


def macro_auc(rows, pos_key="gold"):
    """Per-hate-video AUC of span chunks against non-span chunks of that video."""
    per_video = {}
    for r in rows:
        if r["video_label"] != "hate":
            continue
        per_video.setdefault(r["video_id"], []).append((r["z_isolated"], r[pos_key]))
    vals, used = [], []
    for vid, obs in sorted(per_video.items()):
        p = [z for z, g in obs if g == "span"]
        n = [z for z, g in obs if g == "nonspan"]
        if p and n:
            vals.append(rank_auc(p, n))
            used.append(vid)
    return vals, used


def contrast(pos, neg):
    return {"auc": rank_auc(list(pos), list(neg)),
            "n_pos": len(pos), "n_neg": len(neg),
            "score_pos": describe(pos), "score_neg": describe(neg)}


def analyze(per_chunk_path, counts):
    rows = []
    with open(per_chunk_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    z = np.array([r["z_isolated"] for r in rows], dtype=np.float64)
    ntok = np.array([r["chunk_tokens"] for r in rows], dtype=np.float64)
    rho, pval = stats.spearmanr(z, ntok)

    hv = [r for r in rows if r["video_label"] == "hate"]
    nhv = [r for r in rows if r["video_label"] == "non_hate"]
    span = [r["z_isolated"] for r in hv if r["gold"] == "span"]
    nonspan = [r["z_isolated"] for r in hv if r["gold"] == "nonspan"]
    nonhate = [r["z_isolated"] for r in nhv]

    macro, used = macro_auc(rows)
    strict_pos = [r["z_isolated"] for r in hv
                  if r["overlap_fraction"] >= STRICT_POS]
    strict_neg = [r["z_isolated"] for r in hv
                  if r["overlap_fraction"] <= 0.0]

    # Video-level readout: does the isolated chunk score aggregate into a
    # video verdict at all? Cheap descriptive check, not a method.
    per_vid = {}
    for r in rows:
        per_vid.setdefault(r["video_id"], {"label": r["video_label"], "z": []})
        per_vid[r["video_id"]]["z"].append(r["z_isolated"])
    vmax_p = [max(v["z"]) for v in per_vid.values() if v["label"] == "hate"]
    vmax_n = [max(v["z"]) for v in per_vid.values() if v["label"] == "non_hate"]
    vmean_p = [float(np.mean(v["z"])) for v in per_vid.values() if v["label"] == "hate"]
    vmean_n = [float(np.mean(v["z"])) for v in per_vid.values()
               if v["label"] == "non_hate"]

    return {
        "diagnostic": True,
        "preregistered": False,
        "dataset": "HateMM",
        "split": "test_clean",
        "model": MODEL,
        "arms": ["isolated"],
        "titled_arm_dropped_reason": "every HateMM record has an empty Title field",
        "counts": counts,
        "frozen_text_sha256": FROZEN_TEXT_SHA,
        "within_hate_video": {
            "pooled": contrast(span, nonspan),
            "macro_auc": float(np.mean(macro)) if macro else None,
            "macro_auc_sd": float(np.std(macro, ddof=1)) if len(macro) > 1 else None,
            "macro_auc_median": float(np.median(macro)) if macro else None,
            "n_videos_both_classes": len(used),
            "per_video_auc": dict(zip(used, macro)),
        },
        "within_hate_video_strict": {
            "note": "overlap >= %.2f vs overlap == 0" % STRICT_POS,
            "pooled": contrast(strict_pos, strict_neg),
        },
        "cross_video": {
            "span_vs_nonhate_video_chunks": contrast(span, nonhate),
            "span_vs_all_negatives": contrast(span, nonspan + nonhate),
            "hate_video_chunks_vs_nonhate_video_chunks":
                contrast(span + nonspan, nonhate),
        },
        "video_level": {
            "max_z_auc": rank_auc(vmax_p, vmax_n),
            "mean_z_auc": rank_auc(vmean_p, vmean_n),
            "n_hate_videos": len(vmax_p), "n_nonhate_videos": len(vmax_n),
        },
        "score_all": describe(z),
        "saturation": {
            "frac_abs_gt_13": float(np.mean(np.abs(z) > SATURATION)),
            "frac_gt_13": float(np.mean(z > SATURATION)),
            "frac_lt_minus_13": float(np.mean(z < -SATURATION)),
            "frac_yes_side": float(np.mean(z > 0.0)),
        },
        "spearman_z_vs_chunk_tokens": {"rho": float(rho), "p": float(pval)},
    }


# ------------------------------------------------------------------------ main

def build_items(counts):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    ids = read_split()
    gold, problems, _labels = load_span_gold(ids)
    chunks = {}
    with open(CHUNKS_JSONL) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                chunks[r["video_id"]] = r

    counts["split_videos"] = len(ids)
    counts["span_parse_problems"] = problems
    items = []
    for vid in ids:
        is_hate = vid.startswith("hate_video_")
        rec = chunks.get(vid)
        if not rec or not rec.get("chunks"):
            counts["videos_no_chunks"] += 1
            continue
        spans = usable_spans(rec)
        if spans is None:
            counts["videos_unusable_spans"] += 1
            continue
        gspans = gold.get(vid) or []
        if is_hate and not gspans:
            counts["hate_videos_without_span_gold"] += 1
            continue
        counts["videos_used_hate" if is_hate else "videos_used_non_hate"] += 1
        for k, c in enumerate(rec["chunks"]):
            counts["chunks_total"] += 1
            text = clean_chunk_text(c.get("text"))
            if not text:
                counts["chunks_empty_text"] += 1
                continue
            frac = overlap_fraction(spans[k], gspans) if is_hate else 0.0
            if frac is None:
                counts["chunks_bad_span"] += 1
                continue
            if is_hate:
                lab = "span" if frac > 0.5 else "nonspan"
                counts["chunks_hate_span" if lab == "span"
                       else "chunks_hate_nonspan"] += 1
            else:
                lab = "nonspan"
                counts["chunks_non_hate_video"] += 1
            items.append({
                "video_id": vid,
                "video_label": "hate" if is_hate else "non_hate",
                "chunk_index": k,
                "span": [float(spans[k][0]), float(spans[k][1])],
                "overlap_fraction": frac,
                "text": text,
                "gold": lab,
                "chunk_tokens": len(tok(text, add_special_tokens=False)["input_ids"]),
            })
    counts["chunks_scored"] = len(items)
    return items


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

    status("gold")
    counts = {k: 0 for k in (
        "videos_no_chunks", "videos_unusable_spans",
        "hate_videos_without_span_gold", "videos_used_hate",
        "videos_used_non_hate", "chunks_total", "chunks_empty_text",
        "chunks_bad_span", "chunks_hate_span", "chunks_hate_nonspan",
        "chunks_non_hate_video")}
    items = build_items(counts)
    if args.limit_videos:
        keep = sorted({it["video_id"] for it in items})[:args.limit_videos]
        items = [it for it in items if it["video_id"] in set(keep)]
    logging.info("chunk counts: %s" % json.dumps(counts, indent=2))
    logging.info("frozen text sha256: %s" % json.dumps(FROZEN_TEXT_SHA, indent=2))

    if not args.analyze_only:
        status("forward")
        run_forward(items, per_chunk_path, status)

    status("analyze")
    report = analyze(per_chunk_path, counts)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    head = {k: report[k] for k in ("counts", "within_hate_video",
                                   "within_hate_video_strict", "cross_video",
                                   "video_level", "saturation",
                                   "spearman_z_vs_chunk_tokens")}
    head["within_hate_video"] = {k: v for k, v in head["within_hate_video"].items()
                                 if k != "per_video_auc"}
    logging.info(json.dumps(head, indent=2))
    status("DONE")
    with open(os.path.join(args.out_dir, "DONE"), "w") as f:
        f.write(time.strftime("%F %T") + "\n")


if __name__ == "__main__":
    main()
