"""Isolated-window judgment diagnostic (descriptive, NOT a preregistered test).

Status: this is a DESCRIPTIVE DIAGNOSTIC extending
scripts/duplex/isolated_chunk_diag.py (commit b925e39). It has no
pre-registration, no decision bar and no interpretation grid. It cannot confirm
a method; it can only tell the goal loop whether a capability exists at this
granularity.

Question. The chunk-level diagnostic scored each ASR chunk on its own and found
almost no discrimination: pooled AUC 0.533, macro AUC over videos 0.571, with
65% of chunks saturated at |z| > 13 and Spearman(z, chunk tokens) = 0.50 — the
score tracked how much text the excerpt held more than what the text said. One
reading of that death is that a single ASR chunk is simply too short to carry a
judgeable event. This script tests that reading directly by rebuilding the same
material at a coarser granularity: consecutive chunks of the same video are
grouped into windows spanning at least 45 s, and each window is scored on its
own with the identical frozen prompt. If the capability is real but was starved
of evidence, window AUC should rise well above the chunk numbers. If window AUC
sits at the same place, granularity is not the constraint.

Windowing. Greedy and left-to-right over each video's chunks in temporal order:
a window absorbs chunks until its hull span (last chunk end minus first chunk
start) reaches 45 s, then closes; the final window merges into its predecessor
if its hull span is under 20 s. No chunk is used twice and no chunk is skipped.

Window gold. The chunk rule generalized: every gold segment overlapping any
member chunk's span contributes its overlap duration to the offensive-union,
normal-only or neither bucket, and the window is offensive (normal) when that
bucket holds more than half the overlapped gold time. Overlap is summed over
member chunk spans rather than the window hull, so untranscribed gaps inside a
window carry no vote; a hull-based label is computed alongside and its
disagreement count reported.

Arms. Two arms, each window scored independently in one minimal text-only call:
(a) isolated — the frozen judge's system message, the frozen union rules block,
the window text and the frozen binary question, nothing else; (b) titled — the
identical call with the video's title prepended. The score is the answer margin
at the final position, z = logsumexp(logits[Yes ids]) - logsumexp(logits[No
ids]), read from the logits with nothing generated.

Aggregation control. A window can look more judgeable than its chunks for two
different reasons: the model really judges better when given more text, or the
coarser evaluation unit is simply easier because an offensive window only has
to contain one offensive chunk. The control separates them without a single
extra forward pass. Each window is re-scored from the chunk diagnostic's own
per-chunk margins, by the mean and by the max over its member chunks, and the
resulting AUCs are reported next to the window-call AUCs. If the max over old
chunk scores already matches the window call, the longer excerpt bought no new
capability. Window token count, chunk count and hull span are each scored as
lone predictors for the same reason.

Known limitation of arm (b), carried over from the chunk diagnostic and
verified here: HateClipSeg's annotation file supplies no titles (all 435
entries have an empty Title field), so the titled arm prepends an empty "Video
title:" line. It is therefore not a context arm but a null prompt perturbation,
and it is kept only because the chunk-level titled numbers it is compared
against were produced the same way. The report records title availability
explicitly.

These are offline diagnostic calls. The two-call deployment cap in CLAUDE.md
governs methods; this script proposes no method.

Frozen material is imported, never copied: the rules block, the question, the
prompt builder and the summary helper come from isolated_chunk_diag; the
cohort, the gold segments, the timestamped chunks and the rank-AUC helper come
through it from the sentinel localization pilot, so the sample is exactly the
same 80 videos drawn with seed 20260812.

Output: results/isolated_window_diag/{per_window.jsonl, report.json, STATUS,
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
from data_utils import load_annotations  # noqa: E402
from sentinel_localization_pilot import (  # noqa: E402
    build_cohort,
    clean_chunk_text,
    is_normal_only,
    is_offensive_union,
    rank_auc,
    usable_spans,
)
from isolated_chunk_diag import (  # noqa: E402
    DATASET,
    FROZEN_TEXT_SHA,
    MODEL,
    SAMPLE_SEED,
    SATURATION,
    describe,
    draw_sample,
    user_text,
)

WINDOW_CLOSE_SPAN = 45.0   # close a window once its hull span reaches this
WINDOW_MIN_TAIL = 20.0     # a final window shorter than this merges backwards

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "isolated_window_diag")

# The chunk-level result this diagnostic extends, quoted verbatim from
# results/isolated_chunk_diag/report.json (commit b925e39) so that the two
# granularities can be read side by side without opening a second file.
CHUNK_LEVEL_REFERENCE = {
    "source": "results/isolated_chunk_diag/report.json",
    "n_units": 1157,
    "n_pos": 731,
    "n_neg": 426,
    "arm_isolated": {
        "pooled_auc": 0.5325475424365619,
        "macro_auc_per_video": 0.5711021840296147,
        "n_videos_both_classes": 57,
        "frac_saturated_abs_gt_13": 0.6542783059636992,
        "spearman_z_vs_unit_tokens": 0.5039596797673178,
    },
    "arm_titled": {
        "pooled_auc": 0.533046890554453,
        "macro_auc_per_video": 0.5698257478230044,
        "n_videos_both_classes": 57,
        "frac_saturated_abs_gt_13": 0.5038893690579084,
        "spearman_z_vs_unit_tokens": 0.48677468799999185,
    },
}


# -------------------------------------------------------------------- windows

def build_windows(spans):
    """Greedy left-to-right grouping of chunk indices into windows.

    A window absorbs consecutive chunks until its hull span (last end minus
    first start) reaches WINDOW_CLOSE_SPAN, then closes. A final window whose
    hull span is under WINDOW_MIN_TAIL merges into its predecessor. Returns a
    list of lists of chunk indices covering every chunk exactly once.
    """
    windows, cur = [], []
    for k in range(len(spans)):
        cur.append(k)
        hull = spans[cur[-1]][1] - spans[cur[0]][0]
        if hull >= WINDOW_CLOSE_SPAN:
            windows.append(cur)
            cur = []
    if cur:
        windows.append(cur)
    if len(windows) >= 2:
        tail = windows[-1]
        if spans[tail[-1]][1] - spans[tail[0]][0] < WINDOW_MIN_TAIL:
            windows[-2] = windows[-2] + tail
            windows.pop()
    return windows


def window_gold_label(member_spans, labels, gspans):
    """Overlap-weighted majority gold label of one window.

    Overlap is accumulated over the member chunk spans (the voiced union), not
    over the window hull, so untranscribed gaps inside the window do not vote.
    Returns (label, detail) with label in {"offensive", "normal", None}.
    """
    t_off = t_norm = t_other = 0.0
    for cs, ce in member_spans:
        for lab, (gs, ge) in zip(labels, gspans):
            ov = min(float(ce), float(ge)) - max(float(cs), float(gs))
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


# -------------------------------------------------------------------- scoring

def run_forward(items, per_window_path, status):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    done = set()
    if os.path.exists(per_window_path):
        with open(per_window_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["video_id"], r["window_index"]))
    remaining = [it for it in items
                 if (it["video_id"], it["window_index"]) not in done]
    logging.info("resume: %d windows done, %d remaining"
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
    fout = open(per_window_path, "a")
    for i, it in enumerate(remaining):
        if i % 25 == 0:
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
    logging.info("forward done: %d windows, %.1f s (%.3f s/window, 2 calls each)"
                 % (len(remaining), time.time() - t0,
                    (time.time() - t0) / max(1, len(remaining))))


# ------------------------------------------------------------------- analysis

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

    ntok = np.array([r["window_tokens"] for r in rows], dtype=np.float64)
    rho, pval = stats.spearmanr(z, ntok)
    rho_s, p_s = stats.spearmanr(z, np.array([r["hull_span"] for r in rows]))

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
        "spearman_z_vs_window_tokens": {"rho": float(rho), "p": float(pval)},
        "spearman_z_vs_hull_span": {"rho": float(rho_s), "p": float(p_s)},
    }


CHUNK_SCORES_JSONL = os.path.join(
    PROJECT_ROOT, "results", "isolated_chunk_diag", "per_chunk.jsonl")


def aggregation_control(rows):
    """Window AUCs from the chunk diagnostic's margins, and from length alone.

    Returns None when the chunk diagnostic's per-chunk file is absent.
    """
    if not os.path.exists(CHUNK_SCORES_JSONL):
        return None
    zc = {}
    with open(CHUNK_SCORES_JSONL) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                zc[(r["video_id"], r["chunk_index"])] = r["z_isolated"]

    def auc_of(fn):
        pos, neg, skipped = [], [], 0
        for r in rows:
            v = fn(r)
            if v is None:
                skipped += 1
                continue
            (pos if r["gold"] == "offensive" else neg).append(v)
        return rank_auc(pos, neg), skipped

    def member(r, agg):
        zs = [zc[(r["video_id"], k)] for k in r["chunk_indices"]
              if (r["video_id"], k) in zc]
        return agg(zs) if zs else None

    auc_mean, skipped = auc_of(lambda r: member(r, lambda z: float(np.mean(z))))
    auc_max, _ = auc_of(lambda r: member(r, max))
    out = {
        "chunk_z_mean_pooled_auc": auc_mean,
        "chunk_z_max_pooled_auc": auc_max,
        "windows_without_scored_member_chunk": skipped,
        "window_tokens_alone_auc": auc_of(lambda r: r["window_tokens"])[0],
        "n_chunks_alone_auc": auc_of(lambda r: r["n_chunks"])[0],
        "hull_span_alone_auc": auc_of(lambda r: r["hull_span"])[0],
    }

    z = np.array([r["z_isolated"] for r in rows], dtype=np.float64)
    lab = np.array([r["gold"] for r in rows])
    rng = np.random.default_rng(0)
    p, n = z[lab == "offensive"], z[lab == "normal"]
    bs = [rank_auc(rng.choice(p, p.size), rng.choice(n, n.size))
          for _ in range(4000)]
    out["isolated_pooled_auc_boot95"] = [float(np.percentile(bs, 2.5)),
                                         float(np.percentile(bs, 97.5))]

    per_video = {}
    for r in rows:
        per_video.setdefault(r["video_id"], []).append(r)
    deg = []
    for _v, rs in per_video.items():
        pp = [x["z_isolated"] for x in rs if x["gold"] == "offensive"]
        nn = [x["z_isolated"] for x in rs if x["gold"] == "normal"]
        if pp and nn:
            deg.append(rank_auc(pp, nn))
    out["macro_auc_degenerate_fraction"] = (
        float(np.mean([a in (0.0, 1.0) for a in deg])) if deg else None)
    out["macro_auc_values_isolated"] = sorted(float(a) for a in deg)
    return out


def analyze(per_window_path, counts, geometry, sample, titles_present):
    rows = []
    with open(per_window_path) as f:
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
        "window_rule": {"close_span_s": WINDOW_CLOSE_SPAN,
                        "min_tail_span_s": WINDOW_MIN_TAIL,
                        "gold_overlap_basis": "voiced union of member chunk spans"},
        "title_availability": titles_present,
        "counts": counts,
        "geometry": geometry,
        "frozen_text_sha256": FROZEN_TEXT_SHA,
        "arm_isolated": arm_report(rows, "z_isolated"),
        "arm_titled": arm_report(rows, "z_titled"),
        "arm_agreement": {"sign_agreement": agree,
                          "spearman_rho": float(rho_ab),
                          "mean_shift_titled_minus_isolated":
                              float(np.mean(zb - za)),
                          "mean_abs_shift": float(np.mean(np.abs(zb - za)))},
        "aggregation_control": aggregation_control(rows),
        "chunk_level_reference": CHUNK_LEVEL_REFERENCE,
        "sample_video_ids": sample,
    }


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--build-only", action="store_true",
                    help="build windows and print geometry, run no model")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    status_path = os.path.join(args.out_dir, "STATUS")
    per_window_path = os.path.join(args.out_dir, "per_window.jsonl")
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

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    annotations = load_annotations(DATASET)
    titles_present = {
        "n_sample_videos": len(sample),
        "n_with_nonempty_title": sum(
            1 for v in sample
            if ((annotations.get(v) or {}).get("title") or "").strip()),
        "note": ("HateClipSeg supplies no titles; the titled arm is a null "
                 "prompt perturbation, not a context arm."),
    }

    items = []
    counts = {"cohort_videos": len(cohort), "sample_videos": len(sample),
              "chunks_total": 0, "windows_total": 0,
              "windows_offensive": 0, "windows_normal": 0,
              "windows_dropped_ambiguous": 0,
              "windows_dropped_no_gold_overlap": 0,
              "windows_empty_text": 0,
              "windows_hull_label_disagrees": 0}
    hull_spans, voiced_spans, n_chunks_per_win, win_tokens = [], [], [], []
    wins_per_video = []

    for vid in sample:
        rec = chunks[vid]
        spans = usable_spans(rec)
        labels, gspans = gold[vid]
        title = (annotations.get(vid) or {}).get("title", "") or ""
        counts["chunks_total"] += len(rec["chunks"])
        windows = build_windows(spans)
        wins_per_video.append(len(windows))
        for w, members in enumerate(windows):
            counts["windows_total"] += 1
            member_spans = [spans[k] for k in members]
            hull = member_spans[-1][1] - member_spans[0][0]
            voiced = sum(e - s for s, e in member_spans)
            texts = [clean_chunk_text(rec["chunks"][k].get("text"))
                     for k in members]
            text = " ".join(t for t in texts if t).strip()
            if not text:
                counts["windows_empty_text"] += 1
                continue
            lab, detail = window_gold_label(member_spans, labels, gspans)
            hull_lab, _ = window_gold_label(
                [(member_spans[0][0], member_spans[-1][1])], labels, gspans)
            if lab is None:
                if detail["t_total"] <= 0.0:
                    counts["windows_dropped_no_gold_overlap"] += 1
                else:
                    counts["windows_dropped_ambiguous"] += 1
                continue
            if hull_lab != lab:
                counts["windows_hull_label_disagrees"] += 1
            counts["windows_offensive" if lab == "offensive"
                   else "windows_normal"] += 1
            ntok = len(tok(text, add_special_tokens=False)["input_ids"])
            hull_spans.append(hull)
            voiced_spans.append(voiced)
            n_chunks_per_win.append(len(members))
            win_tokens.append(ntok)
            items.append({
                "video_id": vid, "window_index": w,
                "chunk_indices": members,
                "span": [member_spans[0][0], member_spans[-1][1]],
                "hull_span": hull, "voiced_span": voiced,
                "n_chunks": len(members),
                "text": text, "title": title, "gold": lab,
                "gold_hull": hull_lab,
                "gold_overlap": detail,
                "window_tokens": ntok,
            })
    counts["windows_labeled"] = len(items)

    geometry = {
        "windows_per_video": describe(wins_per_video),
        "hull_span_s": describe(hull_spans),
        "voiced_span_s": describe(voiced_spans),
        "chunks_per_window": describe(n_chunks_per_win),
        "window_tokens": describe(win_tokens),
    }
    logging.info("window counts: %s" % json.dumps(counts, indent=2))
    logging.info("window geometry: %s" % json.dumps(geometry, indent=2))
    logging.info("title availability: %s" % json.dumps(titles_present))

    if args.build_only:
        status("build-only")
        return

    if not args.analyze_only:
        status("forward")
        run_forward(items, per_window_path, status)

    status("analyze")
    report = analyze(per_window_path, counts, geometry, sample, titles_present)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    head = {k: report[k] for k in ("counts", "geometry", "arm_isolated",
                                   "arm_titled", "arm_agreement",
                                   "aggregation_control",
                                   "chunk_level_reference")}
    logging.info(json.dumps(head, indent=2))
    status("DONE")
    with open(os.path.join(args.out_dir, "DONE"), "w") as f:
        f.write(time.strftime("%F %T") + "\n")


if __name__ == "__main__":
    main()
