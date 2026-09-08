"""
DAPT precondition probe: transcript-span perplexity under the frozen judge.

Pre-registration: docs/duplex/PREREG_dapt_precondition.md (frozen 2026-08-10,
commit c5d36fd). The question is whether HateMM's unsaturated mid-band is
harder for the frozen judge to PREDICT than its saturated poles. If it is not,
the familiarity account of the mid-band congestion is false and the DAPT family
is not run.

Measurement. For every video in the held-out test split we rebuild the exact
frozen judge prompt used by the committed test run -- 16 frames at the frozen
pixel budget, the dataset title, the c2 override transcript, the per-corpus
platform rules, the `prag` reader block, the same system message -- and take one
teacher-forced forward pass. Nothing is generated. The per-token negative
log-likelihood is read off the transcript token span alone: the judge's ability
to predict the words of this corpus, with the video in context, and with no
contribution from the boilerplate that surrounds it.

Locating the span. The prompt text is tokenized once with character offsets
before the image placeholders are expanded; the transcript's character range is
known exactly from the frozen prompt template, so the token span is the set of
tokens overlapping that range. Image expansion happens entirely before the
transcript, so the span shifts by a single constant -- the difference between
the expanded and unexpanded sequence lengths -- and the shift is verified by
comparing token ids at the shifted positions.

Two stages:

  python scripts/duplex/dapt_precondition.py score    # GPU, ~615 forwards
  python scripts/duplex/dapt_precondition.py analyze  # CPU, writes report.json

Outputs under results/dapt_precondition/:
  <slug>_nll.jsonl   one record per video
  report.json        the pre-registered statistics
"""

import argparse
import glob as globmod
import json
import logging
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "duplex"))

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "dapt_precondition")

# Corpora, their committed test-run directories, and the z scores the bands come
# from. Both are the 8B judge arm of the held-out test measurement.
CORPORA = [
    ("hatemm", "HateMM"),
    ("implihatevid", "ImpliHateVid"),
]

# Frozen conventions. The band cut is the saturation pilot's; the exclusion
# floor and the judge configuration are the pre-registration's.
Z_POLE = 13.0
MIN_TRANSCRIPT_TOKENS = 20
NUM_FRAMES = 16
MAX_PIXELS = 100352
MIN_PIXELS = 65536
MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# Positions per lm_head chunk when reading log-probabilities off the logits.
LOGPROB_CHUNK = 512


def testrun_dir(slug):
    return os.path.join(PROJECT_ROOT, "results", "testruns", slug)


def nll_path(slug):
    return os.path.join(OUT_DIR, f"{slug}_nll.jsonl")


def load_z(slug):
    """video_id -> z from the committed 8B test scores."""
    z = {}
    path = os.path.join(testrun_dir(slug), "judge_8b", "scores.jsonl")
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                z[r["video_id"]] = float(r["z"])
    return z


def load_rows(slug):
    rows = []
    path = nll_path(slug)
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------- score


def resolve_frames(vid, dataset, num_frames, dataset_roots):
    frame_dir = os.path.join(dataset_roots[dataset], "frames_16", vid)
    if not os.path.isdir(frame_dir):
        return []
    jpgs = sorted(globmod.glob(os.path.join(frame_dir, "*.jpg")))
    if not jpgs:
        return []
    if len(jpgs) > num_frames:
        indices = np.linspace(0, len(jpgs) - 1, num_frames, dtype=int)
        jpgs = [jpgs[i] for i in indices]
    return jpgs


def transcript_char_span(prompt_template, prompt_text, title, transcript):
    """Character range of the transcript inside the rendered prompt text.

    Taken from the frozen template rather than by searching for the transcript,
    which may itself contain the surrounding literal text.
    """
    head_tpl = prompt_template.split("{transcript}")[0]
    head = head_tpl.format(title=title)
    start = len(head)
    end = start + len(transcript)
    if prompt_text[start:end] != transcript:
        raise SystemExit("transcript char span does not reproduce the transcript")
    return start, end


def score(args):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import torch
    from PIL import Image

    from score_duplex_probe import (
        BILIBILI_RULES,
        DUPLEX_PROMPT,
        READER_BLOCKS,
        SYSTEM_MESSAGE,
        YOUTUBE_RULES,
    )
    from data_utils import DATASET_ROOTS, load_annotations, load_clean_split_ids

    os.makedirs(OUT_DIR, exist_ok=True)
    size_kwarg = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}

    plans = []
    for slug, dataset in CORPORA:
        ids = load_clean_split_ids(dataset, "test")
        done = {r["video_id"] for r in load_rows(slug)}
        remaining = [v for v in ids if v not in done]
        logging.info(f"{slug}: {len(ids)} test ids, {len(done)} done, {len(remaining)} remaining")
        if remaining:
            plans.append((slug, dataset, remaining))
    if not plans:
        logging.info("Nothing to do.")
        return

    processor = AutoProcessor.from_pretrained(MODEL)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    logging.info(f"Loaded {MODEL}")

    t0 = time.time()
    n_total = 0
    for slug, dataset, remaining in plans:
        annotations = load_annotations(dataset)
        with open(os.path.join(testrun_dir(slug), "c2_overrides.json")) as f:
            overrides = json.load(f)
        rules_text = BILIBILI_RULES if dataset == "MHClip_ZH" else YOUTUBE_RULES
        out_path = nll_path(slug)
        logging.info(f"=== {slug}: {len(remaining)} videos -> {out_path}")

        for i, vid in enumerate(remaining):
            ann = annotations.get(vid)
            if ann is None:
                logging.warning(f"  {vid}: not in annotations, skipping")
                continue
            frame_paths = resolve_frames(vid, dataset, NUM_FRAMES, DATASET_ROOTS)
            if not frame_paths:
                # The frozen judge run skipped these for the same reason, so
                # they carry no z and are outside the analysis either way.
                logging.warning(f"  {vid}: no frames, skipping")
                continue

            title = ann.get("title", "") or ""
            if vid in overrides:
                transcript = overrides[vid] or ""
                source = "override"
            else:
                transcript = ann.get("transcript", "") or ""
                source = "dataset"

            prompt_text = DUPLEX_PROMPT.format(
                title=title, transcript=transcript, rules=rules_text,
                reader_block=READER_BLOCKS["prag"])
            content = [{"type": "image"} for _ in frame_paths]
            content.append({"type": "text", "text": prompt_text})
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": content},
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            # Character range of the transcript inside the full templated text.
            base = text.rindex(prompt_text)
            c0, c1 = transcript_char_span(DUPLEX_PROMPT, prompt_text, title, transcript)
            c0 += base
            c1 += base

            # Token span before image expansion, then shifted by the expansion.
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            unexp_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            # Every token whose character range overlaps the transcript. The
            # first one carries the space after "Transcript:" and the last one
            # sometimes carries the newline that follows the transcript; both
            # are one token, and the rule is the same for every video.
            if transcript:
                span = [k for k, (a, b) in enumerate(offsets) if b > c0 and a < c1]
            else:
                span = []

            rec = {
                "video_id": vid,
                "n_transcript_chars": len(transcript),
                "transcript_source": source,
            }

            if not span:
                if len(transcript) != 0:
                    raise SystemExit(
                        f"{vid}: empty transcript token span for a {len(transcript)}-char transcript")
                # Empty transcripts carry no span to score; they fall below the
                # pre-registered 20-token floor and are excluded downstream.
                rec.update({"n_transcript_tokens": 0, "mean_nll": None,
                            "sum_nll": None, "n_tokens_total": None})
                with open(out_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                n_total += 1
                continue

            s_unexp, e_unexp = span[0], span[-1] + 1

            images = [Image.open(p).convert("RGB") for p in frame_paths]
            inputs = processor(text=[text], images=images, return_tensors="pt",
                               size=size_kwarg)
            for im in images:
                im.close()
            del images

            grid = inputs["image_grid_thw"]
            ip = processor.image_processor
            per_frame_pixels = [int(h * w) * ip.patch_size * ip.patch_size
                                for _, h, w in grid.tolist()]
            if max(per_frame_pixels) > MAX_PIXELS:
                raise SystemExit(f"{vid}: pixel cap not honored: {max(per_frame_pixels)}")

            inputs = inputs.to(model.device)
            ids = inputs["input_ids"]
            L = int(ids.shape[1])
            delta = L - len(unexp_ids)
            if delta < 0:
                raise SystemExit(f"{vid}: negative image-expansion delta {delta}")
            s, e = s_unexp + delta, e_unexp + delta
            if e > L or s < 1:
                raise SystemExit(f"{vid}: span [{s},{e}) outside sequence of length {L}")
            shifted = ids[0, s:e].tolist()
            if shifted != unexp_ids[s_unexp:e_unexp]:
                raise SystemExit(f"{vid}: token ids at the shifted span do not match")
            n_span = e - s
            if n_span <= 0:
                raise SystemExit(f"{vid}: non-positive span length {n_span}")

            # Teacher-forced forward. Keep only the logits from the position
            # before the span onward; the transcript sits well before the end
            # of the prompt, so this is a fraction of the full logit tensor.
            keep = L - (s - 1)
            with torch.no_grad():
                out = model(**inputs, use_cache=False, output_hidden_states=False,
                            logits_to_keep=keep)
                logits = out.logits[0]  # [keep, V]; row j predicts position (s-1)+j+1
                targets = ids[0, s:e]
                nlls = torch.empty(n_span, dtype=torch.float32, device=logits.device)
                for j0 in range(0, n_span, LOGPROB_CHUNK):
                    j1 = min(j0 + LOGPROB_CHUNK, n_span)
                    chunk = logits[j0:j1].float()
                    lse = torch.logsumexp(chunk, dim=-1)
                    picked = chunk.gather(1, targets[j0:j1].unsqueeze(1)).squeeze(1)
                    nlls[j0:j1] = lse - picked
                    del chunk, lse, picked
                sum_nll = float(nlls.sum())
                mean_nll = float(nlls.mean())
            del out, logits, nlls, inputs

            if not np.isfinite(mean_nll):
                raise SystemExit(f"{vid}: non-finite mean NLL")

            rec.update({
                "n_transcript_tokens": n_span,
                "mean_nll": mean_nll,
                "sum_nll": sum_nll,
                "n_tokens_total": L,
                "span_start": s,
                "span_end": e,
                "exact_char_start": bool(offsets[s_unexp][0] == c0),
                "exact_char_end": bool(offsets[e_unexp - 1][1] == c1),
            })
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

            n_total += 1
            if n_total % 25 == 0 or i == 0:
                el = time.time() - t0
                logging.info(
                    f"  [{n_total}] {slug}/{vid} span={n_span} tok mean_nll={mean_nll:.4f} "
                    f"L={L} {el / max(n_total, 1):.2f}s/video "
                    f"peak_vram={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    logging.info(f"Scored {n_total} videos in {time.time() - t0:.1f}s")


# ------------------------------------------------------------------- analyze


def auc_and_p(pos, neg):
    """AUC of `pos` over `neg` plus the two-sided Mann-Whitney p.

    AUC = U / (n_pos n_neg) with U the Mann-Whitney statistic of the positive
    sample, which is the tie-corrected P(pos > neg) + 0.5 P(pos = neg).
    """
    from scipy import stats
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    res = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    auc = float(res.statistic) / (len(pos) * len(neg))
    greater = stats.mannwhitneyu(pos, neg, alternative="greater")
    return auc, float(res.pvalue), float(greater.pvalue)


def describe(x):
    x = np.asarray(x, dtype=float)
    return {
        "n": int(x.size),
        "median": float(np.median(x)) if x.size else None,
        "mean": float(np.mean(x)) if x.size else None,
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else None,
        "min": float(np.min(x)) if x.size else None,
        "max": float(np.max(x)) if x.size else None,
    }


def analyze(args):
    from scipy import stats

    report = {
        "prereg": "docs/duplex/PREREG_dapt_precondition.md",
        "model": MODEL,
        "z_pole_cut": Z_POLE,
        "min_transcript_tokens": MIN_TRANSCRIPT_TOKENS,
        "corpora": {},
    }

    per_corpus = {}
    for slug, dataset in CORPORA:
        z = load_z(slug)
        rows = [r for r in load_rows(slug) if r["video_id"] in z]
        for r in rows:
            r["z"] = z[r["video_id"]]
        n_all = len(rows)
        kept = [r for r in rows
                if r["n_transcript_tokens"] >= MIN_TRANSCRIPT_TOKENS
                and r["mean_nll"] is not None]
        kept_ids = {id(r) for r in kept}
        excluded = [r["video_id"] for r in rows if id(r) not in kept_ids]
        nll = np.array([r["mean_nll"] for r in kept])
        absz = np.array([abs(r["z"]) for r in kept])
        ntok = np.array([r["n_transcript_tokens"] for r in kept])
        is_mid = absz < Z_POLE

        auc, p_two, p_greater = auc_and_p(nll[is_mid], nll[~is_mid])
        rho_z, p_rho_z = stats.spearmanr(nll, absz)
        rho_len, p_rho_len = stats.spearmanr(nll, ntok)

        entry = {
            "dataset": dataset,
            "n_scored": n_all,
            "n_excluded_short": n_all - len(kept),
            "excluded_ids": sorted(excluded),
            "n_analyzed": len(kept),
            "median_nll": float(np.median(nll)),
            "nll_all": describe(nll),
            "band_mid": describe(nll[is_mid]),
            "band_poles": describe(nll[~is_mid]),
            "auc_mid_over_poles": auc,
            "mannwhitney_p_two_sided": p_two,
            "mannwhitney_p_one_sided_mid_greater": p_greater,
            "spearman_nll_absz": {"rho": float(rho_z), "p": float(p_rho_z)},
            "spearman_nll_ntokens": {"rho": float(rho_len), "p": float(p_rho_len)},
        }
        report["corpora"][slug] = entry
        per_corpus[slug] = {"kept": kept, "nll": nll}

    # The 11 code-Q videos of the false-positive audit, placed in HateMM's NLL
    # distribution. Q = the judge's positive is defensible reportage or
    # counter-speech rather than a clean false positive.
    audit_dir = os.path.join(PROJECT_ROOT, "results", "hatemm_fp_audit")
    with open(os.path.join(audit_dir, "alias_map.json")) as f:
        alias_map = json.load(f)
    q_aliases = []
    with open(os.path.join(audit_dir, "coding.tsv")) as f:
        header = f.readline()
        del header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1] == "Q":
                q_aliases.append(parts[0])
    hm = per_corpus["hatemm"]
    by_id = {r["video_id"]: r for r in hm["kept"]}
    ref = np.sort(hm["nll"])
    code_q = []
    for alias in q_aliases:
        vid = alias_map[alias]
        r = by_id.get(vid)
        if r is None:
            code_q.append({"alias": alias, "video_id": vid, "status": "excluded_or_missing"})
            continue
        pct = 100.0 * float(np.searchsorted(ref, r["mean_nll"], side="right")) / ref.size
        code_q.append({
            "alias": alias, "video_id": vid, "mean_nll": r["mean_nll"],
            "n_transcript_tokens": r["n_transcript_tokens"], "z": r["z"],
            "nll_percentile_within_hatemm": pct,
        })
    present = [c["nll_percentile_within_hatemm"] for c in code_q if "nll_percentile_within_hatemm" in c]
    report["code_q"] = {
        "n_aliases": len(q_aliases),
        "n_placed": len(present),
        "median_percentile": float(np.median(present)) if present else None,
        "videos": code_q,
    }

    hm_e = report["corpora"]["hatemm"]
    ih_e = report["corpora"]["implihatevid"]
    link2_pass = hm_e["auc_mid_over_poles"] >= 0.60 and hm_e["mannwhitney_p_two_sided"] < 0.05
    guard = ih_e["auc_mid_over_poles"] >= hm_e["auc_mid_over_poles"] - 0.03
    report["decision"] = {
        "link1_median_nll_hatemm": hm_e["median_nll"],
        "link1_median_nll_implihatevid": ih_e["median_nll"],
        "link2_pass": bool(link2_pass),
        "link2_bar": "AUC >= 0.60 and Mann-Whitney p < 0.05",
        "implihatevid_guard_triggered": bool(guard),
        "guard_rule": "control AUC within 0.03 of HateMM's or higher",
        "dapt_proceeds": bool(link2_pass),
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    def line(*a):
        print(*a)

    line("=" * 72)
    line("DAPT precondition probe -- transcript-span perplexity")
    line("=" * 72)
    for slug, _ in CORPORA:
        e = report["corpora"][slug]
        line(f"\n{e['dataset']} (test, {e['n_scored']} scored)")
        line(f"  excluded (<{MIN_TRANSCRIPT_TOKENS} transcript tokens): "
             f"{e['n_excluded_short']}   analyzed: {e['n_analyzed']}")
        line(f"  median NLL {e['median_nll']:.4f}   "
             f"mean {e['nll_all']['mean']:.4f} sd {e['nll_all']['sd']:.4f}")
        for band in ("band_mid", "band_poles"):
            b = e[band]
            name = "mid  (|z|<13) " if band == "band_mid" else "poles(|z|>=13)"
            line(f"  {name} n={b['n']:4d} median {b['median']:.4f} "
                 f"mean {b['mean']:.4f} sd {b['sd']:.4f}")
        line(f"  AUC(mid over poles) {e['auc_mid_over_poles']:.4f}   "
             f"MW p(two-sided) {e['mannwhitney_p_two_sided']:.4g}   "
             f"p(one-sided mid greater) {e['mannwhitney_p_one_sided_mid_greater']:.4g}")
        line(f"  Spearman(NLL, |z|)      rho {e['spearman_nll_absz']['rho']:+.4f} "
             f"p {e['spearman_nll_absz']['p']:.4g}")
        line(f"  Spearman(NLL, n_tokens) rho {e['spearman_nll_ntokens']['rho']:+.4f} "
             f"p {e['spearman_nll_ntokens']['p']:.4g}")

    q = report["code_q"]
    med = ("%.1f" % q["median_percentile"]) if q["median_percentile"] is not None else "n/a"
    line(f"\ncode-Q videos ({q['n_placed']}/{q['n_aliases']} placed), "
         f"median NLL percentile within HateMM {med}")
    for c in q["videos"]:
        if "nll_percentile_within_hatemm" in c:
            line(f"  {c['alias']} {c['video_id']:>22}  NLL {c['mean_nll']:.4f}  "
                 f"pct {c['nll_percentile_within_hatemm']:5.1f}  "
                 f"z {c['z']:+.2f}  tok {c['n_transcript_tokens']}")
        else:
            line(f"  {c['alias']} {c['video_id']:>22}  {c['status']}")

    d = report["decision"]
    line("\n" + "-" * 72)
    line(f"Link 1 median NLL: HateMM {d['link1_median_nll_hatemm']:.4f} vs "
         f"ImpliHateVid {d['link1_median_nll_implihatevid']:.4f}")
    line(f"Link 2 ({d['link2_bar']}): {'PASS' if d['link2_pass'] else 'FAIL'}")
    line(f"ImpliHateVid guard ({d['guard_rule']}): "
         f"{'TRIGGERED' if d['implihatevid_guard_triggered'] else 'not triggered'}")
    line(f"DAPT proceeds: {d['dapt_proceeds']}")
    line(f"\nwrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["score", "analyze"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    if args.stage == "score":
        score(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
