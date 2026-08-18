"""Masked packed locator (Arm M) on MultiHateClip EN / ZH, frame-level scored.

What this is. `scripts/duplex/masked_parallel_isolation_pilot.py` (commit
aa403cb) established, on HateMM, that packing the shared rules prefix and all
per-chunk branches into ONE sequence with a block-diagonal attention mask and a
branch-local position restart reproduces the sequential isolated-chunk call
exactly. That mechanism is not re-litigated here. This script carries the same
Arm M forward to MultiHateClip EN and ZH so the method has numbers on the two
corpora the reproduction study's baselines are scored on.

Differences from the HateMM pilot, and only these:

  * Corpus. Cohort = the videos in the frozen gold arrays
    (results/reproduction/gt/mhclip_{en,zh}_test.npz) intersected with the
    videos carrying timestamped Whisper chunks. Chunks come from
    results/interleaved_timeline/<corpus>/timestamped_chunks.jsonl plus
    results/reproduction/asr/<corpus>_test_new/timestamped_chunks.jsonl, the
    late-ASR records for the videos the first pass missed.
  * Rules block. The frozen judge uses BILIBILI_RULES on MHClip_ZH and
    YOUTUBE_RULES everywhere else (src/duplex/score_duplex_probe.py:161-162,
    src/our_method/score_holistic_2b.py:467). That convention is replicated:
    everything else in the prompt -- the lead-in sentence, the system message,
    the question, the template layout -- is byte-identical to the HateMM
    pilot's, and the EN prompt template is asserted at runtime to be identical
    to `isolated_chunk_diag.user_text`.
  * Arms. Arm M only; no Arm C, since the causal-arm collapse was measured on
    HateMM. Fidelity to the sequential computation is checked in two stages.
    The default is a spot check -- three videos re-scored with genuine isolated
    calls -- against a hard bar of Spearman >= 0.99, below which the run stops.
    On MultiHateClip that bar is not measurable: z is tie-dominated (17 distinct
    values across the 75 EN spot chunks, largest tie group 24), so Spearman
    turns sub-quantum bf16 noise into rank swaps and reads 0.980 even though
    Pearson is 0.9987 and every discordant pair is separated by at most 0.25 in
    the reference, one quantum of the score grid. `--sequential-reference`
    therefore scores EVERY chunk both ways and reports both endpoints, which
    measures the thing the bar was a proxy for; it makes the spot bar advisory.

    Where the residual comes from, measured rather than assumed. At N=1 the
    packed pass is BIT-IDENTICAL to the isolated call, so the seam, the block
    mask and the position restart are exact. The residual appears only at N>1
    and is bf16 non-associativity: attention over the longer packed sequence
    tiles its reduction differently. The model is deterministic (the same call
    repeated returns the identical value), and HF's own dispatch contributes a
    difference of the same size -- an all-ones 2-D attention_mask takes a
    different SDPA path than an explicit 4-D mask -- which is why
    `score_sequential` takes the mask form as an argument.
  * Endpoint. Frame-level, against the frozen gold arrays, through
    `scripts/duplex/frame_eval_common.py` -- the same evaluator every baseline
    in the reproduction study goes through. Chunk z is spread over its
    [start, end); frames no scored chunk covers take (corpus-min chunk z) - 1,
    the floor convention frozen in
    docs/duplex/PREREG_frame_level_evaluation_hatemm.md, applied per corpus.
    A gold video with no usable chunk at all is scored all-floor and counted.

Output: results/reproduction/ours/mhclip_{en,zh}/{per_chunk.jsonl,
frame_eval.json, prompt_fingerprints.json, STATUS, DONE}. No transcript text
is written to any output.
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
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts", "reproduction_baselines"))

from score_duplex_probe import (  # noqa: E402
    SYSTEM_MESSAGE,
    build_binary_token_ids,
)
from score_holistic_2b import BILIBILI_RULES, YOUTUBE_RULES  # noqa: E402
from sentinel_localization_pilot import clean_chunk_text, usable_spans  # noqa: E402
from isolated_chunk_diag import (  # noqa: E402
    QUESTION,
    RULES_LEAD_IN,
    user_text as hatemm_user_text,
)
from masked_parallel_isolation_pilot import (  # noqa: E402
    SEAM_MARKER,
    CHUNK_PLACEHOLDER,
    block_mask,
    branch_position_ids,
    causal_mask,
    sequential_position_ids,
)
from hate_common import data as hdata  # noqa: E402
import frame_eval_common as fec  # noqa: E402
from eval_baseline_scores import evaluate_scores, format_report  # noqa: E402

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
FIDELITY_SPEARMAN_BAR = 0.99
SPOT_VIDEOS = 3
FPS = 1.0

CORPUS_RULES = {"mhclip_en": YOUTUBE_RULES, "mhclip_zh": BILIBILI_RULES}
CHUNK_SOURCES = (
    os.path.join(PROJECT_ROOT, "results", "interleaved_timeline",
                 "%(corpus)s", "timestamped_chunks.jsonl"),
    os.path.join(PROJECT_ROOT, "results", "reproduction", "asr",
                 "%(corpus)s_test_new", "timestamped_chunks.jsonl"),
)
OUT_ROOT = os.path.join(PROJECT_ROOT, "results", "reproduction", "ours")


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sub(a, b):
    return None if (a is None or b is None) else float(a - b)


# ------------------------------------------------------------------- prompts

def make_user_text(rules_block):
    """The HateMM pilot's template with the rules block swapped per corpus."""
    def user_text(chunk_text):
        return "%s\n\n%sTranscript excerpt from a video:\n%s\n\n%s" % (
            rules_block, "", chunk_text, QUESTION)
    return user_text


def corpus_prompt(corpus):
    rules_block = RULES_LEAD_IN + "\n" + CORPUS_RULES[corpus]
    ut = make_user_text(rules_block)
    if corpus == "mhclip_en":
        # The EN corpus takes YOUTUBE_RULES, so the template must be the
        # HateMM pilot's byte for byte. Asserted, not assumed.
        for probe in ("<CHUNK>", "abc", ""):
            if ut(probe) != hatemm_user_text(probe):
                raise AssertionError(
                    "mhclip_en template drifted from the HateMM pilot's")
    return rules_block, ut


# ------------------------------------------------------------------- scoring

class Judge:
    """The frozen judge with the packed masked readout, one corpus at a time."""

    def __init__(self, corpus):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.corpus = corpus
        self.rules_block, self.user_text = corpus_prompt(corpus)
        self.processor = AutoProcessor.from_pretrained(MODEL)
        self.tokenizer = self.processor.tokenizer
        ids = build_binary_token_ids(self.tokenizer)
        self.yes_ids, self.no_ids = sorted(ids["Yes"]), sorted(ids["No"])
        logging.info("Yes ids %s No ids %s" % (self.yes_ids, self.no_ids))
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = self.model.device
        self.dtype = torch.bfloat16
        self.yes_t = torch.tensor(self.yes_ids, device=self.device)
        self.no_t = torch.tensor(self.no_ids, device=self.device)
        self.prefix_text, self.suffix_text = self._split()
        self.prefix_ids = self.encode(self.prefix_text)

    def _full_prompt(self, chunk_text):
        msgs = [{"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user",
                 "content": [{"type": "text",
                              "text": self.user_text(chunk_text)}]}]
        return self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

    def _split(self):
        """Same seam as the HateMM pilot: the transcript-excerpt line."""
        template = self._full_prompt(CHUNK_PLACEHOLDER)
        cut = template.index(SEAM_MARKER) + len(SEAM_MARKER)
        prefix_text = template[:cut]
        suffix_text = template[cut + len(CHUNK_PLACEHOLDER):]
        assert prefix_text.endswith("\n")
        assert self._full_prompt("ABC") == prefix_text + "ABC" + suffix_text
        return prefix_text, suffix_text

    def fingerprints(self):
        return {
            "corpus": self.corpus,
            "rules_source": ("BILIBILI_RULES" if self.corpus == "mhclip_zh"
                             else "YOUTUBE_RULES"),
            "rules_block": sha(self.rules_block),
            "question": sha(QUESTION),
            "system_message": sha(SYSTEM_MESSAGE),
            "user_text_template_notitle": sha(self.user_text(CHUNK_PLACEHOLDER)),
            "chat_template_full_prompt": sha(self._full_prompt(CHUNK_PLACEHOLDER)),
            "prefix": sha(self.prefix_text),
            "suffix": sha(self.suffix_text),
            "prefix_tokens": len(self.prefix_ids),
        }

    def encode(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def margin(self, logits_row):
        lg = logits_row.float()
        return float(torch.logsumexp(lg[self.yes_t], 0)
                     - torch.logsumexp(lg[self.no_t], 0))

    def score_sequential(self, chunk_text, mask_form="4d"):
        """One genuine isolated call, the thing Arm M must reproduce.

        ``mask_form`` selects which attention path the reference runs on, and
        it has to be stated rather than defaulted, because the two paths do not
        return the same bf16 value:

          "4d"  an explicit additive causal mask and explicit position ids --
                the same arguments packed_forward passes, so the comparison
                isolates the packing mechanism.
          "2d"  the tokenizer's all-ones 2-D attention_mask, which is what the
                HateMM diagnostic and pilot pass.

        Measured on the MHC spot videos (scratchpad kernel_test.py): the 4d
        path, the no-mask path and the packed pass agree bit for bit, while the
        2d path differs from all three by up to 0.5 logit on EN and 0.25 on ZH.
        The model itself is deterministic -- the same call repeated returns the
        identical value -- so that gap is HF's kernel dispatch, not the method.
        Comparing the packed arm against the 2d path therefore measures an
        attention-kernel difference rather than the mechanism, which is what
        the first spot check on this corpus was doing.
        """
        prompt = self._full_prompt(chunk_text)
        enc = self.tokenizer(prompt, return_tensors="pt",
                             add_special_tokens=False)
        ids = enc["input_ids"].to(self.device)
        n = int(ids.shape[1])
        with torch.no_grad():
            if mask_form == "4d":
                out = self.model(
                    input_ids=ids,
                    attention_mask=causal_mask(n, self.dtype, self.device),
                    position_ids=sequential_position_ids(n, self.device),
                    use_cache=False, logits_to_keep=1)
            elif mask_form == "2d":
                out = self.model(
                    input_ids=ids,
                    attention_mask=enc["attention_mask"].to(self.device),
                    use_cache=False, logits_to_keep=1)
            else:
                raise ValueError(mask_form)
        return self.margin(out.logits[0, -1, :]), n

    def branches(self, chunk_texts):
        """Branch ids, with the mandatory runtime prompt-identity assertion."""
        out = []
        for text in chunk_texts:
            branch = self.encode(text + self.suffix_text)
            full = self.encode(self._full_prompt(text))
            if self.prefix_ids + branch != full:
                raise AssertionError(
                    "seam tokenization mismatch: prefix+branch != sequential "
                    "prompt ids (lens %d+%d vs %d)"
                    % (len(self.prefix_ids), len(branch), len(full)))
            out.append(branch)
        return out

    def packed_forward(self, branches):
        """One masked forward over [prefix, branch_1..branch_N]."""
        p = len(self.prefix_ids)
        lens = [len(b) for b in branches]
        ids = list(self.prefix_ids)
        ends = []
        for b in branches:
            ids.extend(b)
            ends.append(len(ids) - 1)
        mask = block_mask(p, lens, self.dtype, self.device)
        pos = branch_position_ids(p, lens, self.device)
        inp = torch.tensor([ids], device=self.device)
        keep = torch.tensor(ends, device=self.device)
        with torch.no_grad():
            out = self.model(input_ids=inp, attention_mask=mask,
                             position_ids=pos, use_cache=False,
                             logits_to_keep=keep)
        logits = out.logits[0]
        return [self.margin(logits[i, :]) for i in range(len(branches))], len(ids)


def verify_mask_plumbing(judge, chunk_texts):
    """A fully causal 4D mask must reproduce the default no-mask logits."""
    branches = judge.branches(chunk_texts)
    ids = list(judge.prefix_ids)
    for b in branches:
        ids.extend(b)
    inp = torch.tensor([ids], device=judge.device)
    with torch.no_grad():
        base = judge.model(input_ids=inp, use_cache=False,
                           logits_to_keep=8).logits[0].float()
        m = causal_mask(len(ids), judge.dtype, judge.device)
        pos = sequential_position_ids(len(ids), judge.device)
        got = judge.model(input_ids=inp, attention_mask=m, position_ids=pos,
                          use_cache=False, logits_to_keep=8).logits[0].float()
    d = float((base - got).abs().max())
    logging.info("4D-causal vs no-mask max |Delta logit| = %.6f" % d)
    return d


# -------------------------------------------------------------------- cohort

def load_chunk_records(corpus, counts):
    recs = {}
    for pattern in CHUNK_SOURCES:
        path = pattern % {"corpus": corpus}
        n = 0
        if not os.path.exists(path):
            raise SystemExit("missing chunk source %s" % path)
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                vid = r["video_id"]
                if vid in recs:
                    raise SystemExit("duplicate chunk record for %s" % vid)
                recs[vid] = r
                n += 1
        counts.setdefault("chunk_source_records", {})[os.path.relpath(
            path, PROJECT_ROOT)] = n
    return recs


def build_items(corpus, gt, chunk_recs, counts):
    """Scored chunks, plus the per-video span table the frame map needs."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    items = []
    spans_by_video = {}
    for vid in sorted(gt):
        rec = chunk_recs.get(vid)
        if rec is None or not rec.get("chunks"):
            counts["videos_no_chunks"] += 1
            continue
        spans = usable_spans(rec)
        if spans is None:
            counts["videos_unusable_spans"] += 1
            continue
        spans_by_video[vid] = spans
        kept = 0
        for k, c in enumerate(rec["chunks"]):
            counts["chunks_total"] += 1
            text = clean_chunk_text(c.get("text"))
            if not text:
                counts["chunks_empty_text"] += 1
                continue
            items.append({
                "video_id": vid,
                "chunk_index": k,
                "span": [float(spans[k][0]), float(spans[k][1])],
                "text": text,
                "chunk_tokens": len(tok(text,
                                        add_special_tokens=False)["input_ids"]),
            })
            kept += 1
        if kept == 0:
            counts["videos_all_chunks_empty"] += 1
    counts["chunks_scored"] = len(items)
    return items, spans_by_video


# ------------------------------------------------------------- frame mapping

def frame_scores(gt, rows, spans_by_video, counts, key="z_masked"):
    """Chunk z spread over [start, end); uncovered frames take the floor."""
    by_video = {}
    for r in rows:
        by_video.setdefault(r["video_id"], {})[r["chunk_index"]] = r[key]
    floor = min(r[key] for r in rows) - 1.0
    if key == "z_masked":
        counts["uncovered_floor"] = float(floor)

    scores = {}
    for vid in sorted(gt):
        n = int(len(gt[vid]))
        t = np.arange(n, dtype=float) / FPS
        s = np.full(n, floor, dtype=float)
        scored = by_video.get(vid, {})
        spans = spans_by_video.get(vid) or []
        # Whisper segments do not overlap; first covering scored chunk wins,
        # matching frame_eval_common.spans_to_frame_scores.
        filled = np.zeros(n, dtype=bool)
        for k in sorted(scored):
            if k >= len(spans):
                continue
            a, b = float(spans[k][0]), float(spans[k][1])
            if not (b > a):
                continue
            hit = (t >= a) & (t < b) & (~filled)
            s[hit] = float(scored[k])
            filled |= hit
        # Counted once, on the primary column: this runs a second time for the
        # sequential reference over the identical spans, so accumulating both
        # passes would double every frame count.
        if key == "z_masked":
            if not scored:
                counts["videos_scored_all_floor"] += 1
                counts["videos_all_floor_ids"].append(vid)
            counts["frames_total"] += n
            counts["frames_covered"] += int(filled.sum())
            counts["frames_uncovered"] += int(n - filled.sum())
        scores[vid] = s
    return scores


# ---------------------------------------------------------------------- main

def run_corpus(corpus, args):
    out_dir = os.path.join(args.out_root, corpus)
    os.makedirs(out_dir, exist_ok=True)
    per_chunk_path = os.path.join(out_dir, "per_chunk.jsonl")
    status_path = os.path.join(out_dir, "STATUS")

    def status(s):
        with open(status_path, "w") as f:
            f.write("%s  %s\n" % (time.strftime("%F %T"), s))

    status("cohort")
    counts = {k: 0 for k in (
        "videos_no_chunks", "videos_unusable_spans", "videos_all_chunks_empty",
        "chunks_total", "chunks_empty_text", "chunks_scored",
        "videos_scored_all_floor", "frames_total", "frames_covered",
        "frames_uncovered")}
    counts["videos_all_floor_ids"] = []

    gt = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    hate_ids = {v for v in gt if labels.get(v) == 1}
    chunk_recs = load_chunk_records(corpus, counts)
    counts["gold_videos"] = len(gt)
    counts["gold_videos_hate"] = len(hate_ids)
    counts["gold_videos_without_chunk_record"] = len(set(gt) - set(chunk_recs))

    items, spans_by_video = build_items(corpus, gt, chunk_recs, counts)
    by_video = {}
    for it in items:
        by_video.setdefault(it["video_id"], []).append(it)
    for vid in by_video:
        by_video[vid].sort(key=lambda r: r["chunk_index"])
    video_ids = sorted(by_video)
    if args.limit_videos:
        video_ids = video_ids[:args.limit_videos]
    logging.info("%s: %d gold videos (%d hateful), %d with scorable chunks, "
                 "%d chunks" % (corpus, len(gt), len(hate_ids),
                                len(video_ids), len(items)))

    status("load model")
    judge = Judge(corpus)
    fp = judge.fingerprints()
    with open(os.path.join(out_dir, "prompt_fingerprints.json"), "w") as f:
        json.dump(fp, f, indent=2)
    logging.info("prompt fingerprints: %s" % json.dumps(fp, indent=2))

    # ---- spot fidelity: the videos with the most chunks, so the block mask
    # is actually exercised (one-chunk videos make masked and causal coincide).
    status("spot fidelity")
    spot_ids = sorted(video_ids, key=lambda v: (-len(by_video[v]), v))[:SPOT_VIDEOS]
    logging.info("spot videos %s (chunks %s)"
                 % (spot_ids, [len(by_video[v]) for v in spot_ids]))
    plumbing = verify_mask_plumbing(
        judge, [it["text"] for it in by_video[spot_ids[0]]][:4])
    spot_masked, spot_seq, spot_seq2d = [], [], []
    for vid in spot_ids:
        texts = [it["text"] for it in by_video[vid]]
        zs, _ = judge.packed_forward(judge.branches(texts))
        spot_masked.extend(zs)
        for t in texts:
            spot_seq.append(judge.score_sequential(t, "4d")[0])
            spot_seq2d.append(judge.score_sequential(t, "2d")[0])
    zm = np.array(spot_masked)
    rho_spot = float(stats.spearmanr(spot_masked, spot_seq)[0])
    d_spot = float(np.max(np.abs(zm - np.array(spot_seq))))
    identical = bool(np.array_equal(zm, np.array(spot_seq)))
    rho_2d = float(stats.spearmanr(spot_masked, spot_seq2d)[0])
    d_2d = float(np.max(np.abs(zm - np.array(spot_seq2d))))
    logging.info("%s spot fidelity: %d chunks | vs matched-kernel sequential: "
                 "Spearman %.6f, max |Dz| %.4f, bit-identical %s | vs 2-D-mask "
                 "sequential: Spearman %.6f, max |Dz| %.4f"
                 % (corpus, len(spot_masked), rho_spot, d_spot, identical,
                    rho_2d, d_2d))
    spot_passed = bool(rho_spot >= FIDELITY_SPEARMAN_BAR)
    if not spot_passed and not args.sequential_reference:
        status("SPOT FIDELITY FAIL rho=%.4f" % rho_spot)
        raise SystemExit(
            "%s spot fidelity failed (Spearman %.4f < %.2f): stop and report"
            % (corpus, rho_spot, FIDELITY_SPEARMAN_BAR))
    if not spot_passed:
        logging.warning(
            "%s spot Spearman %.4f < %.2f. Superseded by the full sequential "
            "reference below: on this corpus z is tie-dominated, so Spearman "
            "converts sub-quantum bf16 noise into rank swaps. The endpoint "
            "comparison over every chunk is the measurement that decides."
            % (corpus, rho_spot, FIDELITY_SPEARMAN_BAR))

    # ---- full cohort, Arm M only, one forward per video
    status("forward 0/%d" % len(video_ids))
    rows = []
    t0 = time.time()
    tokens_masked = 0
    with open(per_chunk_path, "w") as fout:
        for i, vid in enumerate(video_ids):
            its = by_video[vid]
            zs, ntok = judge.packed_forward(
                judge.branches([it["text"] for it in its]))
            tokens_masked += ntok
            zref = ([judge.score_sequential(it["text"], "4d")[0] for it in its]
                    if args.sequential_reference else [None] * len(its))
            for it, z, zr in zip(its, zs, zref):
                rec = {"video_id": vid,
                       "video_label": "hate" if vid in hate_ids else "non_hate",
                       "chunk_index": it["chunk_index"],
                       "span": it["span"],
                       "chunk_tokens": it["chunk_tokens"],
                       "z_masked": z}
                if zr is not None:
                    rec["z_sequential"] = zr
                rows.append(rec)
                fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if i % 25 == 0:
                status("forward %d/%d  %.1f s" % (i, len(video_ids),
                                                  time.time() - t0))
                logging.info("  %d/%d  %.1f s" % (i, len(video_ids),
                                                  time.time() - t0))
    wall = time.time() - t0
    logging.info("%s: %d videos, %d chunks, %.1f s (%.3f s/video), "
                 "%d packed tokens" % (corpus, len(video_ids), len(rows), wall,
                                       wall / max(1, len(video_ids)),
                                       tokens_masked))

    # ---- frame-level evaluation against the frozen gold arrays
    status("evaluate")
    scores = frame_scores(gt, rows, spans_by_video, counts)
    res = evaluate_scores(scores, gt, hate_ids)

    # The full sequential reference: N genuine isolated calls per video, the
    # computation the one packed forward claims to reproduce. Its endpoint is
    # what decides whether the residual bf16 difference matters at all.
    ref_block = None
    if args.sequential_reference:
        zm = np.array([r["z_masked"] for r in rows], dtype=float)
        zr = np.array([r["z_sequential"] for r in rows], dtype=float)
        ref_scores = frame_scores(gt, rows, spans_by_video, counts,
                                  key="z_sequential")
        ref_res = evaluate_scores(ref_scores, gt, hate_ids)
        ref_block = {
            "note": "one isolated call per chunk, same attention path as the "
                    "packed arm; this is the reference Arm M reproduces",
            "n_chunks": len(rows),
            "spearman_masked_vs_sequential":
                float(stats.spearmanr(zm, zr)[0]),
            "pearson_masked_vs_sequential":
                float(stats.pearsonr(zm, zr)[0]),
            "max_abs_delta_z": float(np.max(np.abs(zm - zr))),
            "mean_abs_delta_z": float(np.mean(np.abs(zm - zr))),
            "frac_chunks_identical": float(np.mean(zm == zr)),
            "n_unique_z_sequential": int(len(np.unique(zr))),
            "frame_level": ref_res,
            "endpoint_delta_masked_minus_sequential": {
                "roc_auc": _sub(res["roc_auc"], ref_res["roc_auc"]),
                "pr_auc": _sub(res["pr_auc"], ref_res["pr_auc"]),
                "within_hate_macro_auc": _sub(res["per_video"]["macro_auc"],
                                              ref_res["per_video"]["macro_auc"]),
            },
        }
        logging.info("%s sequential reference: Spearman %.6f, Pearson %.6f, "
                     "max |Dz| %.4f, endpoint delta ROC %+.5f PR %+.5f"
                     % (corpus, ref_block["spearman_masked_vs_sequential"],
                        ref_block["pearson_masked_vs_sequential"],
                        ref_block["max_abs_delta_z"],
                        ref_block["endpoint_delta_masked_minus_sequential"]["roc_auc"],
                        ref_block["endpoint_delta_masked_minus_sequential"]["pr_auc"]))

    # Video-level readout: max chunk z per video against the video label.
    vmax = {v: (max(r["z_masked"] for r in rows if r["video_id"] == v)
                if any(r["video_id"] == v for r in rows) else counts[
                    "uncovered_floor"]) for v in sorted(gt)}
    video_auc = fec.rank_auc([vmax[v] for v in vmax if v in hate_ids],
                             [vmax[v] for v in vmax if v not in hate_ids])

    payload = {
        "method": "ours_zero_label_locator_arm_M",
        "corpus": corpus,
        "split": "test",
        "model": MODEL,
        "arm": "masked packed, one forward per video, branch-local positions",
        "mechanism_reference": {
            "script": "scripts/duplex/masked_parallel_isolation_pilot.py",
            "commit": "aa403cb",
            "note": "masked-vs-sequential equivalence and the causal-arm "
                    "collapse were established on HateMM and are not rerun",
        },
        "prompt_fingerprints": fp,
        "counts": counts,
        "spot_fidelity": {
            "videos": spot_ids,
            "chunks": len(spot_masked),
            "reference": "sequential isolated calls, one per chunk, on the "
                         "same attention path the packed arm uses",
            "spearman_masked_vs_sequential": rho_spot,
            "max_abs_delta_z": d_spot,
            "bit_identical": identical,
            "bar": FIDELITY_SPEARMAN_BAR,
            "passed": spot_passed,
            "against_2d_mask_reference": {
                "note": "the tokenizer's all-ones 2-D attention_mask sends HF "
                        "to a different SDPA kernel; recorded so the gap is "
                        "on the record, not used for the decision",
                "spearman": rho_2d,
                "max_abs_delta_z": d_2d,
            },
        },
        "mask_plumbing_max_abs_logit_delta": plumbing,
        "runtime": {"videos": len(video_ids), "chunks": len(rows),
                    "seconds": wall, "packed_tokens": tokens_masked,
                    "mllm_calls_per_video": 1},
        "frame_level": res,
        "sequential_reference": ref_block,
        "video_level": {"max_z_auc": video_auc,
                        "n_hate": len(hate_ids),
                        "n_non_hate": len(gt) - len(hate_ids)},
    }
    with open(os.path.join(out_dir, "frame_eval.json"), "w") as f:
        json.dump(payload, f, indent=2, default=float)

    print(format_report(res, "%s / test / ours (z_masked, all frames)" % corpus))
    print("  video-level max-z AUC   %s"
          % ("n/a" if video_auc is None else "%.4f" % video_auc))
    print("")
    status("DONE")
    with open(os.path.join(out_dir, "DONE"), "w") as f:
        f.write("%s\n" % time.strftime("%F %T"))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="mhclip_en,mhclip_zh")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--sequential-reference", action="store_true",
                    help="also score every chunk with a genuine isolated call "
                         "and report both endpoints; makes the spot-check bar "
                         "advisory, since the full comparison supersedes it")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    for corpus in [c.strip() for c in args.corpora.split(",") if c.strip()]:
        if corpus not in CORPUS_RULES:
            raise SystemExit("unknown corpus %r" % corpus)
        run_corpus(corpus, args)


if __name__ == "__main__":
    main()
