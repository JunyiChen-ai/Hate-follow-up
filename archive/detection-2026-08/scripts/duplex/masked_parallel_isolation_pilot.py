"""Masked parallel isolation pilot on HateMM (PREREGISTERED, direction 3).

Pre-registration: docs/duplex/PREREG_masked_parallel_isolation_pilot.md,
frozen at commit 7928b9b before any masked forward pass was run. This script
implements that protocol and nothing else; the decision rule below is copied
from the prereg and is evaluated mechanically.

What is being tested. The sequential isolated-chunk diagnostic
(`hatemm_isolated_chunk_diag.py`, commit 455f669) scores every transcript chunk
of a HateMM test video in its own call and recovers per-segment signal that
every in-context per-segment probe had lost. This pilot asks whether that
computation can be reproduced inside ONE forward pass per video, by packing the
shared rules prefix and all per-chunk branches into a single sequence and
cutting cross-chunk attention with a custom 4D mask.

Two arms over the identical packed token sequence and the identical readout
positions:

  * Arm M (masked): each branch attends to the shared prefix and to its own
    earlier positions only; every branch's position IDs restart at len(prefix).
    If the packing is exact, branch k computes what the sequential isolated
    call for chunk k computed.
  * Arm C (counterfactual): full causal attention, standard sequential
    position IDs. This is the ablation the story predicts must break: if
    cross-chunk attention is the contamination channel, removing the mask must
    collapse within-video separation toward the packed-questions result.

Prompt fidelity is asserted at runtime, not assumed: for every chunk the
concatenation of the prefix token ids and the branch token ids must equal, id
for id, the token ids of the sequential isolated prompt built by the imported
prompt code. The cohort, the chunk gold, the readout token sets and the AUC
machinery are imported from the sequential diagnostic; nothing is
reimplemented here.

Frozen decision rule (prereg): the direction SURVIVES only if BOTH
  (1) Spearman(Arm M, sequential reference) >= 0.99 over all chunks AND every
      frozen AUC contrast within +/-0.01 of the sequential reference
      (0.624 within-hate-video pooled, 0.720 span vs non-hate-video chunks);
  (2) Arm C within-hate-video macro AUC <= Arm M within-hate-video macro
      AUC - 0.05.
Clause-2 failure with clause 1 passing = DIES, no rescue variant. Clause-1
failure alone is an implementation defect: one bounded debugging pass is
permitted, after which the direction is recorded as implementation-infeasible.

Efficiency accounting is descriptive and carries no bar: wall-clock and token
counts for three regimes (sequential loop, plain left-padded batch, packed
masked pass) on a fixed set of videos.

Output: results/masked_parallel_isolation/{per_chunk.jsonl, report.json,
STATUS, DONE}; the caller redirects stdout to run.log. No transcript text is
written to any output.
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
from sentinel_localization_pilot import rank_auc  # noqa: E402
from isolated_chunk_diag import FROZEN_TEXT_SHA, user_text  # noqa: E402
from hatemm_isolated_chunk_diag import (  # noqa: E402
    MODEL,
    build_items,
    contrast,
    describe,
    macro_auc,
)

REF_DIR = os.path.join(PROJECT_ROOT, "results", "hatemm_localization")
REF_PER_CHUNK = os.path.join(REF_DIR, "per_chunk.jsonl")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "masked_parallel_isolation")

# The seam: everything before the chunk-specific text ends with this marker,
# which terminates in a newline, so the split is at a clean newline boundary.
SEAM_MARKER = "Transcript excerpt from a video:\n"
CHUNK_PLACEHOLDER = "<CHUNK>"

# Frozen reference numbers from the sequential diagnostic (prereg clause 1).
FROZEN_REF = {"within_hate_video_pooled_auc": 0.624,
              "span_vs_nonhate_video_chunks_auc": 0.720}
FIDELITY_SPEARMAN_BAR = 0.99
FIDELITY_AUC_TOL = 0.01
COUNTERFACTUAL_GAP_BAR = 0.05

EFFICIENCY_VIDEOS = 30


# ------------------------------------------------------------------- prompts

def build_prompt_parts(processor):
    """Split the frozen sequential prompt into a shared prefix and a suffix.

    Returns (prefix_text, suffix_text) such that, for any chunk text t, the
    sequential isolated prompt equals prefix_text + t + suffix_text.
    """
    def full(text):
        msgs = [{"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user",
                 "content": [{"type": "text", "text": user_text(text)}]}]
        return processor.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)

    template = full(CHUNK_PLACEHOLDER)
    cut = template.index(SEAM_MARKER) + len(SEAM_MARKER)
    prefix_text = template[:cut]
    suffix_text = template[cut + len(CHUNK_PLACEHOLDER):]
    assert prefix_text.endswith("\n")
    assert full("ABC") == prefix_text + "ABC" + suffix_text
    return prefix_text, suffix_text, full


# ------------------------------------------------------------- packed arrays

def block_mask(prefix_len, branch_lens, dtype, device):
    """4D additive mask: branch k sees the prefix and its own past only."""
    total = prefix_len + sum(branch_lens)
    allow = torch.zeros((total, total), dtype=torch.bool)
    idx = torch.arange(prefix_len)
    allow[:prefix_len, :prefix_len] = idx[:, None] >= idx[None, :]
    off = prefix_len
    for n in branch_lens:
        allow[off:off + n, :prefix_len] = True
        j = torch.arange(n)
        allow[off:off + n, off:off + n] = j[:, None] >= j[None, :]
        off += n
    return _to_additive(allow, dtype, device)


def causal_mask(total, dtype, device):
    idx = torch.arange(total)
    return _to_additive(idx[:, None] >= idx[None, :], dtype, device)


def _to_additive(allow, dtype, device):
    m = torch.zeros(allow.shape, dtype=dtype)
    m.masked_fill_(~allow, torch.finfo(dtype).min)
    return m[None, None].to(device)


def branch_position_ids(prefix_len, branch_lens, device):
    """Each branch restarts at len(prefix); mrope needs three identical rows."""
    pos = [torch.arange(prefix_len)]
    for n in branch_lens:
        pos.append(prefix_len + torch.arange(n))
    flat = torch.cat(pos)
    return flat[None, None, :].expand(3, 1, -1).contiguous().to(device)


def sequential_position_ids(total, device):
    flat = torch.arange(total)
    return flat[None, None, :].expand(3, 1, -1).contiguous().to(device)


# ------------------------------------------------------------------- scoring

class Judge:
    """The frozen judge, plus the packed-pass readout."""

    def __init__(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor

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
        self.prefix_text, self.suffix_text, self._full_prompt = \
            build_prompt_parts(self.processor)
        self.prefix_ids = self.encode(self.prefix_text)

    def encode(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def margin(self, logits_row):
        lg = logits_row.float()
        return float(torch.logsumexp(lg[self.yes_t], 0)
                     - torch.logsumexp(lg[self.no_t], 0))

    # -- the sequential reference call, identical to hatemm_isolated_chunk_diag
    def score_sequential(self, chunk_text):
        prompt = self._full_prompt(chunk_text)
        enc = self.tokenizer(prompt, return_tensors="pt",
                             add_special_tokens=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc, use_cache=False, logits_to_keep=1)
        return self.margin(out.logits[0, -1, :]), int(enc["input_ids"].shape[1])

    # -- branch construction with the mandatory fidelity assertion
    def branches(self, chunk_texts):
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

    def packed_forward(self, branches, arm):
        """One forward over [prefix, branch_1..branch_N]; z at each branch end."""
        p = len(self.prefix_ids)
        lens = [len(b) for b in branches]
        ids = list(self.prefix_ids)
        ends = []
        for b in branches:
            ids.extend(b)
            ends.append(len(ids) - 1)
        total = len(ids)
        if arm == "masked":
            mask = block_mask(p, lens, self.dtype, self.device)
            pos = branch_position_ids(p, lens, self.device)
        elif arm == "causal":
            mask = causal_mask(total, self.dtype, self.device)
            pos = sequential_position_ids(total, self.device)
        else:
            raise ValueError(arm)
        inp = torch.tensor([ids], device=self.device)
        keep = torch.tensor(ends, device=self.device)
        with torch.no_grad():
            out = self.model(input_ids=inp, attention_mask=mask,
                             position_ids=pos, use_cache=False,
                             logits_to_keep=keep)
        logits = out.logits[0]
        return [self.margin(logits[i, :]) for i in range(len(branches))], total

    def batch_forward(self, chunk_texts):
        """Plain left-padded batch of the N isolated prompts (efficiency arm)."""
        seqs = [self.encode(self._full_prompt(t)) for t in chunk_texts]
        width = max(len(s) for s in seqs)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        ids, att = [], []
        for s in seqs:
            k = width - len(s)
            ids.append([pad] * k + s)
            att.append([0] * k + [1] * len(s))
        inp = torch.tensor(ids, device=self.device)
        am = torch.tensor(att, device=self.device)
        with torch.no_grad():
            out = self.model(input_ids=inp, attention_mask=am,
                             use_cache=False, logits_to_keep=1)
        z = [self.margin(out.logits[i, -1, :]) for i in range(len(seqs))]
        return z, len(seqs) * width


# ------------------------------------------------------------ verification

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


# ------------------------------------------------------------------ analysis

def arm_stats(rows, key):
    """The diagnostic's frozen contrasts, computed on one arm's scores."""
    r2 = [dict(r, z_isolated=r[key]) for r in rows]
    hv = [r for r in r2 if r["video_label"] == "hate"]
    span = [r["z_isolated"] for r in hv if r["gold"] == "span"]
    nonspan = [r["z_isolated"] for r in hv if r["gold"] == "nonspan"]
    nonhate = [r["z_isolated"] for r in r2 if r["video_label"] == "non_hate"]
    macro, used = macro_auc(r2)
    return {
        "within_hate_video_pooled": contrast(span, nonspan),
        "within_hate_video_macro_auc": float(np.mean(macro)) if macro else None,
        "within_hate_video_macro_auc_sd":
            float(np.std(macro, ddof=1)) if len(macro) > 1 else None,
        "n_videos_both_classes": len(used),
        "span_vs_nonhate_video_chunks": contrast(span, nonhate),
        "score_all": describe([r[key] for r in rows]),
    }


def _sub(a, b):
    return None if (a is None or b is None) else a - b


def analyze(rows, counts, efficiency, plumbing):
    zm = np.array([r["z_masked"] for r in rows], dtype=np.float64)
    zc = np.array([r["z_causal"] for r in rows], dtype=np.float64)
    zr = np.array([r["z_reference"] for r in rows], dtype=np.float64)
    rho_m, p_m = stats.spearmanr(zm, zr)
    rho_c, p_c = stats.spearmanr(zc, zr)

    masked = arm_stats(rows, "z_masked")
    causal = arm_stats(rows, "z_causal")
    reference = arm_stats(rows, "z_reference")

    d_pooled = _sub(masked["within_hate_video_pooled"]["auc"],
                    reference["within_hate_video_pooled"]["auc"])
    d_cross = _sub(masked["span_vs_nonhate_video_chunks"]["auc"],
                   reference["span_vs_nonhate_video_chunks"]["auc"])
    d_pooled_frozen = _sub(masked["within_hate_video_pooled"]["auc"],
                           FROZEN_REF["within_hate_video_pooled_auc"])
    d_cross_frozen = _sub(masked["span_vs_nonhate_video_chunks"]["auc"],
                          FROZEN_REF["span_vs_nonhate_video_chunks_auc"])

    clause1 = bool(rho_m >= FIDELITY_SPEARMAN_BAR
                   and d_pooled_frozen is not None
                   and abs(d_pooled_frozen) <= FIDELITY_AUC_TOL
                   and d_cross_frozen is not None
                   and abs(d_cross_frozen) <= FIDELITY_AUC_TOL)
    gap = _sub(masked["within_hate_video_macro_auc"],
               causal["within_hate_video_macro_auc"])
    clause2 = bool(gap is not None and gap >= COUNTERFACTUAL_GAP_BAR)
    if clause1 and clause2:
        verdict = "SURVIVES"
    elif clause1 and not clause2:
        verdict = "DIES"
    else:
        verdict = "FIDELITY_FAIL"

    return {
        "preregistered": True,
        "prereg": "docs/duplex/PREREG_masked_parallel_isolation_pilot.md",
        "prereg_commit": "7928b9b",
        "dataset": "HateMM",
        "split": "test_clean",
        "model": MODEL,
        "arms": ["masked", "causal"],
        "counts": counts,
        "frozen_text_sha256": FROZEN_TEXT_SHA,
        "prefix_tokens": counts["prefix_tokens"],
        "mask_plumbing_max_abs_logit_delta": plumbing,
        "fidelity": {
            "spearman_masked_vs_reference": {"rho": float(rho_m),
                                             "p": float(p_m)},
            "spearman_causal_vs_reference": {"rho": float(rho_c),
                                             "p": float(p_c)},
            "max_abs_delta_z_masked_vs_reference":
                float(np.max(np.abs(zm - zr))),
            "mean_abs_delta_z_masked_vs_reference":
                float(np.mean(np.abs(zm - zr))),
            "max_abs_delta_z_causal_vs_reference":
                float(np.max(np.abs(zc - zr))),
            "delta_auc_within_pooled_vs_recomputed_reference": d_pooled,
            "delta_auc_cross_video_vs_recomputed_reference": d_cross,
            "delta_auc_within_pooled_vs_frozen": d_pooled_frozen,
            "delta_auc_cross_video_vs_frozen": d_cross_frozen,
        },
        "arm_masked": masked,
        "arm_causal": causal,
        "arm_reference_recomputed": reference,
        "counterfactual_gap_macro_masked_minus_causal":
            None if gap is None else float(gap),
        "decision": {
            "clause1_fidelity": clause1,
            "clause2_counterfactual_gap": clause2,
            "verdict": verdict,
            "bars": {"spearman": FIDELITY_SPEARMAN_BAR,
                     "auc_tolerance": FIDELITY_AUC_TOL,
                     "macro_gap": COUNTERFACTUAL_GAP_BAR,
                     "frozen_reference": FROZEN_REF},
        },
        "efficiency": efficiency,
    }


# ---------------------------------------------------------------------- main

def load_reference():
    ref = {}
    with open(REF_PER_CHUNK) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ref[(r["video_id"], r["chunk_index"])] = r
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--smoke-videos", type=int, default=3)
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--efficiency-videos", type=int, default=EFFICIENCY_VIDEOS)
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
            f.write("%s  %s\n" % (time.strftime("%F %T"), s))

    status("cohort")
    counts = {k: 0 for k in (
        "videos_no_chunks", "videos_unusable_spans",
        "hate_videos_without_span_gold", "videos_used_hate",
        "videos_used_non_hate", "chunks_total", "chunks_empty_text",
        "chunks_bad_span", "chunks_hate_span", "chunks_hate_nonspan",
        "chunks_non_hate_video")}
    items = build_items(counts)
    ref = load_reference()
    for it in items:
        r = ref.get((it["video_id"], it["chunk_index"]))
        if r is None:
            raise AssertionError("cohort drift: %s/%d absent from the "
                                 "sequential reference"
                                 % (it["video_id"], it["chunk_index"]))
        if r["text"] != it["text"]:
            raise AssertionError("chunk text drift: %s/%d"
                                 % (it["video_id"], it["chunk_index"]))
    if len(items) != len(ref):
        raise AssertionError("cohort size %d != reference %d"
                             % (len(items), len(ref)))
    logging.info("cohort matches the sequential reference: %d chunks"
                 % len(items))

    by_video = {}
    for it in items:
        by_video.setdefault(it["video_id"], []).append(it)
    for vid in by_video:
        by_video[vid].sort(key=lambda r: r["chunk_index"])
    video_ids = sorted(by_video)
    if args.limit_videos:
        video_ids = video_ids[:args.limit_videos]

    if not args.analyze_only:
        status("load model")
        judge = Judge()
        counts["prefix_tokens"] = len(judge.prefix_ids)
        counts["prefix_sha256"] = hashlib.sha256(
            judge.prefix_text.encode()).hexdigest()
        counts["suffix_sha256"] = hashlib.sha256(
            judge.suffix_text.encode()).hexdigest()
        logging.info("prefix %d tokens, sha %s"
                     % (counts["prefix_tokens"], counts["prefix_sha256"][:16]))

        # ---- verification order: plumbing, then three videos against the
        # sequential reference, before any full-cohort run.
        status("verify plumbing")
        # Smoke set: the videos with the most chunks, so the block mask is
        # actually exercised (the first videos by id have one chunk each,
        # where masked and causal packing coincide).
        smoke_ids = sorted(video_ids,
                           key=lambda v: (-len(by_video[v]), v)
                           )[:args.smoke_videos]
        logging.info("smoke videos %s (chunks %s)"
                     % (smoke_ids, [len(by_video[v]) for v in smoke_ids]))
        plumbing = verify_mask_plumbing(
            judge, [it["text"] for it in by_video[smoke_ids[0]]][:4])

        status("verify fidelity on %d videos" % len(smoke_ids))
        sm_masked, sm_ref, sm_fresh = [], [], []
        for vid in smoke_ids:
            texts = [it["text"] for it in by_video[vid]]
            zs, _ = judge.packed_forward(judge.branches(texts), "masked")
            sm_masked.extend(zs)
            for it in by_video[vid]:
                sm_ref.append(ref[(vid, it["chunk_index"])]["z_isolated"])
                sm_fresh.append(judge.score_sequential(it["text"])[0])
        rho_s = float(stats.spearmanr(sm_masked, sm_ref)[0])
        d_stored = float(np.max(np.abs(np.array(sm_masked)
                                       - np.array(sm_ref))))
        d_fresh = float(np.max(np.abs(np.array(sm_masked)
                                      - np.array(sm_fresh))))
        logging.info("smoke: %d chunks, Spearman(masked, stored ref) %.6f, "
                     "max |Dz| vs stored %.4f, vs fresh sequential %.4f"
                     % (len(sm_masked), rho_s, d_stored, d_fresh))
        if not (rho_s >= 0.99):
            status("SMOKE FIDELITY FAIL rho=%.4f" % rho_s)
            raise SystemExit(
                "smoke fidelity failed (Spearman %.4f < 0.99): stop and debug "
                "position ids / seam tokenization / mask dtype" % rho_s)

        # ---- full cohort, both arms
        status("forward 0/%d" % len(video_ids))
        fout = open(per_chunk_path, "w")
        t0 = time.time()
        tok_masked = tok_causal = 0
        for i, vid in enumerate(video_ids):
            its = by_video[vid]
            branches = judge.branches([it["text"] for it in its])
            zm, n_m = judge.packed_forward(branches, "masked")
            zc, n_c = judge.packed_forward(branches, "causal")
            tok_masked += n_m
            tok_causal += n_c
            for it, a, b in zip(its, zm, zc):
                fout.write(json.dumps({
                    "video_id": vid,
                    "video_label": it["video_label"],
                    "chunk_index": it["chunk_index"],
                    "gold": it["gold"],
                    "overlap_fraction": it["overlap_fraction"],
                    "chunk_tokens": it["chunk_tokens"],
                    "z_masked": a,
                    "z_causal": b,
                    "z_reference": ref[(vid, it["chunk_index"])]["z_isolated"],
                }) + "\n")
            fout.flush()
            if i % 20 == 0:
                status("forward %d/%d  %.1f s" % (i, len(video_ids),
                                                  time.time() - t0))
                logging.info("  %d/%d  %.1f s" % (i, len(video_ids),
                                                  time.time() - t0))
        fout.close()
        logging.info("both arms done: %d videos, %.1f s"
                     % (len(video_ids), time.time() - t0))

        # ---- descriptive efficiency accounting on a fixed video set
        status("efficiency")
        eff_ids = video_ids[:args.efficiency_videos]
        acc = {"videos": len(eff_ids), "chunks": 0,
               "sequential": {"seconds": 0.0, "tokens": 0},
               "padded_batch": {"seconds": 0.0, "tokens": 0, "failures": 0},
               "packed_masked": {"seconds": 0.0, "tokens": 0}}
        for vid in eff_ids:
            texts = [it["text"] for it in by_video[vid]]
            acc["chunks"] += len(texts)

            torch.cuda.synchronize()
            t = time.time()
            n_seq = 0
            for x in texts:
                n_seq += judge.score_sequential(x)[1]
            torch.cuda.synchronize()
            acc["sequential"]["seconds"] += time.time() - t
            acc["sequential"]["tokens"] += n_seq

            try:
                torch.cuda.synchronize()
                t = time.time()
                _, n_pad = judge.batch_forward(texts)
                torch.cuda.synchronize()
                acc["padded_batch"]["seconds"] += time.time() - t
                acc["padded_batch"]["tokens"] += n_pad
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                acc["padded_batch"]["failures"] += 1
                logging.info("  padded batch OOM on %s (%d chunks)"
                             % (vid, len(texts)))

            branches = judge.branches(texts)
            torch.cuda.synchronize()
            t = time.time()
            _, n_pack = judge.packed_forward(branches, "masked")
            torch.cuda.synchronize()
            acc["packed_masked"]["seconds"] += time.time() - t
            acc["packed_masked"]["tokens"] += n_pack

        for k in ("sequential", "padded_batch", "packed_masked"):
            n = max(1, acc["videos"] - acc[k].get("failures", 0))
            acc[k]["seconds_per_video"] = acc[k]["seconds"] / n
            acc[k]["tokens_per_video"] = acc[k]["tokens"] / n
        acc["full_cohort_tokens_masked"] = tok_masked
        acc["full_cohort_tokens_causal"] = tok_causal
        acc["smoke"] = {"videos": smoke_ids, "chunks": len(sm_masked),
                        "spearman_vs_stored_reference": rho_s,
                        "max_abs_delta_z_vs_stored": d_stored,
                        "max_abs_delta_z_vs_fresh_sequential": d_fresh}
        with open(os.path.join(args.out_dir, "efficiency.json"), "w") as f:
            json.dump(acc, f, indent=2)
        with open(os.path.join(args.out_dir, "counts.json"), "w") as f:
            json.dump(counts, f, indent=2)
    else:
        with open(os.path.join(args.out_dir, "efficiency.json")) as f:
            acc = json.load(f)
        with open(os.path.join(args.out_dir, "counts.json")) as f:
            counts = json.load(f)
        plumbing = None

    status("analyze")
    rows = []
    with open(per_chunk_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    report = analyze(rows, counts, acc, plumbing)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logging.info(json.dumps({k: report[k] for k in (
        "fidelity", "arm_masked", "arm_causal", "arm_reference_recomputed",
        "counterfactual_gap_macro_masked_minus_causal", "decision",
        "efficiency")}, indent=2))
    status("DONE %s" % report["decision"]["verdict"])
    with open(os.path.join(args.out_dir, "DONE"), "w") as f:
        f.write("%s %s\n" % (time.strftime("%F %T"),
                             report["decision"]["verdict"]))


if __name__ == "__main__":
    main()
