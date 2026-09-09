#!/usr/bin/env python3
"""SPVL: single-pass verdict-and-evidence localization (label-free).

One Qwen3-VL-8B forward per video:
  PREFIX   = system message + user turn [K timestamped frames + timestamped Whisper
             transcript + hate-speech rules + reading instruction]   (closed with <|im_end|>)
  BRANCH_0 = user turn asking the whole-video question + assistant header  -> z_video
  BRANCH_i = user turn asking about window i (t1..t2 s, its transcript) + assistant header -> z_i
Attention: prefix causal; each branch sees the prefix and its own past only (block-diagonal),
so every branch equals an independent 3-turn call. Read Yes/No log-odds at each branch end.

Outputs runs/<exp_id>/<run_name>/predictions.jsonl with the raw per-window curve on the 4 fps
grid (score_curve = z of the window covering each frame) and extra.z_video; compose.py adds the
video intercept and rank residual. No labels are read anywhere in this file.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CODE_PATH = "experiments/20260910_spvl/spvl.py"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
FPS = 4.0
MAX_PIXELS = 100352   # per-frame pixel cap (same as the 2026-08 whole-video judge)
MIN_PIXELS = 65536
FILL_UNCOVERED = -12.0  # frames outside any ASR window (asr mode only)
SEED = 0

# ---- frozen prompt material (copied verbatim from the 2026-08 judge; do not edit) ----
SYSTEM_MESSAGE = "You are a content moderation analyst. Answer based strictly on observable evidence."
YOUTUBE_RULES = """1. Encourage violence against groups based on protected status
2. Incite hatred against groups based on protected status
3. Dehumanize groups by comparing to non-human entities
4. Praise or glorify violence against protected groups
5. Use slurs and stereotypes that incite hatred (including through humor, irony, or coded language)
6. Claim groups are inferior based on protected status
7. Promote hateful supremacism
8. Make conspiratorial claims about groups based on protected status
9. Deny or minimize major violent events"""
READER_PRAG = (
    "Judgment mode: full reading. Judge what the video communicates as "
    "a whole: both what is explicitly said or shown and what is conveyed "
    "beyond the surface. Consider coded or euphemistic references, "
    "wordplay and homophones, ironic or mocking framing, how the title "
    "reframes the visuals, the joint effect of images, audio, and text, "
    "and how the referenced group or an informed viewer would receive "
    "the message. A rule is violated if either the explicit content or "
    "the conveyed meaning violates it."
)
# legacy per-chunk prompt (2026-08 `stance_complete_asserted_event`), for the parity arm only
LEGACY_POLICY = (
    "Hateful content is an asserted or endorsed attack, dehumanization, threat, "
    "exclusion, or hateful stereotype directed at a person or protected/social "
    "group. Quotation, neutral reporting, counterspeech, satire, and condemnation "
    "are not endorsement."
)
LEGACY_QUESTION = "Does this transcript contain at least one complete asserted or endorsed hateful event?"

VIDEO_QUESTION = ('Does this video contain content that violates any of the above rules?\n\n'
                  'Answer "Yes" or "No".')


def window_question(i, n, t1, t2, text):
    body = text.strip() if text and text.strip() else "(no speech)"
    return (f"Consider only window {i + 1} of {n}, from {t1:.1f}s to {t2:.1f}s of this video. "
            f"Transcript in this window: {body}\n\n"
            "Does THIS window contain content that violates any of the above rules?\n\n"
            'Answer "Yes" or "No".')


# ------------------------------------------------------------------ data
def load_manifest(path, datasets):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if r["dataset"] in datasets]


def load_asr(dataset):
    path = ROOT / f"data/asr_whisper_large_v3/{dataset}/timestamped_chunks.jsonl"
    out = {}
    for line in open(path):
        r = json.loads(line)
        segs = [(float(c["start"]), float(c["end"]), (c.get("text") or "").strip())
                for c in (r.get("chunks") or []) if c.get("end") is not None and c.get("start") is not None]
        out[r["video_id"]] = [s for s in segs if s[1] > s[0]]
    return out


def frame_paths(dataset, vid, k):
    d = ROOT / f"data/frames_k20/{dataset}/{vid}"
    files = sorted(d.glob("f*_t*.jpg"))
    if not files:
        return []
    if k < len(files):
        idx = np.linspace(0, len(files) - 1, k).round().astype(int)
        files = [files[i] for i in idx]
    out = []
    for f in files:
        t = float(f.stem.split("_t")[1])
        out.append((t, f))
    return out


def fixed_windows(duration, seconds):
    n = max(1, int(math.ceil(duration / seconds - 1e-9)))
    return [(i * seconds, min((i + 1) * seconds, duration)) for i in range(n)]


def window_text(segments, t1, t2):
    """Transcript inside [t1,t2]: segments are sliced proportionally at word boundaries."""
    parts = []
    for s, e, text in segments:
        lo, hi = max(s, t1), min(e, t2)
        if hi <= lo or not text:
            continue
        if s >= t1 and e <= t2:
            parts.append(text)
            continue
        words = text.split()
        if not words:
            continue
        a = int(round((lo - s) / (e - s) * len(words)))
        b = int(round((hi - s) / (e - s) * len(words)))
        piece = " ".join(words[a:b]).strip()
        if piece:
            parts.append(piece)
    return " ".join(parts)


def transcript_block(segments):
    if not segments:
        return "(no speech detected)"
    return "\n".join(f"[{s:.1f}s-{e:.1f}s] {t}" for s, e, t in segments if t)


# ------------------------------------------------------------------ packing
def block_allow(prefix_len, branch_lens):
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
    return allow


def causal_allow(total):
    idx = torch.arange(total)
    return idx[:, None] >= idx[None, :]


def to_mask(allow, dtype, device, kind):
    if kind == "bool":
        return allow[None, None].to(device)
    m = torch.zeros(allow.shape, dtype=dtype)
    m.masked_fill_(~allow, torch.finfo(dtype).min)
    return m[None, None].to(device)


class Judge:
    def __init__(self, mask_kind="bool"):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(MODEL)
        self.tok = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = self.model.device
        self.dtype = torch.bfloat16
        self.mask_kind = mask_kind
        self.yes_ids, self.no_ids = self.binary_ids()
        self.yes_t = torch.tensor(self.yes_ids, device=self.device)
        self.no_t = torch.tensor(self.no_ids, device=self.device)
        logging.info("Yes ids %s No ids %s", self.yes_ids, self.no_ids)
        self.size_kw = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}

    def binary_ids(self):
        def first(s):
            ids = self.tok.encode(s, add_special_tokens=False)
            return ids[0] if ids else None
        out = []
        for label in ("Yes", "No"):
            s = set()
            for v in (label, f" {label}", label.lower(), f" {label.lower()}", label.upper(), f" {label.upper()}"):
                t = first(v)
                if t is not None:
                    s.add(t)
            out.append(sorted(s))
        return out[0], out[1]

    def margin(self, row):
        lg = row.float()
        return float(torch.logsumexp(lg[self.yes_t], 0) - torch.logsumexp(lg[self.no_t], 0))

    def margins_fp32(self, hidden_rows):
        """Yes/No log-odds from hidden states with the lm_head applied in fp32 (avoids bf16 logit steps)."""
        W = self.model.lm_head.weight
        ids = torch.cat([self.yes_t, self.no_t])
        lg = hidden_rows.float() @ W[ids].float().T  # [n, |yes|+|no|]
        ny = len(self.yes_ids)
        return (torch.logsumexp(lg[:, :ny], 1) - torch.logsumexp(lg[:, ny:], 1)).tolist()

    # ---- prompt construction
    def prefix_messages(self, frames, segments, with_context, with_frames):
        content = []
        intro = "Observe this video."
        if with_frames and frames:
            intro += f" Below are {len(frames)} frames sampled from the video, each preceded by its timestamp."
        if with_context:
            intro += " The full transcript with timestamps follows the frames." if (with_frames and frames) \
                else " The full transcript with timestamps follows."
        content.append({"type": "text", "text": intro + "\n"})
        images = []
        if with_frames:
            for t, f in frames:
                content.append({"type": "text", "text": f"[t={t:.1f}s]\n"})
                content.append({"type": "image"})
                images.append(f)
        tail = ""
        if with_context:
            tail += "\nTranscript:\n" + transcript_block(segments) + "\n"
        tail += ("\nBased on this platform's hate speech policy, the following types of content are prohibited:\n"
                 f"{YOUTUBE_RULES}\n\n{READER_PRAG}\n")
        content.append({"type": "text", "text": tail})
        msgs = [{"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": content}]
        return msgs, images

    def render(self, msgs, add_generation_prompt):
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)

    def encode_prefix(self, msgs, image_files):
        from PIL import Image
        text = self.render(msgs, add_generation_prompt=False)
        images = [Image.open(p).convert("RGB") for p in image_files]
        if images:
            enc = self.processor(text=[text], images=images, return_tensors="pt", size=self.size_kw)
        else:
            enc = self.processor(text=[text], return_tensors="pt")
        for im in images:
            im.close()
        if "image_grid_thw" in enc:
            grid = enc["image_grid_thw"]
            merge = getattr(self.processor.image_processor, "merge_size", 2)
            patch = getattr(self.processor.image_processor, "patch_size", 16)
            px = [int(h * w) * patch * patch for _, h, w in grid.tolist()]
            if max(px) > MAX_PIXELS:
                raise SystemExit(f"pixel cap not honoured: {max(px)} > {MAX_PIXELS}")
            self.img_tokens = [int(t * h * w) // (merge ** 2) for t, h, w in grid.tolist()]
        else:
            self.img_tokens = []
        return text, enc

    def branch_ids(self, question):
        text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        return self.tok(text, add_special_tokens=False)["input_ids"], text

    def seam_check_text(self, msgs, prefix_text, question, branch_text):
        full = self.render(msgs + [{"role": "user", "content": [{"type": "text", "text": question}]}],
                           add_generation_prompt=True)
        if full != prefix_text + branch_text:
            raise AssertionError("string seam mismatch")

    def seam_check_tokens(self, msgs, image_files, prefix_ids, question, bids):
        from PIL import Image
        full = self.render(msgs + [{"role": "user", "content": [{"type": "text", "text": question}]}],
                           add_generation_prompt=True)
        images = [Image.open(p).convert("RGB") for p in image_files]
        enc = (self.processor(text=[full], images=images, return_tensors="pt", size=self.size_kw)
               if images else self.processor(text=[full], return_tensors="pt"))
        for im in images:
            im.close()
        got = enc["input_ids"][0].tolist()
        if got != prefix_ids + bids:
            raise AssertionError(f"token seam mismatch: {len(prefix_ids)}+{len(bids)} vs {len(got)}")

    def mm_types(self, enc, input_ids=None):
        """mm_token_type_ids (0 text, 1 image) for the given ids; derived if the processor omitted it."""
        ids = enc["input_ids"] if input_ids is None else input_ids
        if "mm_token_type_ids" in enc and input_ids is None:
            return enc["mm_token_type_ids"].to(self.device)
        img_id = self.model.config.image_token_id
        return (ids == img_id).to(torch.int32).to(self.device)

    # ---- positions
    def prefix_positions(self, enc):
        """[3,1,P] mrope positions for the prefix from the model's own rope index."""
        ids = enc["input_ids"].to(self.device)
        if "image_grid_thw" not in enc:
            P = ids.shape[1]
            return torch.arange(P, device=self.device)[None, None].expand(3, 1, -1).contiguous()
        fn = self.model.model.get_rope_index
        params = inspect.signature(fn).parameters
        kw = {"input_ids": ids}
        if "image_grid_thw" in params:
            kw["image_grid_thw"] = enc["image_grid_thw"].to(self.device)
        if "mm_token_type_ids" in params:
            kw["mm_token_type_ids"] = self.mm_types(enc)
        if "attention_mask" in params:
            kw["attention_mask"] = torch.ones_like(ids)
        pos, _delta = fn(**kw)
        return pos.to(self.device)  # [3,1,P]

    def packed_positions(self, pos_p, branch_lens, sequential=False):
        s = int(pos_p.max().item()) + 1
        parts = [pos_p]
        off = s
        for n in branch_lens:
            start = off if sequential else s
            parts.append(torch.arange(start, start + n, device=self.device)[None, None].expand(3, 1, -1))
            if sequential:
                off += n
        return torch.cat(parts, dim=-1).contiguous()

    # ---- forward
    @torch.no_grad()
    def packed_forward(self, enc, pos_p, branches, arm="block"):
        prefix_ids = enc["input_ids"][0].tolist()
        P = len(prefix_ids)
        lens = [len(b) for b in branches]
        ids = list(prefix_ids)
        ends = []
        for b in branches:
            ids.extend(b)
            ends.append(len(ids) - 1)
        total = len(ids)
        if arm == "block":
            allow = block_allow(P, lens)
            pos = self.packed_positions(pos_p, lens, sequential=False)
        elif arm == "causal":
            allow = causal_allow(total)
            pos = self.packed_positions(pos_p, lens, sequential=True)
        else:
            raise ValueError(arm)
        mask = to_mask(allow, self.dtype, self.device, self.mask_kind)
        kw = {"input_ids": torch.tensor([ids], device=self.device),
              "attention_mask": mask, "position_ids": pos, "use_cache": False,
              "logits_to_keep": torch.tensor(ends, device=self.device)}
        if "pixel_values" in enc:
            kw["pixel_values"] = enc["pixel_values"].to(self.device, self.dtype)
            kw["image_grid_thw"] = enc["image_grid_thw"].to(self.device)
        keep = kw.pop("logits_to_keep")
        out = self.model.model(**kw)
        hidden = out.last_hidden_state[0, keep]  # [N, H]
        z = self.margins_fp32(hidden)
        del out, hidden, mask
        return z, total

    @torch.no_grad()
    def plain_forward(self, msgs, image_files, question):
        """Reference: one ordinary call (no custom mask / positions)."""
        from PIL import Image
        full = self.render(msgs + [{"role": "user", "content": [{"type": "text", "text": question}]}],
                           add_generation_prompt=True)
        images = [Image.open(p).convert("RGB") for p in image_files]
        enc = (self.processor(text=[full], images=images, return_tensors="pt", size=self.size_kw)
               if images else self.processor(text=[full], return_tensors="pt"))
        for im in images:
            im.close()
        kw = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
        if "pixel_values" in kw:
            kw["pixel_values"] = kw["pixel_values"].to(self.dtype)
            kw["mm_token_type_ids"] = self.mm_types(enc, enc["input_ids"])
        out = self.model.model(**kw, use_cache=False)
        hidden = out.last_hidden_state[0, -1:]
        z = self.margins_fp32(hidden)[0]
        del out
        return z, None, int(enc["input_ids"].shape[1])

    @torch.no_grad()
    def legacy_chunk_scores(self, texts, batch=8):
        """2026-08 per-chunk prompt (no system message, single Yes/No ids), batched calls."""
        yes = self.tok.encode("Yes", add_special_tokens=False)[0]
        no = self.tok.encode("No", add_special_tokens=False)[0]
        out = []
        for i in range(0, len(texts), batch):
            prompts = []
            for t in texts[i:i + batch]:
                sample = t if len(t) <= 6000 else t[:3000] + ' ... [middle omitted deterministically] ... ' + t[-3000:]
                prompts.append(f'{LEGACY_POLICY} Transcript: {json.dumps(sample, ensure_ascii=False)} {LEGACY_QUESTION} Answer Yes or No only:')
            msgs = [[{"role": "user", "content": [{"type": "text", "text": p}]}] for p in prompts]
            texts_r = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            inp = self.processor(text=texts_r, padding=True, return_tensors="pt").to(self.device)
            lg = self.model(**inp, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            out.extend((lg[:, yes] - lg[:, no]).cpu().tolist())
        return out


# ------------------------------------------------------------------ per video
def score_video(judge, row, segments, args, verify=False):
    vid, ds, dur = row["video_id"], row["dataset"], float(row["duration"])
    frames = frame_paths(ds, vid, args.frames) if args.frames > 0 else []
    if args.frames > 0 and not frames:
        return None, {"error": "no frames cached"}
    with_context = not args.no_transcript_context
    msgs, image_files = judge.prefix_messages(frames, segments, with_context, args.frames > 0)
    prefix_text, enc = judge.encode_prefix(msgs, image_files)
    prefix_ids = enc["input_ids"][0].tolist()
    if args.windows == "fixed":
        wins = fixed_windows(dur, args.window_seconds)
        wtexts = [window_text(segments, a, b) for a, b in wins]
    else:  # asr
        wins = [(s, e) for s, e, _ in segments]
        wtexts = [t for _, _, t in segments]
    questions = [VIDEO_QUESTION] + [window_question(i, len(wins), a, b, t)
                                    for i, ((a, b), t) in enumerate(zip(wins, wtexts))]
    branches, btexts = [], []
    for q in questions:
        bids, btext = judge.branch_ids(q)
        judge.seam_check_text(msgs, prefix_text, q, btext)
        branches.append(bids)
        btexts.append(btext)
    if verify:
        for q, bids in list(zip(questions, branches))[:3]:
            judge.seam_check_tokens(msgs, image_files, prefix_ids, q, bids)
    pos_p = judge.prefix_positions(enc)
    P = len(prefix_ids)
    # group branches so that P + sum(lens) <= max_tokens
    groups, cur, cur_len = [], [], 0
    for i, b in enumerate(branches):
        if cur and P + cur_len + len(b) > args.max_tokens:
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append(i)
        cur_len += len(b)
    if cur:
        groups.append(cur)
    zs = [None] * len(branches)
    total_tokens = 0
    for g in groups:
        z, tot = judge.packed_forward(enc, pos_p, [branches[i] for i in g], arm=args.mask)
        total_tokens += tot
        for i, v in zip(g, z):
            zs[i] = v
    z_video, z_win = zs[0], zs[1:]
    info = {"prefix_tokens": P, "total_tokens": total_tokens, "n_groups": len(groups),
            "n_windows": len(wins), "img_tokens": judge.img_tokens[:1]}
    if verify:
        checks = {}
        # (i) packed branch_0 vs plain call
        z_ref, _, _ = judge.plain_forward(msgs, image_files, VIDEO_QUESTION)
        z_pk, _ = judge.packed_forward(enc, pos_p, [branches[0]], arm="block")
        checks["video_q_packed_vs_plain_dz"] = abs(z_pk[0] - z_ref)
        # (ii) all-causal packed vs plain sequential of full concatenation (positions from model)
        ids = torch.tensor([prefix_ids + sum(branches[:4], [])], device=judge.device)
        kw = {"input_ids": ids, "use_cache": False, "logits_to_keep": 8}
        if "pixel_values" in enc:
            kw["pixel_values"] = enc["pixel_values"].to(judge.device, judge.dtype)
            kw["image_grid_thw"] = enc["image_grid_thw"].to(judge.device)
        with torch.no_grad():
            base_kw = dict(kw)
            if "pixel_values" in enc:
                base_kw["mm_token_type_ids"] = judge.mm_types(enc, ids.cpu())
            base = judge.model(**base_kw).logits[0].float()
            lens = [len(b) for b in branches[:4]]
            allow = causal_allow(ids.shape[1])
            pos = judge.packed_positions(pos_p, lens, sequential=True)
            got = judge.model(**kw, attention_mask=to_mask(allow, judge.dtype, judge.device, judge.mask_kind),
                              position_ids=pos).logits[0].float()
        checks["causal4d_vs_nomask_max_dlogit"] = float((base - got).abs().max())
        # (iii) 5 windows packed vs 5 plain calls
        n = min(5, len(z_win))
        seq = [judge.plain_forward(msgs, image_files, questions[1 + i])[0] for i in range(n)]
        pk, _ = judge.packed_forward(enc, pos_p, branches[1:1 + n], arm="block")
        checks["windows_packed_vs_plain_max_dz"] = max(abs(a - b) for a, b in zip(pk, seq)) if n else 0.0
        if n >= 3:
            from scipy.stats import spearmanr
            checks["windows_packed_vs_plain_spearman"] = float(spearmanr(pk, seq).correlation)
        checks["peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
        info["verify"] = checks
    # paint the 4 fps curve with the raw window z
    L = int(math.ceil(dur * FPS))
    curve = np.full(L, FILL_UNCOVERED, dtype=float)
    centers = (np.arange(L) + 0.5) / FPS
    if args.windows == "fixed":
        idx = np.minimum((centers // args.window_seconds).astype(int), len(wins) - 1)
        curve = np.asarray(z_win, dtype=float)[idx]
    else:
        for (a, b), z in zip(wins, z_win):
            i0, i1 = max(0, int(math.floor(a * FPS))), min(L, int(math.ceil(b * FPS)))
            if i1 > i0:
                curve[i0:i1] = np.maximum(curve[i0:i1], z)
    pred = {"schema_version": 1, "method": args.method_name, "dataset": ds, "video_id": vid,
            "duration": dur, "native_rate": FPS, "score_curve": [float(x) for x in curve],
            "intervals": [], "error": None, "calls": len(groups), "seed": SEED, "code_path": CODE_PATH,
            "extra": {"z_video": z_video, "windows": [{"i": i, "start": a, "end": b, "z": z}
                                                      for i, ((a, b), z) in enumerate(zip(wins, z_win))],
                      **info}}
    return pred, info


def within_defined_ids(datasets):
    """Video ids whose 4 fps GT has both classes: used ONLY to pick the pilot subset."""
    out = set()
    for ds in datasets:
        g = np.load(ROOT / f"data/gt_4fps/{ds}.npz", allow_pickle=True)
        for vid, y in zip(g["video_ids"], g["y4"]):
            y = np.asarray(y)
            if len(y) and y.min() != y.max():
                out.add((ds, str(vid)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--exp-id", default="20260910_spvl")
    ap.add_argument("--datasets", nargs="+", default=["HateMM", "HateClipSeg"])
    ap.add_argument("--manifest", default=str(ROOT / "data/omsl_v6_inputs/manifests/all_test.jsonl"))
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--no-transcript-context", action="store_true")
    ap.add_argument("--windows", choices=["fixed", "asr"], default="fixed")
    ap.add_argument("--window-seconds", type=float, default=8.0)
    ap.add_argument("--mask", choices=["block", "causal"], default="block")
    ap.add_argument("--mask-kind", choices=["bool", "additive"], default="bool")
    ap.add_argument("--max-tokens", type=int, default=9000)
    ap.add_argument("--only-within-defined", action="store_true",
                    help="pilot subset: videos whose GT has both classes (selection only; labels never scored)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--legacy-chunk-arm", action="store_true",
                    help="reproduce the 2026-08 per-chunk text scorer (parity check), ASR windows, no packing")
    ap.add_argument("--method-name", default=None)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    out_dir = ROOT / "runs" / args.exp_id / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out_dir / "run.pid").write_text(str(os.getpid()))
    tag = args.method_name or ("spvl_" + "_".join([
        f"f{args.frames}", "noctx" if args.no_transcript_context else "ctx",
        args.windows + (f"{args.window_seconds:g}" if args.windows == "fixed" else ""), args.mask]))
    if args.legacy_chunk_arm:
        tag = args.method_name or "legacy_chunk_replica"
    args.method_name = tag
    cfg = {k: v for k, v in vars(args).items()}
    cfg.update({"model": MODEL, "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS, "fps": FPS,
                "fill_uncovered": FILL_UNCOVERED, "code_path": CODE_PATH, "date": time.strftime("%Y-%m-%d"),
                "host": socket.gethostname(), "rules": "YOUTUBE_RULES", "reader": "prag",
                "video_question": VIDEO_QUESTION})
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    rows = load_manifest(args.manifest, args.datasets)
    if args.only_within_defined:
        keep = within_defined_ids(args.datasets)
        rows = [r for r in rows if (r["dataset"], r["video_id"]) in keep]
    if args.limit:
        rows = rows[:args.limit]
    asr = {ds: load_asr(ds) for ds in args.datasets}
    logging.info("videos %d  config %s", len(rows), tag)

    judge = Judge(mask_kind=args.mask_kind)
    pred_path = out_dir / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        for line in open(pred_path):
            r = json.loads(line)
            done.add((r["dataset"], r["video_id"]))
    n_err = 0
    t0 = time.time()
    with open(pred_path, "a") as fh:
        for n, row in enumerate(rows):
            key = (row["dataset"], row["video_id"])
            if key in done:
                continue
            segments = asr[row["dataset"]].get(row["video_id"], [])
            try:
                if args.legacy_chunk_arm:
                    zl = judge.legacy_chunk_scores([t for _, _, t in segments]) if segments else []
                    dur = float(row["duration"]); L = int(math.ceil(dur * FPS))
                    curve = np.full(L, FILL_UNCOVERED)
                    for (s, e, _), z in zip(segments, zl):
                        i0, i1 = max(0, int(math.floor(s * FPS))), min(L, int(math.ceil(e * FPS)))
                        if i1 > i0:
                            curve[i0:i1] = np.maximum(curve[i0:i1], z)
                    pred = {"schema_version": 1, "method": tag, "dataset": row["dataset"], "video_id": row["video_id"],
                            "duration": dur, "native_rate": FPS, "score_curve": [float(x) for x in curve],
                            "intervals": [], "error": None, "calls": max(1, math.ceil(len(segments) / 8)),
                            "seed": SEED, "code_path": CODE_PATH,
                            "extra": {"z_video": None, "windows": [{"i": i, "start": s, "end": e, "z": z}
                                                                   for i, ((s, e, _), z) in enumerate(zip(segments, zl))]}}
                    info = {}
                else:
                    pred, info = score_video(judge, row, segments, args, verify=(args.verify_only or n == 0))
                if pred is None:
                    raise RuntimeError(info.get("error", "unknown"))
                if "verify" in info:
                    logging.info("VERIFY %s %s", row["video_id"], json.dumps(info["verify"]))
                    (out_dir / "verify.json").write_text(json.dumps({"video_id": row["video_id"], **info}, indent=2))
                    if args.verify_only:
                        logging.info("DONE verify-only")
                        return
                fh.write(json.dumps(pred, ensure_ascii=False) + "\n")
                fh.flush()
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                n_err += 1
                logging.error("OOM %s %s", row["video_id"], str(exc)[:200])
                fh.write(json.dumps({"method": tag, "dataset": row["dataset"], "video_id": row["video_id"],
                                     "score_curve": [], "intervals": [], "error": "OOM"}) + "\n")
            except Exception as exc:  # keep going, record the error
                n_err += 1
                logging.exception("FAILED %s: %s", row["video_id"], exc)
                fh.write(json.dumps({"method": tag, "dataset": row["dataset"], "video_id": row["video_id"],
                                     "score_curve": [], "intervals": [], "error": f"{type(exc).__name__}: {exc}"}) + "\n")
            if (n + 1) % 10 == 0:
                el = time.time() - t0
                logging.info("progress %d/%d  %.1fs/video  errors %d", n + 1, len(rows), el / (n + 1), n_err)
    logging.info("DONE videos=%d errors=%d elapsed=%.0fs", len(rows), n_err, time.time() - t0)


if __name__ == "__main__":
    main()
