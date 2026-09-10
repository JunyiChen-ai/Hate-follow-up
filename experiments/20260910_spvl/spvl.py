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
import copy
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
MODEL = "Qwen/Qwen3-VL-8B-Instruct"  # default; --model overrides (family/size robustness study, README §11)
# Per-family processor kwargs so that every frame costs a fixed, documented number of tokens.
# Qwen: pixel cap (<= 91 tokens/frame); InternVL: one 448 tile (256); Gemma 3: 896 without pan-and-scan (256);
# LLaVA-OneVision: base 384 only, no AnyRes grid (recorded from the run; see README §11).
FAMILY_IMAGE_KW = {
    "qwen3_vl": {"size": {"shortest_edge": 65536, "longest_edge": 100352}},
    "qwen2_5_vl": {"size": {"shortest_edge": 65536, "longest_edge": 100352}},
    "internvl": {"crop_to_patches": False},
    "gemma3": {"do_pan_and_scan": False},
    "llava_onevision": {},
}
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


def window_question(i, n, t1, t2, text, kind="joint", style="rules"):
    """kind: joint | visual | speech; style: rules (does it violate) | evidence (is it where the violation occurs)."""
    body = text.strip() if text and text.strip() else "(no speech)"
    head = f"Consider only window {i + 1} of {n}, from {t1:.1f}s to {t2:.1f}s of this video."
    if kind == "joint":
        ctx = f"{head} Transcript in this window: {body}\n\n"
        q = ("Does THIS window contain content that violates any of the above rules?" if style == "rules" else
             "Is THIS window one of the segments where the content that violates the above rules occurs?")
    elif kind == "visual":
        ctx = (f"{head} Look only at the frames whose timestamps fall inside this window and judge the visual "
               f"content alone (imagery, gestures, symbols, on-screen text), ignoring the speech.\n\n")
        q = ("Does the visual content of THIS window violate any of the above rules?" if style == "rules" else
             "Is THIS window one of the segments where visual content that violates the above rules occurs?")
    elif kind == "speech":
        ctx = f"{head} Judge only what is spoken in this window: {body}\n\n"
        q = ("Does the speech in THIS window violate any of the above rules?" if style == "rules" else
             "Is THIS window one of the segments where speech that violates the above rules occurs?")
    else:
        raise ValueError(kind)
    return ctx + q + '\n\nAnswer "Yes" or "No".'


# ------------------------------------------------------------------ data
def load_manifest(path, datasets):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if r["dataset"] in datasets]


def load_asr(dataset):
    path = ROOT / f"data/asr_whisper_large_v3/{dataset}/timestamped_chunks.jsonl"
    out = {}
    for line in open(path):
        r = json.loads(line)
        segs = [(float(c["start"]), float(c["end"]), (c.get("text") or ""))
                for c in (r.get("chunks") or []) if c.get("end") is not None and c.get("start") is not None]
        out[r["video_id"]] = [s for s in segs if s[1] > s[0]]
    return out


def frame_paths(dataset, vid, k, source="k20"):
    """source k20: K of the 20 uniform frames; source w8: every window-centre frame (k ignored)."""
    d = ROOT / f"data/frames_{source}/{dataset}/{vid}"
    files = sorted(d.glob("f*_t*.jpg"))
    if not files:
        return []
    if source == "k20" and k < len(files):
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
    return "\n".join(f"[{s:.1f}s-{e:.1f}s] {t.strip()}" for s, e, t in segments if t.strip())


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
    def __init__(self, mask_kind="bool", model_id=MODEL):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tok = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = self.model.device
        self.dtype = torch.bfloat16
        self.mask_kind = mask_kind
        self.family = self.model.config.model_type
        if self.family not in FAMILY_IMAGE_KW:
            raise SystemExit(f"model_type {self.family} not in FAMILY_IMAGE_KW; add its image kwargs first")
        self.img_kw = dict(FAMILY_IMAGE_KW[self.family])
        if self.family == "llava_onevision":  # single 384 grid cell (base + 1 patch): every frame costs the same
            self.processor.image_processor.image_grid_pinpoints = [[384, 384]]
            self.model.config.image_grid_pinpoints = [[384, 384]]  # the model derives patch counts from its config
        self.image_token_id = getattr(self.model.config, "image_token_id", None)
        # LLaVA-OneVision's template only renders list-typed content (string system / assistant text is dropped)
        # and renders all images before the text of a turn.
        self.list_content = self.family == "llava_onevision"
        self.forward_params = set(inspect.signature(self.model.model.forward).parameters)
        text_cfg = getattr(self.model.config, "text_config", self.model.config)
        self.softcap = getattr(text_cfg, "final_logit_softcapping", None)
        self.yes_ids, self.no_ids = self.binary_ids()
        self.yes_t = torch.tensor(self.yes_ids, device=self.device)
        self.no_t = torch.tensor(self.no_ids, device=self.device)
        # Templates that reject two consecutive user turns (Gemma 3): the question is appended to the prefix's
        # own user turn instead of opening a second one; the prefix string then ends before the end-of-turn.
        try:
            self.render([{"role": "user", "content": [{"type": "text", "text": "a"}]},
                         {"role": "user", "content": [{"type": "text", "text": "b"}]}], True)
            self.same_turn = False
        except Exception:
            self.same_turn = True
        logging.info("model %s family %s same_turn %s Yes ids %s No ids %s softcap %s", model_id, self.family,
                     self.same_turn, self.yes_ids, self.no_ids, self.softcap)
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
        W = self.model.get_output_embeddings().weight
        ids = torch.cat([self.yes_t, self.no_t])
        lg = hidden_rows.float() @ W[ids].float().T  # [n, |yes|+|no|]
        if self.softcap:  # Gemma-style final logit softcapping (monotone; applied as in the model's own forward)
            lg = torch.tanh(lg / self.softcap) * self.softcap
        ny = len(self.yes_ids)
        return (torch.logsumexp(lg[:, :ny], 1) - torch.logsumexp(lg[:, ny:], 1)).tolist()

    # ---- prompt construction
    def prefix_messages(self, frames, segments, with_context, with_frames):
        content = []
        intro = "Observe this video."
        if with_frames and frames:
            if self.list_content:  # images are rendered before the turn's text: give the timestamps as a list
                intro += (f" The {len(frames)} images above are frames sampled from the video, in order; "
                          "frame k was taken at: " + ", ".join(f"frame {k + 1} at t={t:.1f}s" for k, (t, _) in enumerate(frames)) + ".")
            else:
                intro += f" Below are {len(frames)} frames sampled from the video, each preceded by its timestamp."
        if with_context:
            intro += " The full transcript with timestamps follows the frames." if (with_frames and frames) \
                else " The full transcript with timestamps follows."
        content.append({"type": "text", "text": intro + "\n"})
        images = []
        if with_frames:
            for t, f in frames:
                if not self.list_content:
                    content.append({"type": "text", "text": f"[t={t:.1f}s]\n"})
                content.append({"type": "image"})
                images.append(f)
        tail = ""
        if with_context:
            tail += "\nTranscript:\n" + transcript_block(segments) + "\n"
        tail += ("\nBased on this platform's hate speech policy, the following types of content are prohibited:\n"
                 f"{YOUTUBE_RULES}\n\n{READER_PRAG}\n")
        content.append({"type": "text", "text": tail})
        msgs = [self.turn("system", SYSTEM_MESSAGE),
                {"role": "user", "content": content}]
        return msgs, images

    def render(self, msgs, add_generation_prompt):
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)

    SENTINEL = "@@SPVL_PREFIX_END@@"

    @staticmethod
    def _append_text(msgs, text):
        """Copy of msgs with `text` appended to the last user turn's final text element."""
        out = copy.deepcopy(msgs)
        content = out[-1]["content"]
        if isinstance(content, str):
            out[-1]["content"] = content + text
        elif content and content[-1].get("type") == "text":
            content[-1]["text"] += text
        else:
            content.append({"type": "text", "text": text})
        return out

    def turn(self, role, text):
        return {"role": role, "content": [{"type": "text", "text": text}] if self.list_content else text}

    def conv(self, msgs, question, history=None):
        """Message list for asking `question` after the prefix (and optional earlier turns `history`)."""
        history = list(history or [])
        if not self.same_turn:
            return msgs + history + [{"role": "user", "content": [{"type": "text", "text": question}]}]
        if history:  # [user Q0, assistant A] -> Q0 merged into the prefix turn, then A, then the question
            q0 = history[0]["content"][0]["text"] if isinstance(history[0]["content"], list) else history[0]["content"]
            return self._append_text(msgs, "\n\n" + q0) + history[1:] + \
                [{"role": "user", "content": [{"type": "text", "text": question}]}]
        return self._append_text(msgs, "\n\n" + question)

    def encode_prefix(self, msgs, image_files):
        from PIL import Image
        if self.same_turn:
            full = self.render(self._append_text(msgs, self.SENTINEL), add_generation_prompt=False)
            # open user turn; each branch closes it. Trailing whitespace moves into the branch so the
            # tokenizer's multi-newline tokens never straddle the seam (token seam check would fail otherwise).
            text = full[:full.index(self.SENTINEL)].rstrip()
        else:
            text = self.render(msgs, add_generation_prompt=False)
        images = [Image.open(p).convert("RGB") for p in image_files]
        enc = self.encode(text, images)
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
        elif images and self.image_token_id is not None:
            n_img = int((enc["input_ids"][0] == self.image_token_id).sum())
            self.img_tokens = [n_img // len(images)] * len(images)
        else:
            self.img_tokens = []
        self._prefix_text = text
        return text, enc

    def encode(self, text, images):
        # add_special_tokens=False: the chat template already carries BOS where the family uses one (Gemma)
        if images:
            imgs = [images] if self.family == "llava_onevision" else images  # nested = one multi-image sample
            return self.processor(text=[text], images=imgs, return_tensors="pt", add_special_tokens=False, **self.img_kw)
        return self.processor(text=[text], return_tensors="pt", add_special_tokens=False)

    def model_inputs(self, enc):
        """Tensors from a processor output that the model forward accepts, on device (generic across families)."""
        kw = {}
        for k, v in enc.items():
            if k in self.forward_params and torch.is_tensor(v):
                kw[k] = v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device)
        if "mm_token_type_ids" in self.forward_params and "mm_token_type_ids" not in kw and "pixel_values" in kw:
            kw["mm_token_type_ids"] = self.mm_types(enc, enc["input_ids"])
        return kw

    def branch_ids(self, question, msgs=None):
        """User turn + assistant header as rendered by the model's own chat template (suffix after the prefix)."""
        if msgs is None:  # Qwen ChatML (the 2026-09-10 runs)
            text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        else:
            full = self.render(self.conv(msgs, question), add_generation_prompt=True)
            if not full.startswith(self._prefix_text):
                raise AssertionError("chat template does not extend the prefix string")
            text = full[len(self._prefix_text):]
        return self.tok(text, add_special_tokens=False)["input_ids"], text

    def answer_ids(self, msgs, question, answer):
        """Text the template adds for a completed assistant turn (answer + end-of-turn), after Q + header."""
        head = self.render(self.conv(msgs, question), add_generation_prompt=True)
        full = self.render(self.conv(msgs, question) + [self.turn("assistant", answer)],
                           add_generation_prompt=False)
        if not full.startswith(head):
            raise AssertionError("assistant turn does not extend the question rendering")
        text = full[len(head):]
        return self.tok(text, add_special_tokens=False)["input_ids"], text

    def seam_check_text(self, msgs, prefix_text, question, branch_text):
        full = self.render(self.conv(msgs, question), add_generation_prompt=True)
        if full != prefix_text + branch_text:
            raise AssertionError("string seam mismatch")

    def seam_check_tokens(self, msgs, image_files, prefix_ids, question, bids):
        from PIL import Image
        full = self.render(self.conv(msgs, question), add_generation_prompt=True)
        images = [Image.open(p).convert("RGB") for p in image_files]
        enc = self.encode(full, images)
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

    def packed_positions(self, pos_p, branch_lens, sequential=False, ext_len=0):
        """[3,1,T] positions: prefix (mrope), optional text extension continuing sequentially,
        then each branch restarting at the same offset (block) or continuing (sequential)."""
        s = int(pos_p.max().item()) + 1
        parts = [pos_p]
        if ext_len:
            parts.append(torch.arange(s, s + ext_len, device=self.device)[None, None].expand(3, 1, -1))
            s += ext_len
        off = s
        for n in branch_lens:
            start = off if sequential else s
            parts.append(torch.arange(start, start + n, device=self.device)[None, None].expand(3, 1, -1))
            if sequential:
                off += n
        return torch.cat(parts, dim=-1).contiguous()

    # ---- forward
    @torch.no_grad()
    def packed_forward(self, enc, pos_p, branches, arm="block", ext_ids=None):
        """One forward over [prefix (+ext), branch_1..branch_N]; Yes/No log-odds at each branch end."""
        ext_ids = ext_ids or []
        prefix_ids = enc["input_ids"][0].tolist() + list(ext_ids)
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
            pos = self.packed_positions(pos_p, lens, sequential=False, ext_len=len(ext_ids))
        elif arm == "causal":
            allow = causal_allow(total)
            pos = self.packed_positions(pos_p, lens, sequential=True, ext_len=len(ext_ids))
        else:
            raise ValueError(arm)
        mask = to_mask(allow, self.dtype, self.device, self.mask_kind)
        kw = {"input_ids": torch.tensor([ids], device=self.device),
              "attention_mask": mask, "position_ids": pos, "use_cache": False}
        if "pixel_values" in enc:
            kw["pixel_values"] = enc["pixel_values"].to(self.device, self.dtype)
            kw["image_grid_thw"] = enc["image_grid_thw"].to(self.device)
        keep = torch.tensor(ends, device=self.device)
        out = self.model.model(**kw)
        hidden = out.last_hidden_state[0, keep]  # [N, H]
        z = self.margins_fp32(hidden)
        del out, hidden, mask
        return z, total

    @torch.no_grad()
    def plain_forward(self, msgs, image_files, question, history=None):
        """Reference: one ordinary call (no custom mask / positions / cache). history = extra turns before the question."""
        from PIL import Image
        full = self.render(self.conv(msgs, question, history), add_generation_prompt=True)
        images = [Image.open(p).convert("RGB") for p in image_files]
        enc = self.encode(full, images)
        for im in images:
            im.close()
        kw = self.model_inputs(enc)
        out = self.model.model(**kw, use_cache=False)
        hidden = out.last_hidden_state[0, -1:]
        z = self.margins_fp32(hidden)[0]
        del out
        return z, None, int(enc["input_ids"].shape[1])

    # ---- cache path (family-agnostic isolation: prefix KV cache + one short forward per branch)
    @torch.no_grad()
    def prefix_cache(self, enc):
        """Prefix forward with use_cache; returns the DynamicCache (prefix length = input_ids length)."""
        kw = self.model_inputs(enc)
        out = self.model.model(**kw, use_cache=True)
        cache = out.past_key_values
        del out
        return cache

    @torch.no_grad()
    def extend_cache(self, cache, ids):
        """Append text tokens to the cache in place (used for the model's own verdict turn)."""
        # no attention_mask: no padding anywhere, and Qwen-VL derives branch positions from the cache length
        # + rope_deltas only when the mask is absent (a full-length mask would yield prefix-length positions)
        kw = {"input_ids": torch.tensor([ids], device=self.device), "past_key_values": cache, "use_cache": True}
        out = self.model.model(**kw)
        del out
        return cache

    @torch.no_grad()
    def cached_branch(self, cache, ids):
        """Yes/No log-odds at the end of `ids` given the prefix cache; the cache is deep-copied, so branches
        never see each other (exactly one independent call per branch)."""
        c = copy.deepcopy(cache)
        kw = {"input_ids": torch.tensor([ids], device=self.device), "past_key_values": c, "use_cache": True}
        out = self.model.model(**kw)
        z = self.margins_fp32(out.last_hidden_state[0, -1:])[0]
        del out, c
        return z

    @torch.no_grad()
    def legacy_chunk_scores(self, texts, batch=8):
        """2026-08 per-chunk prompt (no system message, single Yes/No ids), batched calls."""
        yes = self.tok.encode("Yes", add_special_tokens=False)[0]
        no = self.tok.encode("No", add_special_tokens=False)[0]
        self.tok.padding_side = "left"  # last position must be the prompt end for every row
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
def run_groups(judge, enc, pos_p, branches, args, ext_ids=None):
    """Forward all branches, splitting into groups so that prefix + group <= max_tokens."""
    P = enc["input_ids"].shape[1] + len(ext_ids or [])
    groups, cur, cur_len = [], [], 0
    for i, b in enumerate(branches):
        if cur and P + cur_len + len(b) > args.max_tokens:
            groups.append(cur); cur, cur_len = [], 0
        cur.append(i); cur_len += len(b)
    if cur:
        groups.append(cur)
    zs, total = [None] * len(branches), 0
    for g in groups:
        z, tot = judge.packed_forward(enc, pos_p, [branches[i] for i in g], arm=args.mask, ext_ids=ext_ids)
        total += tot
        for i, v in zip(g, z):
            zs[i] = v
    return zs, total, len(groups)


def score_video(judge, row, segments, args, verify=False):
    vid, ds, dur = row["video_id"], row["dataset"], float(row["duration"])
    frames = frame_paths(ds, vid, args.frames, args.frame_source) if args.frames > 0 else []
    if args.frames > 0 and not frames:
        return None, {"error": "no frames cached"}
    with_context = not args.no_transcript_context
    msgs, image_files = judge.prefix_messages(frames, segments, with_context, args.frames > 0)
    prefix_text, enc = judge.encode_prefix(msgs, image_files)
    prefix_ids = enc["input_ids"][0].tolist()
    # windows
    if args.windows == "fixed":
        wins = fixed_windows(dur, args.window_seconds)
        wtexts = [window_text(segments, a, b) for a, b in wins]
    elif args.windows == "asr_split":  # ASR segments cut into <= S-second pieces; silent gaps stay unscored
        wins, wtexts = [], []
        for s0, e0, t in segments:
            n = max(1, int(math.ceil((e0 - s0) / args.window_seconds - 1e-9)))
            for j in range(n):
                a_, b_ = s0 + j * (e0 - s0) / n, s0 + (j + 1) * (e0 - s0) / n
                wins.append((a_, b_)); wtexts.append(window_text([(s0, e0, t)], a_, b_))
    else:  # asr
        wins = [(s, e) for s, e, _ in segments]
        wtexts = [t for _, _, t in segments]
    # branch specs: (window index, kind, question)
    kinds = {"joint": ["joint"], "dual": ["visual", "speech"], "triple": ["joint", "visual", "speech"]}[args.branches]
    specs = []
    for i, ((a, b), t) in enumerate(zip(wins, wtexts)):
        for kind in kinds:
            if kind == "speech" and not (t and t.strip()):
                continue  # no speech in this window: no speech branch
            if kind == "visual" and args.frames == 0:
                continue
            specs.append((i, kind, window_question(i, len(wins), a, b, t, kind=kind, style=args.window_question)))
    if not specs:  # e.g. dual without frames and no speech anywhere
        specs = [(i, "joint", window_question(i, len(wins), a, b, t)) for i, ((a, b), t) in enumerate(zip(wins, wtexts))]
    # asr mode on a video without transcript: no windows at all -> whole-video score only, curve stays FILL_UNCOVERED
    tmpl = msgs if args.isolation == "cache" else None  # cache path: branch text from the model's own template
    b0_ids, b0_text = judge.branch_ids(VIDEO_QUESTION, tmpl)
    judge.seam_check_text(msgs, prefix_text, VIDEO_QUESTION, b0_text)
    branches, btexts = [], []
    for _, _, q in specs:
        bids, btext = judge.branch_ids(q, tmpl)
        branches.append(bids); btexts.append(btext)
    if verify:
        judge.seam_check_tokens(msgs, image_files, prefix_ids, VIDEO_QUESTION, b0_ids)
        for (_, _, q), bids in list(zip(specs, branches))[:2]:
            judge.seam_check_tokens(msgs, image_files, prefix_ids, q, bids)
    P = len(prefix_ids)
    info = {"prefix_tokens": P, "n_windows": len(wins), "n_branches": len(specs), "img_tokens": judge.img_tokens[:1]}
    if args.isolation == "cache":
        cache = judge.prefix_cache(enc)
        z_video = judge.cached_branch(cache, b0_ids)
        history = []
        if args.stance == "none":
            stance_answer, ext_ids, ngroups = None, [], 1
        else:
            stance_answer = "Yes" if z_video > 0 else "No"
            ans_ids, ans_text = judge.answer_ids(msgs, VIDEO_QUESTION, stance_answer)
            ext_ids = b0_ids + ans_ids
            history = [{"role": "user", "content": [{"type": "text", "text": VIDEO_QUESTION}]},
                       judge.turn("assistant", stance_answer)]
            full = judge.render(judge.conv(msgs, specs[0][2], history), add_generation_prompt=True) if specs else None
            if not specs:
                pass
            elif judge.same_turn:  # window questions open a new user turn after the answer: re-derive their text
                head = prefix_text + b0_text + ans_text
                if not full.startswith(head):
                    raise AssertionError("stance seam mismatch (same-turn template)")
                branches, btexts = [], []
                for _, _, q in specs:
                    t = judge.render(judge.conv(msgs, q, history), add_generation_prompt=True)[len(head):]
                    branches.append(judge.tok(t, add_special_tokens=False)["input_ids"]); btexts.append(t)
            elif full != prefix_text + b0_text + ans_text + btexts[0]:
                raise AssertionError("stance seam mismatch: chat template renders the assistant turn differently")
            judge.extend_cache(cache, ext_ids)
            ngroups = 2
        zb = [judge.cached_branch(cache, b) for b in branches]
        total = P + len(ext_ids) + len(b0_ids) + sum(len(b) for b in branches)
        if verify:
            checks = {}
            z_ref, _, _ = judge.plain_forward(msgs, image_files, VIDEO_QUESTION)
            checks["video_q_cache_vs_plain_dz"] = abs(z_video - z_ref)
            n = min(5, len(zb))
            seq = [judge.plain_forward(msgs, image_files, specs[i][2], history=history)[0] for i in range(n)]
            checks["windows_cache_vs_plain_max_dz"] = max(abs(a - b) for a, b in zip(zb[:n], seq)) if n else 0.0
            if n >= 3:
                from scipy.stats import spearmanr
                checks["windows_cache_vs_plain_spearman"] = float(spearmanr(zb[:n], seq).correlation)
            checks["peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
            checks["img_tokens_per_frame"] = judge.img_tokens[:1]
            info["verify"] = checks
            # gate: a position / mask error shifts branches by several nats; bf16 kernel noise stays well below 1
            if max(checks["video_q_cache_vs_plain_dz"], checks["windows_cache_vs_plain_max_dz"]) >= 1.0 or \
                    checks.get("windows_cache_vs_plain_spearman", 1.0) < 0.9:
                raise SystemExit(f"VERIFY GATE FAILED: {json.dumps(checks)}")
        del cache
        verify = False  # mask-path verify blocks below are skipped
    pos_p = judge.prefix_positions(enc) if args.isolation == "mask" else None
    if args.isolation == "cache":
        pass
    elif args.stance == "none":
        zs, total, ngroups = run_groups(judge, enc, pos_p, [b0_ids] + branches, args)
        z_video, zb = zs[0], zs[1:]
        stance_answer = None
    else:  # verdict: stage 1 = whole-video answer; stage 2 = windows conditioned on the model's own answer
        zs, total1, _ = run_groups(judge, enc, pos_p, [b0_ids], args)
        z_video = zs[0]
        stance_answer = "Yes" if z_video > 0 else "No"
        ans_text = f"{stance_answer}<|im_end|>\n"
        ext_ids = b0_ids + judge.tok(ans_text, add_special_tokens=False)["input_ids"]
        # string seam: prefix + Q0 + answer + q_i must equal the chat template of the 4-turn conversation
        full = judge.render(msgs + [{"role": "user", "content": [{"type": "text", "text": VIDEO_QUESTION}]},
                                   {"role": "assistant", "content": stance_answer},
                                   {"role": "user", "content": [{"type": "text", "text": specs[0][2]}]}],
                            add_generation_prompt=True) if specs else None
        if specs and full != prefix_text + b0_text + ans_text + btexts[0]:
            raise AssertionError("stance seam mismatch: chat template renders the assistant turn differently")
        zb, total2, ngroups = run_groups(judge, enc, pos_p, branches, args, ext_ids=ext_ids)
        total = total1 + total2
        ngroups += 1
    info.update({"total_tokens": total, "n_groups": ngroups, "stance_answer": stance_answer})
    # aggregate branch scores per window
    per = [dict() for _ in wins]
    for (i, kind, _), z in zip(specs, zb):
        per[i][kind] = z
    z_win = []
    for d in per:
        vals = [v for v in d.values()]
        z_win.append(max(vals) if vals else FILL_UNCOVERED)
    if verify and args.stance == "none":
        checks = {}
        z_ref, _, _ = judge.plain_forward(msgs, image_files, VIDEO_QUESTION)
        z_pk, _ = judge.packed_forward(enc, pos_p, [b0_ids], arm="block")
        checks["video_q_packed_vs_plain_dz"] = abs(z_pk[0] - z_ref)
        n = min(5, len(zb))
        seq = [judge.plain_forward(msgs, image_files, specs[i][2])[0] for i in range(n)]
        pk, _ = judge.packed_forward(enc, pos_p, branches[:n], arm="block")
        checks["windows_packed_vs_plain_max_dz"] = max(abs(a - b) for a, b in zip(pk, seq)) if n else 0.0
        if n >= 3:
            from scipy.stats import spearmanr
            checks["windows_packed_vs_plain_spearman"] = float(spearmanr(pk, seq).correlation)
        checks["peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
        info["verify"] = checks
    elif verify:
        info["verify"] = {"note": "stance mode: seam asserted; packing equivalence verified in stance=none runs",
                          "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
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
            "intervals": [], "error": None, "calls": info["n_groups"], "seed": SEED, "code_path": CODE_PATH,
            "extra": {"z_video": z_video, "stance_answer": stance_answer,
                      "windows": [{"i": i, "start": a, "end": b, "z": z, **{f"z_{k}": v for k, v in per[i].items()}}
                                  for i, ((a, b), z) in enumerate(zip(wins, z_win))],
                      **{k: v for k, v in info.items() if k != "verify"}}}
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
    ap.add_argument("--windows", choices=["fixed", "asr", "asr_split"], default="fixed")
    ap.add_argument("--window-seconds", type=float, default=8.0)
    ap.add_argument("--mask", choices=["block", "causal"], default="block")
    ap.add_argument("--stance", choices=["none", "verdict"], default="none",
                    help="verdict: windows are judged after the model's own whole-video answer (2 forwards)")
    ap.add_argument("--branches", choices=["joint", "dual", "triple"], default="joint",
                    help="dual: visual + speech branch per window (max); triple: joint + visual + speech")
    ap.add_argument("--window-question", choices=["rules", "evidence"], default="rules")
    ap.add_argument("--frame-source", choices=["k20", "w8"], default="k20",
                    help="k20: K uniform frames; w8: one frame per 8 s window (data/frames_w8)")
    ap.add_argument("--mask-kind", choices=["bool", "additive"], default="additive")
    ap.add_argument("--max-tokens", type=int, default=9000)
    ap.add_argument("--only-within-defined", action="store_true",
                    help="pilot subset: videos whose GT has both classes (selection only; labels never scored)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--legacy-chunk-arm", action="store_true",
                    help="reproduce the 2026-08 per-chunk text scorer (parity check), ASR windows, no packing")
    ap.add_argument("--method-name", default=None)
    ap.add_argument("--model", default=MODEL, help="HF id of the MLLM (README §11 family/size study)")
    ap.add_argument("--model-tag", default=None,
                    help="short model name; runs go to runs/<exp_id>/mllm/<tag>/<run_name> and prefix the method tag")
    ap.add_argument("--isolation", choices=["mask", "cache"], default="mask",
                    help="mask: packed forward with a block mask (Qwen3-VL); cache: prefix KV cache + one forward per branch (any family)")
    args = ap.parse_args()
    if args.isolation == "mask" and args.model != MODEL:
        raise SystemExit("--isolation mask is only validated for Qwen3-VL-8B; use --isolation cache for other models")

    torch.manual_seed(SEED)
    out_dir = ROOT / "runs" / args.exp_id / (f"mllm/{args.model_tag}/{args.run_name}" if args.model_tag else args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out_dir / "run.log"), logging.StreamHandler(sys.stdout)])
    logging.info("host %s", socket.gethostname())
    (out_dir / "run.pid").write_text(str(os.getpid()))
    tag = args.method_name or ("spvl_" + "_".join([
        f"f{args.frames}", "noctx" if args.no_transcript_context else "ctx",
        args.windows + (f"{args.window_seconds:g}" if args.windows == "fixed" else ""), args.mask,
        f"st{args.stance}", f"br{args.branches}", f"q{args.window_question}", f"fs{args.frame_source}"]))
    if args.legacy_chunk_arm:
        tag = args.method_name or "legacy_chunk_replica"
    if args.isolation == "cache" and not args.method_name:
        tag += "_isocache"
    if args.model_tag and not args.method_name:
        tag = f"{args.model_tag}__{tag}"
    args.method_name = tag
    cfg = {k: v for k, v in vars(args).items()}
    cfg.update({"model": args.model, "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS, "fps": FPS,
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

    judge = Judge(mask_kind=args.mask_kind, model_id=args.model)
    logging.info("model %s isolation %s", args.model, args.isolation)
    import transformers
    cfg.update({"same_turn": judge.same_turn, "list_content": judge.list_content, "img_kw": judge.img_kw,
                "family": judge.family, "transformers": transformers.__version__, "torch": torch.__version__})
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    pred_path = out_dir / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        for line in open(pred_path):
            r = json.loads(line)
            if not r.get("error"):  # error rows are retried; the evaluator keeps the last row per video
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
