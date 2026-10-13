"""Qwen3-Omni Thinker as the r6 reader (candidate 43): the same conversation layout as `src/mllm_judge.Judge`
(system, 20 frames with time labels, transcript, rules, verdict question + the model's own stance), Yes/No log-odds
read at the answer position, one prefix KV cache per video, independent window branches restored with `crop`.
New against the Qwen3-VL judge: a branch may carry images and / or one audio clip next to the question, and every
suffix gets explicit 3-D rope positions (the thinker only derives them itself when an attention mask is given and
the cache is empty). Text-only output; the talker is never loaded.
"""
import copy
import inspect
import logging
import time

import numpy as np
import torch
from PIL import Image

from src.mllm_judge import MAX_PIXELS, MIN_PIXELS, READER_PRAG, SYSTEM_MESSAGE, VIDEO_QUESTION, YOUTUBE_RULES
from src.video_inputs import transcript_block

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SR = 16000


def tick(j):
    if j.device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


class OmniJudge:
    def __init__(self, model_id=MODEL):
        from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration
        self.model_id = model_id
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
        self.tok = self.processor.tokenizer
        self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = self.model.device
        self.dtype = torch.bfloat16
        cfg = self.model.config
        self.image_token_id = cfg.image_token_id
        self.audio_token_id = cfg.audio_token_id
        self.img_kw = {"size": {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}}
        self.forward_params = set(inspect.signature(self.model.forward).parameters)
        self.softcap = None
        self.yes_ids, self.no_ids = self.label_ids("Yes"), self.label_ids("No")
        assert not (set(self.yes_ids) & set(self.no_ids))
        self.forward_calls = 0
        self.audio_calls = 0
        self.vision_calls = 0
        self._hidden = None
        self.model.lm_head.register_forward_pre_hook(lambda _m, args: setattr(self, "_hidden", args[0]))
        self.model.audio_tower.register_forward_pre_hook(lambda *_: setattr(self, "audio_calls", self.audio_calls + 1))
        self.model.visual.register_forward_pre_hook(lambda *_: setattr(self, "vision_calls", self.vision_calls + 1))
        self.model.model.register_forward_pre_hook(lambda *_: setattr(self, "forward_calls", self.forward_calls + 1))
        try:  # two consecutive user turns render? (the Qwen templates do)
            self.render([{"role": "user", "content": [{"type": "text", "text": "a"}]},
                         {"role": "user", "content": [{"type": "text", "text": "b"}]}], True)
            self.same_turn = False
        except Exception:  # noqa: BLE001
            self.same_turn = True
        assert not self.same_turn, "the Omni chat template must accept consecutive user turns"
        self.img_tokens = []
        logging.info("model %s Yes %s No %s image_token %s audio_token %s", model_id, self.yes_ids, self.no_ids,
                     self.image_token_id, self.audio_token_id)

    # ---- token sets and read-out
    def label_ids(self, label):
        s = set()
        for v in (label, f" {label}", label.lower(), f" {label.lower()}", label.upper(), f" {label.upper()}"):
            ids = self.tok.encode(v, add_special_tokens=False)
            if ids:
                s.add(ids[0])
        return sorted(s)

    def margins_fp32(self, hidden_rows):
        W = self.model.lm_head.weight
        ids = torch.tensor(self.yes_ids + self.no_ids, device=W.device)
        lg = hidden_rows.float() @ W[ids].float().T
        ny = len(self.yes_ids)
        return (torch.logsumexp(lg[:, :ny], 1) - torch.logsumexp(lg[:, ny:], 1)).tolist()

    # ---- prompt construction (same strings as src/mllm_judge.Judge.prefix_messages)
    @staticmethod
    def turn(role, text):
        return {"role": role, "content": text}

    def prefix_messages(self, frames, segments):
        content = []
        intro = ("Observe this video." + f" Below are {len(frames)} frames sampled from the video, each preceded by its timestamp."
                 + " The full transcript with timestamps follows the frames.")
        content.append({"type": "text", "text": intro + "\n"})
        images = []
        for t, f in frames:
            content.append({"type": "text", "text": f"[t={t:.1f}s]\n"})
            content.append({"type": "image"})
            images.append(f)
        tail = ("\nTranscript:\n" + transcript_block(segments) + "\n"
                + "\nBased on this platform's hate speech policy, the following types of content are prohibited:\n"
                + f"{YOUTUBE_RULES}\n\n{READER_PRAG}\n")
        content.append({"type": "text", "text": tail})
        return [self.turn("system", SYSTEM_MESSAGE), {"role": "user", "content": content}], images

    def render(self, msgs, add_generation_prompt):
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)

    def conv(self, msgs, content, history=None):
        """Messages for one more user turn whose content is a list of parts (text / image / audio)."""
        return msgs + list(history or []) + [{"role": "user", "content": content}]

    def encode(self, text, images=(), audios=()):
        kw = dict(text=[text], return_tensors="pt", add_special_tokens=False, use_audio_in_video=False, **self.img_kw)
        if images:
            kw["images"] = list(images)
        if audios:
            kw["audio"] = [np.asarray(a, dtype=np.float32) for a in audios]
            kw["sampling_rate"] = SR
        enc = self.processor(**kw)
        if "image_grid_thw" in enc:
            grid = enc["image_grid_thw"]
            merge = self.processor.image_processor.merge_size
            patch = self.processor.image_processor.patch_size
            px = [int(h * w) * patch * patch for _, h, w in grid.tolist()]
            if max(px) > MAX_PIXELS:
                raise SystemExit(f"pixel cap not honoured: {max(px)} > {MAX_PIXELS}")
            enc["image_counts"] = [int(t * h * w) // (merge ** 2) for t, h, w in grid.tolist()]
            assert int((enc["input_ids"][0] == self.image_token_id).sum()) == sum(enc["image_counts"])
        if audios:
            n_audio_tokens = int((enc["input_ids"][0] == self.audio_token_id).sum())
            enc["audio_tokens"] = n_audio_tokens
            assert n_audio_tokens > 0
        return enc

    @staticmethod
    def open_images(paths):
        images = []
        try:
            for p in paths:
                with Image.open(p) as original:
                    images.append(original.convert("RGB"))
            return images
        except Exception:
            for im in images:
                im.close()
            raise

    def encode_text_with_media(self, text, image_files=(), audios=()):
        images = self.open_images(image_files) if image_files else []
        try:
            return self.encode(text, images, audios)
        finally:
            for im in images:
                im.close()

    def model_inputs(self, enc):
        kw = {}
        for k, v in enc.items():
            if k in self.forward_params and torch.is_tensor(v):
                kw[k] = v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device)
        return kw

    # ---- rope positions for a suffix (relative to 0; the caller adds the logical start)
    def positions(self, enc):
        ids = enc["input_ids"].to(self.device)
        grids = enc["image_grid_thw"].to(self.device) if "image_grid_thw" in enc else None
        audio_lens = enc["feature_attention_mask"].sum(-1).to(self.device) if "feature_attention_mask" in enc else None
        pos, delta = self.model.get_rope_index(ids, grids, None, torch.ones_like(ids), False, audio_lens, None)
        return pos, delta

    # ---- forward paths
    @torch.no_grad()
    def forward_hidden(self, enc, position_ids=None, cache=None):
        """The thinker's own forward (audio tower, vision tower, placement, DeepStack inside); returns the last hidden row."""
        kw = self.model_inputs(enc)
        if cache is not None:
            kw.pop("attention_mask", None)      # the processor's mask covers the suffix only; the model builds the causal mask over cache + suffix
            kw["position_ids"] = position_ids
        kw["use_audio_in_video"] = False
        self._hidden = None
        out = self.model(**kw, past_key_values=cache, use_cache=cache is not None)
        h = self._hidden[0, -1].clone()
        del out
        return h

    @torch.no_grad()
    def prefix_cache(self, enc):
        """Prefix forward with the model's own rope index (empty cache + attention mask); returns the cache and the
        logical start of the next token."""
        self.model.rope_deltas = None
        kw = self.model_inputs(enc)
        kw["use_audio_in_video"] = False
        self._hidden = None
        out = self.model(**kw, past_key_values=None, use_cache=True)
        cache = out.past_key_values
        del out
        pos, delta = self.positions(enc)
        assert cache.get_seq_length() == enc["input_ids"].shape[1]
        assert torch.equal(delta, self.model.rope_deltas), "rope deltas of the prefix forward differ from get_rope_index"
        return cache, int(pos.max()) + 1

    @torch.no_grad()
    def text_step(self, cache, ids, logical_start):
        """Text-only continuation on `cache` (in place); returns the last hidden row and the next logical start."""
        n = len(ids)
        pos = (torch.arange(n, device=self.device) + logical_start).view(1, 1, n).expand(3, 1, n)
        self._hidden = None
        out = self.model(input_ids=torch.tensor([ids], device=self.device), position_ids=pos, past_key_values=cache,
                         use_cache=True, use_audio_in_video=False)
        h = self._hidden[0, -1].clone()
        del out
        return h, logical_start + n

    def branch_text_margin(self, cache, ids, logical_start):
        """Yes/No log-odds after text `ids`; the cache is restored afterwards."""
        n = cache.get_seq_length()
        try:
            h, _ = self.text_step(cache, ids, logical_start)
            return self.margins_fp32(h[None])[0]
        finally:
            cache.crop(n)
            assert cache.get_seq_length() == n, (cache.get_seq_length(), n)

    def branch_media_margin(self, cache, enc, logical_start):
        """Yes/No log-odds after a suffix with images / audio (`enc` from `encode`); the cache is restored."""
        n = cache.get_seq_length()
        rel, _ = self.positions(enc)
        try:
            h = self.forward_hidden(enc, rel + logical_start, cache)
            return self.margins_fp32(h[None])[0], int(rel.max()) + 1
        finally:
            cache.crop(n)
            assert cache.get_seq_length() == n, (cache.get_seq_length(), n)

    @torch.no_grad()
    def plain_margin(self, text, image_files=(), audios=()):
        """Reference: one ordinary full forward (no cache, the model's own rope index)."""
        enc = self.encode_text_with_media(text, image_files, audios)
        self.model.rope_deltas = None
        h = self.forward_hidden(enc)
        return self.margins_fp32(h[None])[0], enc["input_ids"][0].tolist()

    # ---- text helpers for the stance turn and text branches
    def branch_ids(self, head_text, msgs, content, history=None):
        full = self.render(self.conv(msgs, content, history), add_generation_prompt=True)
        if not full.startswith(head_text):
            raise AssertionError("chat template does not extend the current conversation string")
        text = full[len(head_text):]
        return self.tok(text, add_special_tokens=False)["input_ids"], text

    def answer_ids(self, msgs, content, answer, history=None):
        head = self.render(self.conv(msgs, content, history), add_generation_prompt=True)
        full = self.render(self.conv(msgs, content, history) + [self.turn("assistant", answer)], add_generation_prompt=False)
        if not full.startswith(head):
            raise AssertionError("chat template does not extend the assistant turn")
        text = full[len(head):]
        return self.tok(text, add_special_tokens=False)["input_ids"], text


@torch.no_grad()
def build(j, frames, segments):
    """Prefix (frames + transcript + rules) and the verdict turn with the model's own stance, as r6."""
    msgs, files = j.prefix_messages(frames, segments)
    text = j.render(msgs, add_generation_prompt=False)
    enc = j.encode_text_with_media(text, files)
    cache, start = j.prefix_cache(enc)
    prefix_tokens = int(enc["input_ids"].shape[1])
    qcontent = [{"type": "text", "text": VIDEO_QUESTION}]
    qid, qtext = j.branch_ids(text, msgs, qcontent)
    h, start2 = j.text_step(cache, qid, start)
    z = j.margins_fp32(h[None])[0]
    stance = "Yes" if z > 0 else "No"
    aid, atext = j.answer_ids(msgs, qcontent, stance)
    _, start3 = j.text_step(cache, aid, start2)
    history = [{"role": "user", "content": qcontent}, j.turn("assistant", stance)]
    assert cache.get_seq_length() == prefix_tokens + len(qid) + len(aid)
    return cache, dict(msgs=msgs, files=files, history=history, head=text + qtext + atext, logical_start=start3,
                       global_margin=z, stance=stance, stance_cache_tokens=cache.get_seq_length(), prefix_tokens=prefix_tokens,
                       image_counts=list(enc["image_counts"]), prefix_text=text)


@torch.no_grad()
def text_margin(j, cache, ctx, question):
    ids, _ = j.branch_ids(ctx["head"], ctx["msgs"], [{"type": "text", "text": question}], ctx["history"])
    return j.branch_text_margin(cache, ids, ctx["logical_start"]), dict(suffix_tokens=len(ids))


@torch.no_grad()
def media_margin(j, cache, ctx, parts, image_files=(), audios=()):
    """`parts`: content list of the user turn (text / image / audio items in order); images and audio clips in the
    same order as their items."""
    full = j.render(ctx["msgs"] + ctx["history"] + [{"role": "user", "content": parts}], True)
    assert full.startswith(ctx["head"])
    text = full[len(ctx["head"]):]
    enc = j.encode_text_with_media(text, image_files, audios)
    z, _ = j.branch_media_margin(cache, enc, ctx["logical_start"])
    d = dict(suffix_tokens=int(enc["input_ids"].shape[1]))
    if image_files:
        d["image_counts"] = list(enc["image_counts"])
    if audios:
        d["audio_tokens"] = int(enc["audio_tokens"])
        d["audio_samples"] = [int(len(a)) for a in audios]
    return z, d, text


def standalone(j, ctx, parts, image_files=(), audios=()):
    """The same conversation as one standalone forward (prefix images + branch media, no cache)."""
    full = j.render(ctx["msgs"] + ctx["history"] + [{"role": "user", "content": parts}], True)
    z, ids = j.plain_margin(full, list(ctx["files"]) + list(image_files), audios)
    return z, ids
