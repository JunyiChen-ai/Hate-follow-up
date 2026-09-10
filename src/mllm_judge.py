"""Frozen-MLLM judge with a prefix KV cache: independent branches, sequential branches, greedy generation,
multi-choice log-probability read-out. Family-agnostic (Qwen2.5/3-VL, InternVL-HF, Gemma 3, LLaVA-OneVision).

Promoted from the cache path of experiments/20260910_spvl/spvl.py (2026-09-11; the packed-mask path stays
there). The prompt material (system message, rules, reader instruction, whole-video question) is the frozen
2026-08 text and must not be edited.
"""
from __future__ import annotations

import copy
import inspect
import logging

import torch

from src.video_inputs import transcript_block

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MAX_PIXELS = 100352
MIN_PIXELS = 65536
FAMILY_IMAGE_KW = {
    "qwen3_vl": {"size": {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}},
    "qwen2_5_vl": {"size": {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}},
    "internvl": {"crop_to_patches": False},
    "gemma3": {"do_pan_and_scan": False},
    "llava_onevision": {},
}

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
VIDEO_QUESTION = ('Does this video contain content that violates any of the above rules?\n\n'
                  'Answer "Yes" or "No".')


class Judge:
    def __init__(self, model_id=MODEL):
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
        self.family = self.model.config.model_type
        if self.family not in FAMILY_IMAGE_KW:
            raise SystemExit(f"model_type {self.family} not in FAMILY_IMAGE_KW; add its image kwargs first")
        self.img_kw = dict(FAMILY_IMAGE_KW[self.family])
        if self.family == "llava_onevision":
            self.processor.image_processor.image_grid_pinpoints = [[384, 384]]
            self.model.config.image_grid_pinpoints = [[384, 384]]
        self.image_token_id = getattr(self.model.config, "image_token_id", None)
        self.list_content = self.family == "llava_onevision"
        self.forward_params = set(inspect.signature(self.model.model.forward).parameters)
        text_cfg = getattr(self.model.config, "text_config", self.model.config)
        self.softcap = getattr(text_cfg, "final_logit_softcapping", None)
        self.eos_ids = set()
        for t in (self.tok.eos_token_id, getattr(self.model.generation_config, "eos_token_id", None)):
            if isinstance(t, int):
                self.eos_ids.add(t)
            elif isinstance(t, (list, tuple)):
                self.eos_ids.update(int(x) for x in t)
        try:
            self.render([{"role": "user", "content": [{"type": "text", "text": "a"}]},
                         {"role": "user", "content": [{"type": "text", "text": "b"}]}], True)
            self.same_turn = False
        except Exception:
            self.same_turn = True
        self.loose_stance_seam = False
        self.yes_ids, self.no_ids = self.label_ids("Yes"), self.label_ids("No")
        logging.info("model %s family %s same_turn %s Yes %s No %s eos %s", model_id, self.family, self.same_turn,
                     self.yes_ids, self.no_ids, sorted(self.eos_ids))

    # ---- token sets
    def label_ids(self, label):
        s = set()
        for v in (label, f" {label}", label.lower(), f" {label.lower()}", label.upper(), f" {label.upper()}"):
            ids = self.tok.encode(v, add_special_tokens=False)
            if ids:
                s.add(ids[0])
        return sorted(s)

    def _logits_fp32(self, hidden_rows, ids):
        W = self.model.get_output_embeddings().weight
        lg = hidden_rows.float() @ W[ids].float().T
        if self.softcap:
            lg = torch.tanh(lg / self.softcap) * self.softcap
        return lg

    def margins_fp32(self, hidden_rows):
        """Yes/No log-odds (fp32 lm_head)."""
        ids = torch.tensor(self.yes_ids + self.no_ids, device=self.device)
        lg = self._logits_fp32(hidden_rows, ids)
        ny = len(self.yes_ids)
        return (torch.logsumexp(lg[:, :ny], 1) - torch.logsumexp(lg[:, ny:], 1)).tolist()

    def choice_logprobs(self, hidden_row, choices):
        """log-probabilities of each choice (a list of token-id lists) under the softmax restricted to the union
        of the choice tokens; returns a list aligned with `choices`."""
        flat = [t for c in choices for t in c]
        ids = torch.tensor(flat, device=self.device)
        lg = self._logits_fp32(hidden_row[None], ids)[0]
        lp = torch.log_softmax(lg, 0)
        out, k = [], 0
        for c in choices:
            out.append(float(torch.logsumexp(lp[k:k + len(c)], 0)))
            k += len(c)
        return out

    # ---- prompt construction
    def turn(self, role, text):
        return {"role": role, "content": [{"type": "text", "text": text}] if self.list_content else text}

    def prefix_messages(self, frames, segments, with_context=True, with_frames=True):
        content = []
        intro = "Observe this video."
        if with_frames and frames:
            if self.list_content:
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
        return [self.turn("system", SYSTEM_MESSAGE), {"role": "user", "content": content}], images

    def render(self, msgs, add_generation_prompt):
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)

    SENTINEL = "@@PREFIX_END@@"

    @staticmethod
    def _append_text(msgs, text):
        out = copy.deepcopy(msgs)
        content = out[-1]["content"]
        if isinstance(content, str):
            out[-1]["content"] = content + text
        elif content and content[-1].get("type") == "text":
            content[-1]["text"] += text
        else:
            content.append({"type": "text", "text": text})
        return out

    def conv(self, msgs, question, history=None):
        """Message list for asking `question` after the prefix and optional earlier turns (user/assistant pairs)."""
        history = list(history or [])
        if not self.same_turn:
            return msgs + history + [{"role": "user", "content": [{"type": "text", "text": question}]}]
        if history:
            q0 = history[0]["content"][0]["text"] if isinstance(history[0]["content"], list) else history[0]["content"]
            return self._append_text(msgs, "\n\n" + q0) + history[1:] + \
                [{"role": "user", "content": [{"type": "text", "text": question}]}]
        return self._append_text(msgs, "\n\n" + question)

    def encode(self, text, images):
        if images:
            imgs = [images] if self.family == "llava_onevision" else images
            return self.processor(text=[text], images=imgs, return_tensors="pt", add_special_tokens=False, **self.img_kw)
        return self.processor(text=[text], return_tensors="pt", add_special_tokens=False)

    def encode_prefix(self, msgs, image_files):
        from PIL import Image
        if self.same_turn:
            full = self.render(self._append_text(msgs, self.SENTINEL), add_generation_prompt=False)
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

    def model_inputs(self, enc):
        kw = {}
        for k, v in enc.items():
            if k in self.forward_params and torch.is_tensor(v):
                kw[k] = v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device)
        if "mm_token_type_ids" in self.forward_params and "mm_token_type_ids" not in kw and "pixel_values" in kw:
            kw["mm_token_type_ids"] = (enc["input_ids"] == self.image_token_id).to(torch.int32).to(self.device)
        return kw

    def branch_ids(self, msgs, question, history=None, head_text=None):
        """Tokens of the user turn + assistant header for `question`, as the suffix after `head_text`
        (default: the prefix string; with history: prefix + earlier turns)."""
        full = self.render(self.conv(msgs, question, history), add_generation_prompt=True)
        head = self._prefix_text if head_text is None else head_text
        if not full.startswith(head):
            raise AssertionError("chat template does not extend the current conversation string")
        text = full[len(head):]
        return self.tok(text, add_special_tokens=False)["input_ids"], text

    def answer_ids(self, msgs, question, answer, history=None):
        """Tokens the template adds for a completed assistant turn (answer + end-of-turn) after Q + header."""
        head = self.render(self.conv(msgs, question, history), add_generation_prompt=True)
        full = self.render(self.conv(msgs, question, history) + [self.turn("assistant", answer)],
                           add_generation_prompt=False)
        if full.startswith(head):
            text = full[len(head):]
        else:
            probe = self.render([self.turn("user", "x"), self.turn("assistant", "YQZ"), self.turn("user", "z")],
                                add_generation_prompt=False)
            after = probe[probe.index("YQZ") + 3:]
            nxt = after.find("<|im_start|>") if "<|im_start|>" in after else -1
            text = answer + (after[:nxt] if nxt >= 0 else after)
            self.loose_stance_seam = True
        return self.tok(text, add_special_tokens=False)["input_ids"], text

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

    # ---- cache path
    @torch.no_grad()
    def prefix_cache(self, enc):
        out = self.model.model(**self.model_inputs(enc), use_cache=True)
        cache = out.past_key_values
        del out
        return cache

    @torch.no_grad()
    def _step(self, cache, ids):
        """Forward `ids` on `cache` (in place); returns the last hidden row."""
        out = self.model.model(input_ids=torch.tensor([ids], device=self.device), past_key_values=cache, use_cache=True)
        h = out.last_hidden_state[0, -1]
        del out
        return h

    def extend_cache(self, cache, ids):
        self._step(cache, ids)
        return cache

    def cached_margin(self, cache, ids, in_place=False):
        """Yes/No log-odds at the end of `ids`. in_place=False: on a deep copy (independent branch);
        in_place=True: the cache keeps the tokens (sequential branch)."""
        c = cache if in_place else copy.deepcopy(cache)
        z = self.margins_fp32(self._step(c, ids)[None])[0]
        if not in_place:
            del c
        return z

    def cached_choices(self, cache, ids, choices, in_place=False):
        c = cache if in_place else copy.deepcopy(cache)
        lp = self.choice_logprobs(self._step(c, ids), choices)
        if not in_place:
            del c
        return lp

    @torch.no_grad()
    def cached_generate(self, cache, ids, max_new_tokens=160, in_place=False):
        """Greedy generation after `ids`; returns (text, generated ids). Stops at an EOS / end-of-turn token."""
        c = cache if in_place else copy.deepcopy(cache)
        h = self._step(c, ids)
        W = self.model.get_output_embeddings().weight
        gen = []
        for _ in range(max_new_tokens):
            lg = h.float() @ W.float().T
            if self.softcap:
                lg = torch.tanh(lg / self.softcap) * self.softcap
            nxt = int(lg.argmax())
            if nxt in self.eos_ids:
                break
            gen.append(nxt)
            h = self._step(c, [nxt])
        if not in_place:
            del c
        return self.tok.decode(gen, skip_special_tokens=True).strip(), gen

    @torch.no_grad()
    def plain_margin(self, msgs, image_files, question, history=None):
        """Reference: one ordinary full call (no cache)."""
        from PIL import Image
        full = self.render(self.conv(msgs, question, history), add_generation_prompt=True)
        images = [Image.open(p).convert("RGB") for p in image_files]
        enc = self.encode(full, images)
        for im in images:
            im.close()
        out = self.model.model(**self.model_inputs(enc), use_cache=False)
        z = self.margins_fp32(out.last_hidden_state[0, -1:])[0]
        del out
        return z
