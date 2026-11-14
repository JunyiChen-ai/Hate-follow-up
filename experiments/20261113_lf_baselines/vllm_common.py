#!/usr/bin/env python3
"""Shared pieces of the Qwen3-VL-8B label-free baselines of this directory (README).

- vLLM engine for Qwen/Qwen3-VL-8B-Instruct with every content-hashing path off (prefix caching off, multimodal
  processor cache off, request-id item ids, and vLLM's multimodal hasher replaced by an error after start-up), as in
  experiments/20261008_baselines/clara/clara_rationale.py (CLAUDE.md hash ban).
- Frames: ffmpeg decodes a video at a fixed rate into JPEG files under .cache/tmp (frame k at k / fps s, longer side
  scaled to MAX_SIDE); the caller deletes them after use. No frame or feature is kept.
- Text of a time span: the forced-aligned words whose midpoint lies in the span (the words the Reader and the naive
  Qwen3-VL-8B per-window row read), or the proportional slice of the Whisper segments when a video has no alignment.
- Read-outs from generation log-probabilities (logprobs = 20): the log-odds of a 0/1 label token, and the expected
  value of a 0-1 score written with one decimal.
No label is read anywhere in this module.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from src import lf_baselines as L            # noqa: E402
from src.video_inputs import window_text     # noqa: E402

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
RUNS = os.environ.get("LF_RUN_ROOT") or os.path.join(REPO, "runs", "20261113_lf_baselines")
ALIGNED = os.path.join(REPO, "data", "asr_whisper_aligned")
TMP = os.path.join(REPO, ".cache", "tmp", "lf_q3vl")
MAX_SIDE = 640
NEG = -1e4                                   # log-probability bound when a token is outside the top 20


# ----------------------------------------------------------------------------- text
def aligned_words(ds):
    path = os.path.join(ALIGNED, ds, "words.jsonl")
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            out[r["video_id"]] = r
    return out


def span_text(rec, segments, t1, t2):
    """Words of [t1, t2): aligned words by midpoint (src/window_evidence.py aligned_slice), else the proportional
    slice of the Whisper segments (src/video_inputs.py window_text). Whitespace collapsed; '' when nothing."""
    if rec is not None and rec.get("status") == "aligned":
        text = " ".join(w["w"] for w in rec["words"] if t1 <= (w["s"] + w["e"]) / 2 < t2)
    else:
        text = window_text(segments, t1, t2)
    return " ".join(text.split())


# ----------------------------------------------------------------------------- frames
def decode_frames(path, fps, key):
    """ffmpeg -> JPEG frames at `fps` in TMP/<key>/; returns [(time_s, file)] sorted by time."""
    d = os.path.join(TMP, key)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    scale = ("scale='if(gt(iw,ih),min(iw,%d),-2)':'if(gt(iw,ih),-2,min(ih,%d))'" % (MAX_SIDE, MAX_SIDE))
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", path, "-map", "0:v:0", "-vf", "fps=%g,%s" % (fps, scale),
           "-q:v", "3", os.path.join(d, "%06d.jpg")]
    subprocess.run(cmd, check=True, timeout=1800)
    files = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    return [(i / fps, os.path.join(d, f)) for i, f in enumerate(files)]


def drop_frames(key):
    shutil.rmtree(os.path.join(TMP, key), ignore_errors=True)


def frames_in(frames, t1, t2):
    """Frames with time in [t1, t2); the nearest frame if none falls inside (a span shorter than the frame step)."""
    inside = [f for t, f in frames if t1 <= t < t2]
    if inside or not frames:
        return inside
    mid = (t1 + t2) / 2
    return [min(frames, key=lambda x: abs(x[0] - mid))[1]]


def uniform(items, k):
    """k items at uniform positions (all of them if there are at most k)."""
    if len(items) <= k:
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]


def open_images(files, reader_size=False):
    """PIL RGB images. reader_size: resize to the size the Reader's processor gives Qwen3-VL (smart_resize with factor
    32 and src/mllm_judge.py MIN_PIXELS / MAX_PIXELS), so that vLLM generation, which receives no processor arguments
    (passing them makes vLLM hash the request's image ids, blocked by the hash guard), and the fp32 read-out through
    Judge see images of the same size (Judge's processor leaves an image already at that size unchanged)."""
    from PIL import Image
    out = []
    for f in files:
        with Image.open(f) as im:
            img = im.convert("RGB")
        if reader_size:
            from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
            from src.mllm_judge import MAX_PIXELS, MIN_PIXELS
            h, w = smart_resize(img.height, img.width, factor=32, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
            if (h, w) != (img.height, img.width):
                img = img.resize((w, h), Image.BICUBIC)
        out.append(img)
    return out


# ----------------------------------------------------------------------------- engine
class Engine:
    """A Qwen model in vLLM (vision-language by default, `text_only` for a text LLM such as Qwen3-8B, whose chat
    template is applied with thinking off), greedy, sample log-probabilities on request."""

    def __init__(self, model=MODEL, max_model_len=16384, max_images=20, text_only=False, log=print, gpu_mem=0.90):
        import vllm.multimodal.hasher as H
        from transformers import AutoProcessor, AutoTokenizer
        from vllm import LLM

        def _no_hash(*a, **k):
            # vLLM 0.11 always passes tokenization kwargs ({"truncation": False}), so it would hash even a request
            # that carries its own item ids; the id string we supplied is returned as the key instead, so nothing is
            # hashed. Any call with real media content still stops the run.
            items = [v for key, v in k.items() if key in ("image", "video", "audio")]
            if len(items) == 1 and isinstance(items[0], str):
                return items[0]
            raise RuntimeError("content hashing is not allowed (CLAUDE.md hash ban)")

        self.model = model
        self.text_only = text_only
        if text_only:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                           enable_prefix_caching=False, seed=0)
        else:
            self.processor = AutoProcessor.from_pretrained(model)
            self.tokenizer = self.processor.tokenizer
            self.llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                           enable_prefix_caching=False, mm_processor_cache_gb=0,
                           limit_mm_per_prompt={"image": max_images, "video": 0}, seed=0)
        # start-up profiling uses synthetic images that vLLM keys internally; the guard is armed afterwards, so no
        # frame of ours can be hashed (requests carry request-id item ids, caches are off)
        H.MultiModalHasher.hash_kwargs = staticmethod(_no_hash)
        log("vLLM loaded: %s, max_model_len %d%s" % (model, max_model_len, " (text only)" if text_only else ""))

    def request(self, images, text, uid, prefill="", system=None):
        """One user turn: images first, then the text (MARS / Qwen order); `prefill` starts the assistant turn."""
        if self.text_only:
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
            prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                        enable_thinking=False) + prefill
            return {"prompt": prompt}
        msgs = []
        if system:
            msgs.append({"role": "system", "content": [{"type": "text", "text": system}]})
        msgs.append({"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]})
        prompt = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill
        req = {"prompt": prompt}
        if images:
            req["multi_modal_data"] = {"image": images}
            req["multi_modal_uuids"] = {"image": ["%s/img%03d" % (uid, i) for i in range(len(images))]}
        return req

    def n_tokens(self, text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def generate(self, reqs, max_tokens, logprobs=None):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=logprobs, seed=0)
        outs = self.llm.generate(reqs, sp, use_tqdm=False)
        return [o.outputs[0] for o in outs]


# ----------------------------------------------------------------------------- fp32 read-out (HF, HateVLM env)
class Scorer:
    """One ordinary forward of Qwen3-VL-8B in HF transformers (src/mllm_judge.py Judge: bf16 weights, the Reader's
    image size) on a user turn plus a fixed start of the assistant turn; fp32 output layer over a few single-token
    candidates, softmax restricted to them. Used to read a method's own final answer (a 0/1 label or a score digit)
    from the position where the method writes it, so that scores are continuous instead of rounded."""

    def __init__(self, model=MODEL):
        import torch
        from src.mllm_judge import Judge
        self.torch = torch
        self.j = Judge(model)
        self._ids = {}

    def token_id(self, s):
        if s not in self._ids:
            ids = self.j.tok.encode(s, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError("%r is not one token: %s" % (s, ids))
            self._ids[s] = ids[0]
        return self._ids[s]

    def render(self, n_images, text, prefill="", system=None):
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []
        msgs.append({"role": "user", "content": [{"type": "image"}] * n_images + [{"type": "text", "text": text}]})
        return self.j.render(msgs, add_generation_prompt=True) + prefill

    def next_logprobs(self, images, text, prefill, candidates, system=None):
        """images: PIL images (already opened). Returns {candidate: log-probability} under the restricted softmax."""
        torch = self.torch
        full = self.render(len(images), text, prefill, system)
        with torch.inference_mode():
            enc = self.j.encode(full, images)
            out = self.j.model.model(**self.j.model_inputs(enc), use_cache=False)
            ids = torch.tensor([self.token_id(c) for c in candidates], device=self.j.device)
            lg = self.j._logits_fp32(out.last_hidden_state[0, -1:], ids)[0]
            lp = torch.log_softmax(lg, 0).tolist()
            del out
        return dict(zip(candidates, lp))

    def score_digits(self, images, text, prefill, system=None):
        """A 0-1 score written right after `prefill`: one forward on prefill + "0." gives the first-digit distribution
        (the row before "0") and the decimal-digit distribution after "0." (the last row), both over '0'..'9'.
        Returns (lp_first, lp_decimal) for expected_score."""
        torch = self.torch
        full = self.render(len(images), text, prefill + "0.", system)
        with torch.inference_mode():
            enc = self.j.encode(full, images)
            ids = enc["input_ids"][0].tolist()
            if self.j.tok.decode(ids[-2:]) != "0." or self.j.tok.decode(ids[-1:]) != ".":
                raise RuntimeError("'0.' is not the last two tokens: %r" % self.j.tok.decode(ids[-4:]))
            out = self.j.model.model(**self.j.model_inputs(enc), use_cache=False)
            cand = torch.tensor([self.token_id(d) for d in DIGITS], device=self.j.device)
            lg = self.j._logits_fp32(out.last_hidden_state[0, [-3, -1]], cand)
            lp = torch.log_softmax(lg, 1).tolist()
            del out
        return dict(zip(DIGITS, lp[0])), dict(zip(DIGITS, lp[1]))


def label_prefix(text, key="label"):
    """The generated text up to (not including) the first 0/1 digit that follows `"<key>":` (optionally a quote and
    spaces), i.e. the assistant prefix after which the method writes its label; None if there is none."""
    m = re.search(r'"%s"\s*:\s*"?\s*(?=[01])' % re.escape(key), text or "")
    return text[:m.end()] if m else None


def expected_score(lp_first, lp_decimal):
    """Expected value on [0, 1] of a score written as '0.d' or '1' (a first digit d >= 2 is read on a 0-10 scale as
    d / 10): lp_first over '0'..'9' at the first digit, lp_decimal over '0'..'9' after '0.'."""
    p = {int(k): math.exp(v) for k, v in lp_first.items()}
    q = {int(k): math.exp(v) for k, v in lp_decimal.items()}
    e0 = sum(k / 10 * v for k, v in q.items()) / sum(q.values())
    return sum(v * (e0 if d == 0 else (1.0 if d == 1 else d / 10)) for d, v in p.items()) / sum(p.values())


DIGITS = [str(d) for d in range(10)]


# ----------------------------------------------------------------------------- read-outs from vLLM log-probabilities
def _lse(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _token_lp(step, want):
    """log P of the tokens at one generated position whose text, stripped of spaces and quotes, equals `want`
    (several tokenisations add up); None if none is in the top 20."""
    return _lse([lp.logprob for lp in step.values() if (lp.decoded_token or "").strip().strip('"') == want])


def _floor(step):
    return min(lp.logprob for lp in step.values()) if step else NEG


def yes_no_logodds(out):
    """log P(Yes) - log P(No) at the first generated token (Yes / yes / YES and the No forms, with or without a
    leading space, added up). A side outside the top 20 is bounded by the smallest top-20 log-probability."""
    step = (out.logprobs or [{}])[0]
    py = _lse([lp.logprob for lp in step.values() if (lp.decoded_token or "").strip().lower() == "yes"])
    pn = _lse([lp.logprob for lp in step.values() if (lp.decoded_token or "").strip().lower() == "no"])
    fl = _floor(step)
    status = "ok" if (py is not None and pn is not None) else "bounded"
    return (py if py is not None else fl) - (pn if pn is not None else fl), status


def label_logodds(out, key="label"):
    """log P(1) - log P(0) at the first generated token that follows `"<key>":` (optionally a quote).
    Returns (z, status). A side outside the top 20 is bounded by the smallest top-20 log-probability."""
    toks = out.token_ids
    steps = out.logprobs or []
    text = ""
    pat = re.compile(r'"%s"\s*:\s*"?\s*$' % re.escape(key))
    for i, step in enumerate(steps):
        tok = step.get(toks[i])
        piece = tok.decoded_token if tok is not None else ""
        if pat.search(text) and piece.strip().strip('"')[:1] in ("0", "1"):
            p1, p0 = _token_lp(step, "1"), _token_lp(step, "0")
            fl = _floor(step)
            status = "ok" if (p1 is not None and p0 is not None) else "bounded"
            return (p1 if p1 is not None else fl) - (p0 if p0 is not None else fl), status
        text += piece
    return None, "no_label"


def expected_decimal(steps_first, steps_decimal):
    """Expected score on [0, 1] from the first digit distribution ('0' or '1', or a 0-10 integer d >= 2 read as d / 10)
    and, for the '0.' branch, the decimal digit distribution. Each argument is one position's top-20 dict."""
    mass, val = 0.0, 0.0
    p = {}
    for d in range(10):
        lp = _token_lp(steps_first, str(d))
        if lp is not None:
            p[d] = math.exp(lp)
    if not p:
        return None
    if steps_decimal is not None:
        q = {k: math.exp(_token_lp(steps_decimal, str(k))) for k in range(10) if _token_lp(steps_decimal, str(k)) is not None}
        qs = sum(q.values())
        e0 = sum(k / 10 * v for k, v in q.items()) / qs if qs > 0 else 0.0
    else:
        e0 = 0.0
    for d, v in p.items():
        mass += v
        val += v * (e0 if d == 0 else (1.0 if d == 1 else d / 10))
    return val / mass


def clean_json(text):
    """MARS's clean_json_response: strip a ```json fence only if the text starts and ends with one, then json.loads;
    None when it does not parse."""
    t = (text or "").strip()
    if t.startswith("```json") and t.endswith("```"):
        t = t[len("```json"):-3].strip()
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        return None


def log_to(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(msg):
        import time
        line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line, flush=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")
    return log
