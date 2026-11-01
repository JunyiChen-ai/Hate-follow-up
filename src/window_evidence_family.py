"""The adopted window-evidence reader on every MLLM family of src/mllm_judge.FAMILY_IMAGE_KW (experiments/20261101_backbones).

Reader (as src/window_evidence.py + src/stance_cache.py): shared prefix (20 timestamped frames + full transcript + rules)
encoded once -> whole-video verdict -> the model's own Yes/No written back as the stance turn -> per 8 s window an
isolated visual branch (the window's 0.5 fps frames right before the r6 visual question) and an isolated speech branch
(the r6 speech question with the forced-aligned word slice). Windows without an in-window frame keep the question-only
visual read; windows without words have no speech branch.

Two code paths, chosen by the family:
- mRoPE families (qwen3_vl, qwen2_5_vl): the adopted code unchanged (stance_cache.build / margin,
  window_evidence.frames_margin), so Qwen3-VL-8B reproduces the adopted reads exactly.
- 1D-RoPE families (internvl, gemma3, llava_onevision): the same conversation through the family's chat template,
  positions from the cache length, every branch on a deep copy of the shared cache (Gemma 3's sliding-window cache
  layers cannot be cropped back once full). Template adaptations, as in the 2026-09-10 family study:
  Gemma's template rejects two consecutive user turns, so a question without a preceding answer joins the prefix's
  user turn (Judge.same_turn); LLaVA-OneVision renders a turn's images before its text, so the window's frame times are
  given as a list after its frames (as its prefix does), and its stance turn is the answer + end-of-turn after the
  generation header (Judge.loose_stance_seam), so its branches are rendered without the stance history and appended
  after the cached stance tokens.

Also here: the per-window read alone (no whole-video prefix, verdict or stance; only the window's frames and aligned
words with the joint question, one plain call per window) and plain uncached references for the porting check.
No labels are read anywhere in this module.
"""
import copy
import math

import numpy as np
import torch

from src import stance_cache, window_evidence
from src.mllm_judge import VIDEO_QUESTION, yesno_question
from src.video_inputs import fixed_windows, window_text
from src.window_evidence import NATIVE_LABEL, WINDOW_S, aligned_slice, in_window

MROPE = ("qwen3_vl", "qwen2_5_vl")


def is_mrope(j):
    return j.family in MROPE


def open_images(paths):
    return window_evidence._open_images(paths)


def close_images(images):
    for im in images:
        im.close()


# ---------------------------------------------------------------- conversation with list content
def frame_items(j, frames, question):
    """Content items of one user turn: the window's frames with their times, then the question."""
    if not frames:
        return [dict(type="text", text=question)]
    if j.list_content:  # LLaVA-OneVision: images are rendered before the turn's text
        times = ", ".join(f"frame {k + 1} at t={t:.1f}s" for k, (t, _) in enumerate(frames))
        lead = f"The {len(frames)} images above are frames sampled from this window, in order; frame k was taken at: {times}.\n"
        return [dict(type="image") for _ in frames] + [dict(type="text", text=lead + question)]
    items = []
    for t, _ in frames:
        items += [dict(type="text", text=NATIVE_LABEL.format(time=t)), dict(type="image")]
    items.append(dict(type="text", text=question))
    return items


def conv_items(j, msgs, items, history=None):
    """Judge.conv with a list of content items (text / image) as the asked turn."""
    history = list(history or [])
    if not j.same_turn:
        return msgs + history + [dict(role="user", content=copy.deepcopy(items))]
    if history:
        q0 = history[0]["content"][0]["text"] if isinstance(history[0]["content"], list) else history[0]["content"]
        return j._append_text(msgs, "\n\n" + q0) + history[1:] + [dict(role="user", content=copy.deepcopy(items))]
    # same-turn template, nothing answered yet: the asked content joins the prefix's own user turn
    first, rest = (items[0]["text"], items[1:]) if items[0]["type"] == "text" else ("", items)
    out = j._append_text(msgs, "\n\n" + first)
    content = out[-1]["content"]
    assert isinstance(content, list)
    content.extend(copy.deepcopy(rest))
    return out


# ---------------------------------------------------------------- shared context (prefix + verdict + stance)
@torch.no_grad()
def build(j, frames, segments):
    if is_mrope(j):
        cache, ctx = stance_cache.build(j, frames, segments)
        ctx["prefix_text"] = j._prefix_text
        return cache, ctx
    msgs, files = j.prefix_messages(frames, segments)
    text, enc = j.encode_prefix(msgs, files)
    cache = prefix_cache_full(j, enc)
    qid, qtext = j.branch_ids(msgs, VIDEO_QUESTION)
    z = j.cached_margin(cache, qid, in_place=True)
    stance = "Yes" if z > 0 else "No"
    aid, atext = j.answer_ids(msgs, VIDEO_QUESTION, stance)
    j.extend_cache(cache, aid)
    history = [dict(role="user", content=[dict(type="text", text=VIDEO_QUESTION)]), j.turn("assistant", stance)]
    n_prefix = int(enc["input_ids"].shape[1])
    assert cache.get_seq_length() == n_prefix + len(qid) + len(aid)
    return cache, dict(msgs=msgs, files=files, history=history, prefix_text=text, head=text + qtext + atext,
                       head_ids=enc["input_ids"][0].tolist() + qid + aid, global_margin=z, stance=stance,
                       stance_cache_tokens=cache.get_seq_length(), prefix_tokens=n_prefix, image_counts=list(j.img_tokens))


def _suffix_text(j, ctx, items):
    """Text of the asked turn + generation header, as the continuation of the cached conversation."""
    if j.loose_stance_seam:  # the template renders a finished assistant turn differently from its generation header
        full = j.render(conv_items(j, ctx["msgs"], items), True)
        head = ctx["prefix_text"]
    else:
        full = j.render(conv_items(j, ctx["msgs"], items, ctx["history"]), True)
        head = ctx["head"]
    assert full.startswith(head), "chat template does not extend the cached conversation"
    return full[len(head):]


def _image_token_check(j, enc, n_images):
    """Every window frame costs the prefix frames' token count. Exception: LLaVA-OneVision reads a turn with a single
    image in its single-image (AnyRes) mode, which costs more tokens; this happens only for a window with one frame."""
    ids = enc["input_ids"][0]
    if j.image_token_id is None or not n_images or not j.img_tokens:
        return
    if j.family == "llava_onevision" and n_images == 1:
        return
    assert int((ids == j.image_token_id).sum()) == n_images * j.img_tokens[0], "image token count differs from the prefix frames"


@torch.no_grad()
def prefix_cache_full(j, enc):
    """1D families: the prefix forward on an explicit full-length DynamicCache (every layer keeps every token).
    For InternVL / LLaVA-OneVision this is the cache the model builds itself (no sliding layers). For Gemma 3 the
    model's own cache keeps only the last 1023 tokens in its sliding-window layers, and transformers 5.15.1 then builds
    the image-block mask of a suffix with images over the full length (shape error); with full-length layers the
    sliding window is applied by the mask alone, as in one uncached forward."""
    from transformers import DynamicCache
    out = j.model.model(**j.model_inputs(enc), past_key_values=DynamicCache(), use_cache=True)
    cache = out.past_key_values
    del out
    return cache


def _in_place(j):
    """Gemma 3 (12B weights + a 2.4 GB full-length cache) reads branches in place and crops back; the others on a copy."""
    return j.family == "gemma3"


@torch.no_grad()
def _forward_suffix(j, cache, text, image_paths, head_ids=None):
    """One forward of `text` (+ images) on the shared cache, isolated (a deep copy, or in place and cropped back);
    Yes/No log-odds at its end."""
    n = cache.get_seq_length()
    c = cache if _in_place(j) else copy.deepcopy(cache)
    try:
        return _forward_on(j, c, text, image_paths, head_ids)
    finally:
        if _in_place(j):
            cache.crop(n)
            assert cache.get_seq_length() == n
        else:
            del c


def _forward_on(j, c, text, image_paths, head_ids=None):
    images = open_images(image_paths)
    try:
        enc = j.encode(text, images) if images else j.encode(text, [])
    finally:
        close_images(images)
    _image_token_check(j, enc, len(image_paths))
    kw = j.model_inputs(enc)
    kw.pop("attention_mask", None)  # the processor's mask covers the suffix only; the model builds the mask over cache + suffix
    if "token_type_ids" in kw and j.family == "gemma3":
        # Gemma 3 lets an image's tokens attend to each other both ways; transformers 5.15.1 indexes the image-block ids by
        # absolute position, so they must cover the cached tokens too (cached image tokens = 1, as the processor marks them)
        assert head_ids is not None and len(head_ids) == c.get_seq_length()
        cached = torch.tensor([[int(t == j.image_token_id) for t in head_ids]], dtype=kw["token_type_ids"].dtype, device=j.device)
        kw["token_type_ids"] = torch.cat((cached, kw["token_type_ids"]), 1)
    out = j.model.model(**kw, past_key_values=c, use_cache=True)
    hidden = out.last_hidden_state[0, -1].clone()
    del out
    return j.margins_fp32(hidden[None])[0], enc["input_ids"][0].tolist()


@torch.no_grad()
def margin(j, cache, ctx, question):
    """Text-only branch on the shared cache."""
    if is_mrope(j):
        return stance_cache.margin(j, cache, ctx, question)
    z, _ = _forward_suffix(j, cache, _suffix_text(j, ctx, [dict(type="text", text=question)]), [], ctx["head_ids"])
    return z


@torch.no_grad()
def frames_margin(j, cache, ctx, frames, question):
    """The window's frames, then the question, as one user turn on the shared cache."""
    if is_mrope(j):
        return window_evidence.frames_margin(j, cache, ctx, frames, question)
    items = frame_items(j, frames, question)
    z, _ = _forward_suffix(j, cache, _suffix_text(j, ctx, items), [p for _, p in frames], ctx["head_ids"])
    return z


# ---------------------------------------------------------------- the adopted reader, one video
@torch.no_grad()
def read_video(j, row, native, frames, segments, rec):
    """All window reads of one video. Returns (ctx, traces, cache)."""
    cache, ctx = build(j, native, segments)
    wins = fixed_windows(float(row["duration"]), WINDOW_S)
    traces = []
    for i, (t1, t2) in enumerate(wins):
        body = window_text(segments, t1, t2)
        abody = aligned_slice(rec, t1, t2, body)
        vq = yesno_question(i, len(wins), t1, t2, body, "visual")
        inside = in_window(frames, t1, t2)
        v = frames_margin(j, cache, ctx, inside, vq) if inside else margin(j, cache, ctx, vq)
        s = margin(j, cache, ctx, yesno_question(i, len(wins), t1, t2, abody, "speech")) if abody is not None and abody.strip() else None
        traces.append(dict(i=i, bounds=[t1, t2], local_times=[t for t, _ in inside], text_aligned=abody, z_visual=v, z_speech=s))
    return ctx, traces, cache


# ---------------------------------------------------------------- per window alone (no prefix, verdict, stance)
@torch.no_grad()
def plain_margin_items(j, msgs, items, image_paths):
    """One ordinary uncached call: msgs + the asked items."""
    if is_mrope(j):
        j.model.model.rope_deltas = None
    text = j.render(conv_items(j, msgs, items), True)
    return plain_margin_text(j, text, image_paths)


@torch.no_grad()
def plain_margin_text(j, text, image_paths):
    if is_mrope(j):
        j.model.model.rope_deltas = None
    images = open_images(image_paths)
    try:
        enc = j.encode(text, images) if images else j.encode(text, [])
    finally:
        close_images(images)
    out = j.model.model(**j.model_inputs(enc), use_cache=False)
    hidden = out.last_hidden_state[0, -1].clone()
    del out
    if is_mrope(j):
        j.model.model.rope_deltas = None
    return j.margins_fp32(hidden[None])[0], enc["input_ids"][0].tolist()


def alone_messages(j):
    """The rules-only prefix of the old per-window-alone arm: no frames, no transcript, no verdict, no stance."""
    msgs, files = j.prefix_messages([], [], with_context=False, with_frames=False)
    assert not files
    return msgs


@torch.no_grad()
def read_alone(j, row, frames, segments, rec):
    """Per window, one plain call: rules-only prefix + the window's 0.5 fps frames + the joint question with the
    window's aligned words. Returns the per-window log-odds."""
    msgs = alone_messages(j)
    wins = fixed_windows(float(row["duration"]), WINDOW_S)
    out = []
    for i, (t1, t2) in enumerate(wins):
        abody = aligned_slice(rec, t1, t2, window_text(segments, t1, t2))
        inside = in_window(frames, t1, t2)
        q = yesno_question(i, len(wins), t1, t2, abody, "joint")
        z, _ = plain_margin_items(j, msgs, frame_items(j, inside, q), [p for _, p in inside])
        out.append(z)
    return out


# ---------------------------------------------------------------- reader arms (experiments/20261101_reader_ablation §2)
ARMS = ("no_transcript", "no_prefix_frames", "no_stance", "asr_windows", "joint")
# A4 here: at most this many in-window frames per segment window (uniformly spaced over the segment's 0.5 fps frames).
# Whisper segments reach 479 s (240 frames: 175k image tokens on LLaVA-OneVision, past its 32k context), so the
# reader-ablation A4 (every frame of the segment) cannot be read on every backbone. 4 = the frames of one fixed 8 s
# window, the fixed grid's per-read budget (experiments/20261101_backbones/README.md §2).
ASR_MAX_FRAMES = 4


def cap_frames(inside, k=ASR_MAX_FRAMES):
    if len(inside) <= k:
        return inside
    return [inside[int(i)] for i in np.linspace(0, len(inside) - 1, k).round().astype(int)]


@torch.no_grad()
def build_arm(j, frames, segments, with_context=True, with_frames=True, stance=True):
    """Shared context with the reader-ablation switches. mRoPE families: src.reader_arms.build_context (the reader
    ablation's own code). 1D families: the same steps; without the stance turn the verdict is read on a copy of the
    prefix cache, so the window branches follow the prefix directly."""
    if is_mrope(j):
        from src.reader_arms import build_context
        cache, ctx = build_context(j, frames, segments, with_context=with_context, with_frames=with_frames, stance=stance)
        ctx["prefix_text"] = j._prefix_text
        return cache, ctx
    msgs, files = j.prefix_messages(frames if with_frames else [], segments, with_context=with_context, with_frames=with_frames)
    text, enc = j.encode_prefix(msgs, files)
    cache = prefix_cache_full(j, enc)
    n_prefix = int(enc["input_ids"].shape[1])
    qid, qtext = j.branch_ids(msgs, VIDEO_QUESTION)
    if stance:
        z = j.cached_margin(cache, qid, in_place=True)
        st = "Yes" if z > 0 else "No"
        aid, atext = j.answer_ids(msgs, VIDEO_QUESTION, st)
        j.extend_cache(cache, aid)
        history = [dict(role="user", content=[dict(type="text", text=VIDEO_QUESTION)]), j.turn("assistant", st)]
        head, head_ids = text + qtext + atext, enc["input_ids"][0].tolist() + qid + aid
    else:
        n0 = cache.get_seq_length()
        z = j.cached_margin(cache, qid, in_place=_in_place(j))
        if _in_place(j):
            cache.crop(n0)
        st, history, head, head_ids = None, [], text, enc["input_ids"][0].tolist()
    assert cache.get_seq_length() == len(head_ids)
    return cache, dict(msgs=msgs, files=files, history=history, prefix_text=text, head=head, head_ids=head_ids, global_margin=z, stance=st,
                       stance_cache_tokens=cache.get_seq_length(), prefix_tokens=n_prefix, image_counts=list(j.img_tokens))


@torch.no_grad()
def read_arm(j, arm, row, native, frames, segments, rec):
    """One reader-ablation arm on one video, as experiments/20261101_reader_ablation/read_arms.py reads it.
    Returns (ctx, windows, per-window reads)."""
    assert arm in ARMS
    cache, ctx = build_arm(j, native, segments, with_context=arm != "no_transcript", with_frames=arm != "no_prefix_frames",
                           stance=arm != "no_stance")
    if arm == "asr_windows":
        segs = [(float(s), float(e), t) for s, e, t in segments]
        wins = [(s, e) for s, e, _ in segs]
        bodies = [aligned_slice(rec, s, e, t) for s, e, t in segs]
    else:
        wins = fixed_windows(float(row["duration"]), WINDOW_S)
        bodies = [aligned_slice(rec, t1, t2, window_text(segments, t1, t2)) for t1, t2 in wins]
    n, reads = len(wins), []
    for i, ((t1, t2), body) in enumerate(zip(wins, bodies)):
        inside = in_window(frames, t1, t2)
        if arm == "asr_windows":
            inside = cap_frames(inside)
        has_speech = body is not None and body.strip() != ""
        if arm == "joint":
            q = yesno_question(i, n, t1, t2, body, "joint")
            r = dict(z_joint=frames_margin(j, cache, ctx, inside, q) if inside else margin(j, cache, ctx, q))
        else:
            vq = yesno_question(i, n, t1, t2, body, "visual")
            r = dict(z_visual=frames_margin(j, cache, ctx, inside, vq) if inside else margin(j, cache, ctx, vq))
            if has_speech:
                r["z_speech"] = margin(j, cache, ctx, yesno_question(i, n, t1, t2, body, "speech"))
        reads.append(r)
    del cache
    return ctx, wins, reads


def alone_prediction(row, z, seconds, code_path):
    ww = [dict(i=i, start=w[0], end=w[1], z=v) for i, (w, v) in enumerate(zip(fixed_windows(float(row["duration"]), WINDOW_S), z))]
    idx = np.clip(((np.arange(math.ceil(float(row["duration"]) * 4)) + .5) / 4 // WINDOW_S).astype(int), 0, len(ww) - 1)
    return dict(schema_version=1, method="backbones_alone", dataset=row["dataset"], video_id=row["video_id"], duration=float(row["duration"]),
                native_rate=4, score_curve=np.asarray([w["z"] for w in ww])[idx].tolist(), intervals=[], error=None, seed=0,
                calls=len(ww), code_path=code_path, extra=dict(z_video=None, windows=ww, standalone_seconds=seconds))


# ---------------------------------------------------------------- porting check: cached branch vs one plain call
def _branch_suffix(j, ctx, items):
    if is_mrope(j):  # as window_evidence.frames_margin / stance_cache.margin render it
        full = j.render(ctx["msgs"] + ctx["history"] + [dict(role="user", content=copy.deepcopy(items))], True)
        assert full.startswith(ctx["head"])
        return full[len(ctx["head"]):]
    return _suffix_text(j, ctx, items)


@torch.no_grad()
def plain_reference(j, ctx, items, image_paths):
    """The same input as a cached branch (cached string + asked turn), read in one ordinary uncached forward with all
    images (prefix frames + the branch's frames). Returns (z, token seam identical)."""
    text = _branch_suffix(j, ctx, items)
    images = open_images(image_paths)
    try:
        suffix_ids = (j.encode(text, images) if images else j.encode(text, []))["input_ids"][0].tolist()
    finally:
        close_images(images)
    z, ids = plain_margin_text(j, ctx["head"] + text, [str(p) for p in ctx["files"]] + [str(p) for p in image_paths])
    if is_mrope(j):  # the adopted path does not keep the cached ids; their count is the cache length
        seam = len(ids) == ctx["stance_cache_tokens"] + len(suffix_ids) and ids[-len(suffix_ids):] == suffix_ids
    else:
        seam = ids == ctx["head_ids"] + suffix_ids
    return z, seam


@torch.no_grad()
def plain_check(j, row, frames, segments, ctx, traces, n_windows=4):
    """Verdict and the first n windows' visual / speech branches against plain uncached calls on the same input."""
    from scipy.stats import spearmanr
    vz, _ = plain_margin_items(j, ctx["msgs"], [dict(type="text", text=VIDEO_QUESTION)], [str(p) for p in ctx["files"]])
    wins = fixed_windows(float(row["duration"]), WINDOW_S)
    cached, plain, dz_vis, dz_sp, seams = [ctx["global_margin"]], [vz], [], [], []
    for t in traces[:n_windows]:
        i, (t1, t2) = t["i"], t["bounds"]
        body = window_text(segments, t1, t2)
        inside = in_window(frames, t1, t2)
        vq = yesno_question(i, len(wins), t1, t2, body, "visual")
        if not (j.family == "llava_onevision" and len(inside) == 1):  # single-image mode in the branch, multi-image in one plain call
            zv, seam = plain_reference(j, ctx, frame_items(j, inside, vq), [p for _, p in inside])
            dz_vis.append(abs(zv - t["z_visual"]))
            seams.append(seam)
            cached.append(t["z_visual"]); plain.append(zv)
        if t["z_speech"] is not None:
            zs, seam = plain_reference(j, ctx, [dict(type="text", text=yesno_question(i, len(wins), t1, t2, t["text_aligned"], "speech"))], [])
            dz_sp.append(abs(zs - t["z_speech"]))
            seams.append(seam)
            cached.append(t["z_speech"]); plain.append(zs)
    rho = float(spearmanr(cached, plain).correlation) if len(cached) >= 3 else None
    rho = None if rho is not None and rho != rho else rho  # constant reads give no rank correlation
    return dict(verdict_dz=abs(vz - ctx["global_margin"]), visual_max_dz=max(dz_vis, default=0.0), speech_max_dz=max(dz_sp, default=0.0),
                branches=len(cached), spearman=rho, token_seam_identical=all(seams))
