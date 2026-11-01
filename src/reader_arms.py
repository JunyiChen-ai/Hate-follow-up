"""Reader ablation arms of the adopted window-evidence reader (experiments/20261101_reader_ablation, 2026-11-01).

Every arm changes one part of the adopted reader (src/stance_cache.build + src/window_evidence) and keeps the rest:
- build_context: src.stance_cache.build with switches for the prefix transcript, the prefix frames and the stance
  turn; with all switches on it runs the same steps as build() (the smoke checks the reads against the stored ones).
- asr_windows: Whisper segments as windows.
- SequentialReader: branches that see each other (one growing conversation, the model's own Yes/No written back
  after every branch; nothing is cropped).
- record: prediction record for fixed-grid or segment windows, two branches or one joint branch, in the schema the
  decoder reads (extra.windows with z_visual / z_speech or z_joint, extra.z_video).
The existing src/ modules are imported unchanged. No labels are read anywhere in this module.
"""
import math

import numpy as np
import torch

from src.mllm_judge import VIDEO_QUESTION
from src.stance_cache import positions
from src.video_inputs import fixed_windows
from src.window_evidence import NATIVE_LABEL, WINDOW_S, _open_images

FILL_UNCOVERED = -12.0  # 4 fps cells no segment window covers (as experiments/20260922_til/til_measure.py)


# ---------------------------------------------------------------- shared context
@torch.no_grad()
def build_context(j, frames, segments, with_context=True, with_frames=True, stance=True):
    """Prefix KV cache + whole-video verdict. with_context=False: no transcript in the prefix; with_frames=False: no
    frames in the prefix; stance=False: the verdict is read on the prefix and the cache is cropped back, so the window
    branches follow the prefix directly. All True = src.stance_cache.build."""
    j.model.model.rope_deltas = None
    msgs, files = j.prefix_messages(frames if with_frames else [], segments, with_context=with_context, with_frames=with_frames)
    text, enc = j.encode_prefix(msgs, files)
    cache = j.prefix_cache(enc)
    n0 = cache.get_seq_length()
    qid, qtext = j.branch_ids(msgs, VIDEO_QUESTION)
    grid = enc["image_grid_thw"].to(j.device) if "image_grid_thw" in enc else None
    if stance:
        z = j.cached_margin(cache, qid, in_place=True)
        st = "Yes" if z > 0 else "No"
        aid, atext = j.answer_ids(msgs, VIDEO_QUESTION, st)
        j.extend_cache(cache, aid)
        history = [{"role": "user", "content": [{"type": "text", "text": VIDEO_QUESTION}]}, j.turn("assistant", st)]
        ids = torch.cat((enc["input_ids"].to(j.device), torch.tensor([qid + aid], device=j.device)), 1)
        head = text + qtext + atext
    else:
        rope0 = None if j.model.model.rope_deltas is None else j.model.model.rope_deltas.clone()
        z = j.cached_margin(cache, qid, in_place=True)
        cache.crop(n0)
        j.model.model.rope_deltas = rope0
        st, history, head = None, [], text
        ids = enc["input_ids"].to(j.device)
    p, delta = positions(j, ids, grid)
    if j.model.model.rope_deltas is None:
        # text-only prefix (with_frames=False): the model leaves rope_deltas unset and uses plain positions
        assert grid is None and int(delta.abs().max()) == 0
        j.model.model.rope_deltas = delta.clone()
    assert ids.shape[1] == cache.get_seq_length() and torch.equal(delta, j.model.model.rope_deltas)
    return cache, dict(msgs=msgs, files=files, history=history, head=head, positions=p, rope=delta.clone(), global_margin=z,
                       stance=st, stance_cache_tokens=cache.get_seq_length(), stance_cache_logical_start=int(p.max()) + 1,
                       prefix_tokens=enc["input_ids"].shape[1], image_counts=list(j.img_tokens))


# ---------------------------------------------------------------- segment windows
def asr_windows(segments):
    """[(start, end, text)] of the Whisper segments, in order, as loaded for the prefix (load_asr)."""
    return [(float(s), float(e), t) for s, e, t in segments]


# ---------------------------------------------------------------- branches that see each other
class SequentialReader:
    """One growing conversation on the shared cache after the verdict turn. Each call appends a user turn (labelled
    frames, if any, then the question), reads the Yes/No log-odds at its end, and writes the model's own answer
    ("Yes" if the log-odds > 0, else "No") as its assistant turn. Positions are explicit (each turn continues from
    the largest position so far, as src.window_evidence.frames_margin does for one turn)."""

    def __init__(self, j, cache, ctx):
        assert cache.get_seq_length() == ctx["stance_cache_tokens"]
        self.j, self.cache = j, cache
        self.msgs, self.history, self.head = ctx["msgs"], list(ctx["history"]), ctx["head"]
        self.tokens, self.next_pos = ctx["stance_cache_tokens"], ctx["stance_cache_logical_start"]
        self.rope = ctx["rope"]
        self.turns = 0

    def _forward(self, enc, rel):
        j = self.j
        kwargs = j.model_inputs(enc)
        kwargs.pop("attention_mask", None)
        pos = rel + self.next_pos
        kwargs["position_ids"] = pos
        out = j.model.model(**kwargs, past_key_values=self.cache, use_cache=True)
        hidden = out.last_hidden_state[0, -1].clone()
        del out
        self.next_pos = int(pos.max()) + 1
        self.tokens = self.cache.get_seq_length()
        # keep the implicit-position state consistent with the explicit positions (not used by this class)
        j.model.model.rope_deltas = torch.full_like(self.rope, self.next_pos - self.tokens)
        return hidden

    @torch.no_grad()
    def read(self, frames, question):
        j = self.j
        labelled = [(NATIVE_LABEL.format(time=t), p) for t, p in frames]
        content = []
        for label, path in labelled:
            content += [dict(type="text", text=label), dict(type="image", image=str(path))]
        content.append(dict(type="text", text=question))
        user = dict(role="user", content=content)
        full = j.render(self.msgs + self.history + [user], True)
        assert full.startswith(self.head) and self.cache.get_seq_length() == self.tokens
        text = full[len(self.head):]
        assert text.count(question) == 1
        if labelled:
            images = _open_images([p for _, p in labelled])
            try:
                enc = j.encode(text, images)
            finally:
                for image in images:
                    image.close()
            grid = enc["image_grid_thw"].to(j.device)
            assert len(grid) == len(labelled)
        else:
            enc = j.encode(text, [])
            grid = None
        ids = enc["input_ids"].to(j.device)
        rel, _ = positions(j, ids, grid)
        z = j.margins_fp32(self._forward(enc, rel)[None])[0]
        answer = "Yes" if z > 0 else "No"
        after = j.render(self.msgs + self.history + [user, j.turn("assistant", answer)], False)
        assert after.startswith(full), "template re-rendered the earlier turns"
        atext = after[len(full):]
        assert atext.startswith(answer)
        aenc = j.tok(atext, add_special_tokens=False, return_tensors="pt")
        aids = aenc["input_ids"].to(j.device)
        arel = torch.arange(aids.shape[1], device=j.device).view(1, 1, -1).expand(3, 1, -1)
        self._forward({"input_ids": aids}, arel)
        self.history += [user, j.turn("assistant", answer)]
        self.head = after
        self.turns += 1
        return z


# ---------------------------------------------------------------- output record
def record(row, ctx, wins, reads, method, seconds, code_path, segment_windows=False):
    """reads: per window a dict of branch reads ({"z_visual", "z_speech"} or {"z_joint"}; a missing branch is
    absent). Fixed grid: each 4 fps cell takes its window's z (as window_prediction); segment windows: cells covered
    by a window take the max z over the windows covering them, uncovered cells FILL_UNCOVERED."""
    dur = float(row["duration"])
    ww = []
    for (t1, t2), r in zip(wins, reads):
        one = dict(i=len(ww), start=t1, end=t2, **r)
        one["z"] = max(r.values())
        ww.append(one)
    L = math.ceil(dur * 4)
    if not segment_windows:
        assert ww and wins == fixed_windows(dur, WINDOW_S)
        idx = np.clip(((np.arange(L) + .5) / 4 // WINDOW_S).astype(int), 0, len(ww) - 1)
        curve = np.asarray([w["z"] for w in ww])[idx]
    else:
        curve = np.full(L, FILL_UNCOVERED)
        for w in ww:
            i0, i1 = max(0, int(math.floor(w["start"] * 4))), min(L, int(math.ceil(w["end"] * 4)))
            if i1 > i0:
                curve[i0:i1] = np.maximum(curve[i0:i1], w["z"])
    return dict(schema_version=1, method=method, dataset=row["dataset"], video_id=row["video_id"], duration=dur, native_rate=4,
                score_curve=curve.tolist(), intervals=[], error=None, seed=0, calls=1 + sum(len(r) for r in reads), code_path=code_path,
                extra=dict(z_video=ctx["global_margin"], stance=ctx["stance"], prefix_tokens=ctx.get("prefix_tokens"),
                           stance_cache_tokens=ctx["stance_cache_tokens"], windows=ww, standalone_seconds=seconds))
