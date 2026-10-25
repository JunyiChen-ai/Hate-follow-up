"""Candidate 50 (act-state reading, mask form A1): attention-access masks and verdict-span state transfer.

No labels, no GT, no hashes. Implementation tag c50-a1-v1 (2026-10-25).

Sources. Copied, not imported (archived code is never imported):
- `QueryAccess` and `window_keep` are copied from archive/experiments/20261002_m1_grounder/grounder.py (Grounder,
  2026-10-02), with two changes:
  (1) one mask for every text layer. The arms here use only two of Grounder's schedules: `base` (full causal mask at
      every layer) and `all_local` (restricted mask at every layer). The late / early layer split is dropped.
  (2) `window_keep` gains the verdict span [prefix_len, cached_len) for A1 / A1_other.
  The mask tensors are built exactly as in Grounder: bf16 additive mask, finfo.min, shape (1, 1, q, cached + q).
  So `base` and `all_local` replicate Grounder's computation token for token.
Shared modules, used unchanged:
- src/window_token_regions.py (token regions);
- src/pre_rotary_memory.py (`observe` = pre-rotary key / value capture, `rotate` = RoPE at chosen positions);
- src/stance_cache.py (`positions` = Qwen3-VL get_rope_index on the full id sequence).
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

import numpy as np
import torch

from src.pre_rotary_memory import observe, rotate
from src.stance_cache import positions

ARMS = ("base", "all_local", "A1", "A1_other")
RESTRICTED = {"base": False, "all_local": True, "A1": True, "A1_other": True}
SPAN_ARMS = ("A1", "A1_other")   # arms that also see the verdict span (verdict question turn + stance answer)


def window_keep(regions, i, cached_len, arm):
    """Columns of the cached sequence a branch of window i may attend to (its own query tokens are always visible).

    base:       everything.
    all_local:  scaffold (system, intro, rules, reader instruction, template tokens) + window i's prefix frame tokens
                (timestamp line + image) + window i's transcript tokens (proportional rule) - Grounder's all_local.
    A1, A1_other: all_local + the verdict span [P, cached_len) = the verdict question turn, the assistant header and
                the one-word stance answer with its end-of-turn tokens.
    """
    P = regions["prefix_len"]
    assert cached_len >= P and arm in ARMS
    keep = np.ones(cached_len, bool)
    if arm == "base":
        return keep
    keep[:P] = regions["scaffolding"] | regions["local"][i]
    keep[P:] = arm in SPAN_ARMS
    return keep


class QueryAccess:
    """Explicit masks in every arm keep the SDPA plumbing constant across arms (Grounder's design)."""

    def __init__(self, judge):
        assert judge.family == "qwen3_vl"
        self.judge = judge
        self.layers = judge.model.model.language_model.layers
        self.active = None
        self.visited = []
        self.hooks = [layer.self_attn.register_forward_pre_hook(self._hook(i), with_kwargs=True)
                      for i, layer in enumerate(self.layers)]

    def _hook(self, i):
        def hook(module, args, kwargs):
            if self.active is None:
                return
            self.visited.append(i)
            mask = self.active
            if "attention_mask" in kwargs:
                kwargs = {**kwargs, "attention_mask": mask}
            else:
                args = list(args)
                assert len(args) >= 3
                args[2] = mask
                args = tuple(args)
            return args, kwargs
        return hook

    @contextmanager
    def restrict(self, cached_len, query_len, keep, restricted):
        assert self.active is None
        assert len(keep) == cached_len
        device, dtype = self.judge.device, self.judge.dtype
        keys = torch.arange(cached_len + query_len, device=device)
        queries = torch.arange(query_len, device=device) + cached_len
        allowed = keys[None, :] <= queries[:, None]
        full = torch.zeros((query_len, cached_len + query_len), device=device, dtype=dtype)
        full.masked_fill_(~allowed, torch.finfo(dtype).min)
        if restricted:
            mask = full.clone()
            mask[:, :cached_len].masked_fill_(~torch.as_tensor(keep, device=device)[None, :], torch.finfo(dtype).min)
        else:
            assert bool(np.all(keep)), "unrestricted arm with a partial keep vector"
            mask = full
        self.active = mask[None, None]
        self.visited = []
        try:
            yield
        finally:
            self.active = None
        assert self.visited == list(range(len(self.layers))), self.visited

    def margin(self, cache, ids, keep, restricted, copy_cache=False):
        n = cache.get_seq_length()
        with self.restrict(n, len(ids), keep, restricted):
            value = self.judge.cached_margin(cache, ids, in_place=not copy_cache)
        if not copy_cache:
            cache.crop(n)
        assert cache.get_seq_length() == n
        return value

    def close(self):
        for hook in self.hooks:
            hook.remove()


# ---------------------------------------------------------------- verdict-span states

@contextmanager
def capture(j):
    """Pre-rotary keys (k_norm output) and values (v_proj output) of every text layer for ONE forward."""
    with observe(j) as (keys, values):
        yield keys, values


def join_forwards(parts):
    """parts: [(keys, values), ...] from consecutive forwards -> per layer (key, value), concatenated along tokens.
    Shapes per layer: (1, kv_heads, tokens, head_dim)."""
    n = len(parts[0][0])
    assert all(len(k) == len(v) == n for k, v in parts)
    return [(torch.cat([k[layer] for k, _ in parts], 2), torch.cat([v[layer] for _, v in parts], 2))
            for layer in range(n)]


def span_positions(j, enc, span_ids):
    """M-RoPE position ids (3, 1, len(span_ids)) of the verdict span that follows the prefix `enc`.

    Computed by get_rope_index on the full id sequence (src/stance_cache.positions) and checked against the
    incremental rule the model uses for cached forwards (text positions = cache index + rope_deltas, all three
    axes equal)."""
    P = int(enc["input_ids"].shape[1])
    ids = torch.cat((enc["input_ids"].to(j.device), torch.tensor([list(span_ids)], device=j.device)), 1)
    p, delta = positions(j, ids, enc["image_grid_thw"].to(j.device))
    span = p[:, :, P:]
    d = int(delta.reshape(-1)[0])
    text = (torch.arange(P, P + len(span_ids), device=span.device, dtype=span.dtype) + d)[None, None].expand(3, 1, -1)
    assert torch.equal(span, text), "verdict span positions are not plain text positions"
    live = getattr(j.model.model, "rope_deltas", None)
    if live is not None:
        assert int(live.reshape(-1)[0]) == d, ("rope_deltas of the live model differ from get_rope_index", int(live.reshape(-1)[0]), d)
    return span


def rotate_states(states, position, rotary):
    """Pre-rotary (key, value) per layer -> (key rotated to `position`, value). Values carry no position."""
    return [(rotate(k, position, rotary), v) for k, v in states]


def rotation_check(cache, P, rotated):
    """Own re-rotated span vs the cached span: (max |dk|, max |dk| / max |k|, max |dv|) over layers."""
    dk = dkr = dv = 0.0
    for layer, (k, v) in zip(cache.layers, rotated):
        ck, cv = layer.keys[:, :, P:], layer.values[:, :, P:]
        assert ck.shape == k.shape and cv.shape == v.shape, (ck.shape, k.shape, cv.shape, v.shape)
        diff = float((ck.float() - k.float()).abs().max())
        dk = max(dk, diff)
        dkr = max(dkr, diff / max(float(ck.float().abs().max()), 1e-12))
        dv = max(dv, float((cv.float() - v.float()).abs().max()))
    return dk, dkr, dv


def swapped_cache(cache, P, rotated):
    """Deep copy of `cache` whose verdict span [P, end) holds the given (rotated key, value) per layer."""
    out = copy.deepcopy(cache)
    assert len(out.layers) == len(rotated)
    for layer, (k, v) in zip(out.layers, rotated):
        assert layer.keys[:, :, P:].shape == k.shape and layer.values[:, :, P:].shape == v.shape
        layer.keys[:, :, P:] = k.to(layer.keys.dtype)
        layer.values[:, :, P:] = v.to(layer.values.dtype)
    return out


def donor_map(sign, ids):
    """Donor of each video: the next video in sorted id order (cyclic) of the same corpus with the same verdict sign.
    sign: video id -> bool (z_video > 0). Returns id -> donor id (None when the sign group has one video)."""
    out = {}
    order = sorted(ids)
    for s in (True, False):
        group = [v for v in order if sign[v] == s]
        for k, v in enumerate(group):
            out[v] = group[(k + 1) % len(group)] if len(group) > 1 else None
    return out
