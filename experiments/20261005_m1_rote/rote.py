"""Interval RoTE in the actual native Qwen temporal subspace; no label access."""
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'CLAUDE.md').is_file())
sys.path.insert(0, str(ROOT))
SPEC = json.loads((Path(__file__).parent / 'spec.json').read_text())


def temporal_pairs(config):
    """Infer channels from the actual interleaved mRoPE routing, not a new theta."""
    head = config.head_dim
    section = config.rope_parameters['mrope_section']
    assert head % 2 == 0 and sum(section) == head // 2
    assert config.rope_parameters['rope_type'] == 'default'
    assert config.rope_parameters['mrope_interleaved'] is True
    native = torch.arange(head // 2)
    time = torch.ones(head // 2, dtype=torch.bool)
    for axis in (1, 2):
        time[axis:section[axis] * 3:3] = False
    pairs = native[time]
    assert len(pairs) == section[0]
    inv = 1 / config.rope_parameters['rope_theta'] ** (2 * native.float() / head)
    return pairs, inv[pairs]


def coefficients(intervals, frequencies):
    """Paper Eq6: mean-sinc normalization, with no radius remapping or clamp."""
    intervals = torch.as_tensor(intervals, dtype=torch.float64, device=frequencies.device)
    assert intervals.ndim == 2 and intervals.shape[1] == 2
    assert torch.isfinite(intervals).all() and (intervals[:, 1] >= intervals[:, 0]).all()
    center = SPEC['gamma'] * intervals.mean(-1)
    radius = SPEC['gamma'] * (intervals[:, 1] - intervals[:, 0]) / 2
    phase = center[:, None] * frequencies.double()[None]
    sinc = torch.sinc(radius[:, None] * frequencies.double()[None] / torch.pi)
    norm = sinc.mean(-1, keepdim=True)
    assert torch.isfinite(norm).all() and (norm > 0).all(), 'RoTE normalization domain unavailable'
    return (phase.cos() * sinc / norm).float(), (phase.sin() * sinc / norm).float(), norm[:, 0]


def rotate_pairs(x, cosine, sine):
    """Selected Qwen pairs are stored [first half, second half]."""
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1)


def token_sources(tokenizer, text, spans, expected_ids):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded['input_ids'] == list(expected_ids), 'source token seam mismatch'
    result = []
    for token, (lo, hi) in enumerate(encoded['offset_mapping']):
        source_ids = [sid for start, end, sid in spans if hi > start and lo < end]
        if source_ids:
            result.append(dict(token=token, source_ids=sorted(set(source_ids))))
    return result


def prefix_sources(j, text, enc, segments):
    pieces = text.split(j.processor.image_token)
    assert len(pieces) == len(j.img_tokens) + 1
    expanded = pieces[0]
    for piece, count in zip(pieces[1:], j.img_tokens):
        expanded += j.processor.image_token * count + piece
    cursor = expanded.index('\nTranscript:\n') + len('\nTranscript:\n')
    spans = []
    for sid, (start, end, words) in enumerate(segments):
        if not words.strip():
            continue
        line = f'[{start:.1f}s-{end:.1f}s] {words.strip()}'
        a = expanded.index(line, cursor)
        b = a + len(line)
        spans.append((a + len(f'[{start:.1f}s-{end:.1f}s] '), b, sid))
        cursor = b
    return token_sources(j.tok, expanded, spans, enc['input_ids'][0].tolist())


def window_parts(segments, start, end):
    """Exact native proportional word selection, retaining original segment IDs."""
    parts = []
    for sid, (a, b, text) in enumerate(segments):
        lo, hi = max(a, start), min(b, end)
        if hi <= lo or not text:
            continue
        if a >= start and b <= end:
            piece = text
        else:
            words = text.split()
            if not words:
                continue
            left = int(round((lo-a)/(b-a)*len(words)))
            right = int(round((hi-a)/(b-a)*len(words)))
            piece = ' '.join(words[left:right]).strip()
        if piece:
            parts.append((piece, sid))
    return parts


def question_sources(j, suffix, ids, segments, start, end, body):
    parts = window_parts(segments, start, end)
    joined = ' '.join(p for p, _ in parts)
    assert joined == body
    marker = 'Judge only what is spoken in this window: '
    body_start = suffix.index(marker) + len(marker)
    stripped = body.strip()
    assert suffix[body_start:body_start+len(stripped)] == stripped
    trim = len(body) - len(body.lstrip())
    cursor = 0
    spans = []
    for piece, sid in parts:
        a = max(0, cursor - trim)
        b = min(len(stripped), cursor + len(piece) - trim)
        if b > a:
            spans.append((body_start+a, body_start+b, sid))
        cursor += len(piece) + 1
    return token_sources(j.tok, suffix, spans, ids)


def source_intervals(mapping, segments):
    return [[min(segments[i][0] for i in row['source_ids']),
             max(segments[i][1] for i in row['source_ids'])] for row in mapping]


class IntervalAttention:
    """Capture native pre-rotation keys, intervene only inside independent S."""
    def __init__(self, j):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        self.j = j
        self.native_attention = sdpa_attention_forward
        self.config = j.model.config.text_config
        self.pairs, frequencies = temporal_pairs(self.config)
        self.frequencies = frequencies.to(j.device)
        self.channels = torch.cat((self.pairs, self.pairs + self.config.head_dim//2)).to(j.device)
        assert torch.equal(j.model.model.language_model.rotary_emb.inv_freq[self.pairs.to(j.device)], self.frequencies)
        self.active = False
        self.capture = False
        self.prefix_keys = {}
        self.q = {}
        self.k = {}
        self.hooks = []
        self.layers = list(j.model.model.language_model.layers)
        self.original_implementation = self.config._attn_implementation
        assert self.original_implementation == 'sdpa'
        ALL_ATTENTION_FUNCTIONS.register('interval_rote', self.dispatch)
        self.config._attn_implementation = 'interval_rote'
        for number, layer in enumerate(self.layers):
            self.hooks.append(layer.self_attn.q_norm.register_forward_hook(
                lambda module, inputs, output, n=number: self.capture_q(n, output)))
            self.hooks.append(layer.self_attn.k_norm.register_forward_hook(
                lambda module, inputs, output, n=number: self.capture_k(n, output)))

    def capture_q(self, layer, output):
        if self.active:
            self.q[layer] = output.index_select(-1, self.channels).transpose(1, 2).detach()

    def capture_k(self, layer, output):
        if self.capture:
            self.prefix_keys[layer] = output.index_select(1, self.prefix_indices).index_select(-1, self.channels).transpose(1, 2).detach()
        if self.active:
            self.k[layer] = output.index_select(1, self.body_indices).index_select(-1, self.channels).transpose(1, 2).detach()

    @contextmanager
    def native_prefix(self, segments):
        original = self.j.prefix_cache
        def captured(enc):
            self.prefix_mapping = prefix_sources(self.j, self.j._prefix_text, enc, segments)
            self.prefix_indices = torch.tensor([r['token'] for r in self.prefix_mapping], dtype=torch.long, device=self.j.device)
            self.prefix_intervals = source_intervals(self.prefix_mapping, segments)
            self.prefix_keys.clear()
            self.capture = True
            try:
                return original(enc)
            finally:
                self.capture = False
        self.j.prefix_cache = captured
        try:
            yield
        finally:
            self.j.prefix_cache = original
            self.capture = False

    @contextmanager
    def speech(self, cache_length, mapping, segments, start, end):
        assert not self.active and len(self.prefix_keys) == len(self.layers)
        self.cache_length = cache_length
        self.body_mapping = mapping
        self.body_indices = torch.tensor([r['token'] for r in mapping], dtype=torch.long, device=self.j.device)
        intervals = self.prefix_intervals + source_intervals(mapping, segments)
        self.key_indices = torch.cat((self.prefix_indices, self.body_indices + cache_length))
        assert len(set(self.key_indices.tolist())) == len(self.key_indices)
        self.cos, self.sin, self.norm = coefficients(np.asarray(intervals).reshape(-1, 2), self.frequencies)
        phase = SPEC['gamma'] * (start+end)/2 * self.frequencies
        self.query_cos, self.query_sin = phase.cos(), phase.sin()
        self.layer_calls = 0
        self.active = True
        try:
            yield
            assert self.layer_calls == len(self.layers)
        finally:
            self.active = False
            self.q.clear()
            self.k.clear()

    def dispatch(self, module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        if not self.active:
            return self.native_attention(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
        from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv
        layer = module.layer_idx
        assert dropout == 0. and scaling == module.scaling
        assert key.shape[2] == self.cache_length + query.shape[2]
        assert attention_mask is not None, 'cached suffix requires actual native causal mask'
        q0 = self.q.pop(layer)
        k0 = torch.cat((self.prefix_keys[layer], self.k.pop(layer)), 2)
        new_q = rotate_pairs(q0, self.query_cos, self.query_sin)
        new_k = rotate_pairs(k0, self.cos[None, None], self.sin[None, None])
        old_q = query.index_select(-1, self.channels).float()
        old_k = key.index_select(2, self.key_indices).index_select(-1, self.channels).float()
        new_k = repeat_kv(new_k, module.num_key_value_groups)
        old_k = repeat_kv(old_k, module.num_key_value_groups)
        delta = (new_q @ new_k.transpose(-1, -2) - old_q @ old_k.transpose(-1, -2)) * scaling
        bias = torch.zeros((*query.shape[:3], key.shape[2]), dtype=query.dtype, device=query.device)
        bias[:, :, :, self.key_indices] = delta.to(query.dtype)
        self.layer_calls += 1
        assert 'position_bias' not in kwargs
        return self.native_attention(module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, position_bias=bias, **kwargs)

    def close(self):
        assert not self.active and not self.capture
        for hook in self.hooks:
            hook.remove()
        self.config._attn_implementation = self.original_implementation
        self.prefix_keys.clear()
