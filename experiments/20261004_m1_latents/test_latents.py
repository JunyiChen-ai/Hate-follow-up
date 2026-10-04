"""CPU checks against real SDPA and a tiny real Qwen3-VL cache, not mock equations."""
import copy
import inspect
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from latents import (attention_probabilities, QueryAttention, support_indices,
                     wrong_support_indices, warmup, search, LatentReader)


def attention_checks():
    torch.manual_seed(3)
    for gqa in (False, True):
        q = torch.randn(1, 4, 5, 16)
        k = torch.randn(1, 2 if gqa else 4, 11, 16)
        v = torch.randn_like(k)
        for kind in ('bool', 'float', 'causal', 'none'):
            mask = torch.arange(11)[None, :] <= torch.arange(5)[:, None] + 6
            causal = kind == 'causal'
            if kind == 'float':
                mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, float('-inf'))
            elif kind in ('causal', 'none'):
                mask = None
            p = attention_probabilities(q, k, mask, causal, .3, gqa)
            expanded = v.repeat_interleave(2, dim=1) if gqa else v
            expected = p @ expanded
            actual = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal,
                                                    scale=.3, enable_gqa=gqa)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            capture = QueryAttention([1, 3], torch.tensor([0, 2, 4]), 5, 1)
            with capture.capture():
                got = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal,
                                                     scale=.3, enable_gqa=gqa)
            assert torch.equal(got, actual)
            torch.testing.assert_close(capture.relevance(), p[:, :, [1, 3], :][:, :, :, [0, 2, 4]].mean((0, 1, 2)))
    p, n = support_indices(torch.ones(24))
    assert p.tolist() == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert set(p.flatten().tolist()).isdisjoint(n.flatten().tolist())
    mapped, bindings = wrong_support_indices(torch.tensor([[0, 2], [3, 8]]), [3, 6])
    assert mapped.tolist() == [[3, 7], [0, 2]] and bindings[-1]['target_patch'] == 2
    print('ATTENTION_CHECKS_PASS all masks/GQA, exact kernel preservation, disjoint ties, content mappings')


def optimization_checks():
    torch.manual_seed(2)
    positive, negative = torch.randn(4, 2, 64), torch.randn(4, 4, 64)
    initial = positive.mean(1)
    h, losses = warmup(initial, positive, negative)
    assert losses[-1] < losses[0] and not h.requires_grad and not torch.equal(h, initial)
    evaluated = []
    def evaluate(state):
        evaluated.append(state.clone())
        return float(state[0, 0]), [], 0.
    best, trace = search(h, evaluate, torch.Generator().manual_seed(0))
    rewards = [float(t[0, 0]) for t in evaluated]
    assert trace['best_reward'] == max(rewards)
    assert torch.equal(best, evaluated[rewards.index(max(rewards))])
    assert len(evaluated) == 16
    _, tied = search(h, lambda _: (0., [], 0.), torch.Generator().manual_seed(0))
    assert tied['best_index'] == -1
    print('OPTIMIZATION_CHECKS_PASS Adam modifies inputs; selected state was evaluated; ties retain initial')


def cache_checks():
    from transformers import Qwen3VLConfig, Qwen3VLModel
    from src.mllm_judge import Judge
    config = Qwen3VLConfig(text_config=dict(vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        rope_scaling=dict(rope_type='default', mrope_section=[2, 3, 3], mrope_interleaved=True)),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
        out_hidden_size=64, deepstack_visual_indexes=[]))
    config._attn_implementation = 'sdpa'
    torch.manual_seed(5)
    model = Qwen3VLModel(config).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    W = torch.randn(128, 64)
    j = Judge.__new__(Judge)
    j.model = SimpleNamespace(model=model, get_output_embeddings=lambda: SimpleNamespace(weight=W))
    j.device, j.dtype, j.softcap = torch.device('cpu'), torch.float32, None
    j.yes_ids, j.no_ids = [1, 2], [3, 4]
    j.forward_params = set(inspect.signature(model.forward).parameters)
    ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
    positions = torch.arange(8)[None, None].expand(3, 1, -1)
    with torch.no_grad():
        out = model(input_ids=ids, position_ids=positions, attention_mask=torch.ones_like(ids), use_cache=True)
        cache = out.past_key_values
        model.rope_deltas = torch.zeros(1, 1, dtype=torch.long)
        delta = model.rope_deltas.clone()
        h = torch.randn(4, 64)
        reader = LatentReader(j)
        a = reader.evaluate(cache, h, delta)
        b = reader.evaluate(cache, h, delta, fresh=True)
        assert a == b and cache.get_seq_length() == 8
        full_inputs = torch.cat((model.get_input_embeddings()(ids), h[None]), dim=1)
        full_positions = torch.arange(12)[None, None].expand(3, 1, -1)
        full = model(inputs_embeds=full_inputs, position_ids=full_positions,
                     attention_mask=torch.ones(1, 12, dtype=torch.long), use_cache=False)
        reference = j.margins_fp32(full.last_hidden_state[0, -1:])[0]
        assert abs(a[2] - reference) < 2e-5, (a[2], reference)
        changed = reader.evaluate(cache, torch.roll(h, 1, 0), delta)
        assert changed[2] != a[2]
        assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
        # Nonzero rotary delta is also compared with a full pass using identical positions.
        shifted = torch.full((1, 1), -3, dtype=torch.long)
        # Prefix positions remain fixed; only the four appended latent positions use the delta.
        got = reader.evaluate(cache, h, shifted)
        full_positions = full_positions.clone()
        full_positions[:, :, 8:] -= 3
        full = model(inputs_embeds=full_inputs, position_ids=full_positions,
                     attention_mask=torch.ones(1, 12, dtype=torch.long), use_cache=False)
        reference = j.margins_fp32(full.last_hidden_state[0, -1:])[0]
        assert abs(got[2] - reference) < 2e-5
    print('CACHE_CHECKS_PASS real Qwen3 cached/fresh/full equivalence with zero/nonzero delta; slot intervention')


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    torch.set_num_threads(1)
    attention_checks()
    optimization_checks()
    cache_checks()
    print('CPU_CHECKS_DONE')
