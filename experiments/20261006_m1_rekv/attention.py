"""Scoped Qwen attention for retrieval; native prefix and S stay untouched."""
from contextlib import contextmanager
import types
import torch
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv
from retrieval import mean_query, remote_selection
from layout import pack_positions, rotate_key, suffix_positions, causal_mask


@contextmanager
def attention_scope(j, factory):
    layers = j.model.model.language_model.layers
    saved = []
    try:
        for layer in layers:
            att = layer.self_attn
            previous = att.__dict__.get('forward')
            saved.append((att, previous))
            att.forward = types.MethodType(factory(att.layer_idx), att)
        yield
    finally:
        for att, previous in saved:
            if previous is None:
                att.__dict__.pop('forward', None)
            else:
                att.forward = previous


@contextmanager
def capture_source(j):
    """Observe actual normalized current K and projected V without changing them."""
    keys, values, hooks = {}, {}, []
    layers = j.model.model.language_model.layers
    try:
        for index, layer in enumerate(layers):
            def key_hook(module, args, output, index=index):
                assert index not in keys
                keys[index] = output.transpose(1, 2).detach().clone()
            def value_hook(module, args, output, index=index, att=layer.self_attn):
                assert index not in values
                values[index] = output.reshape(*output.shape[:-1], -1, att.head_dim).transpose(1, 2).detach().clone()
            hooks.append(layer.self_attn.k_norm.register_forward_hook(key_hook))
            hooks.append(layer.self_attn.v_proj.register_forward_hook(value_hook))
        yield keys, values
        assert len(keys) == len(values) == len(layers)
    finally:
        for hook in hooks:
            hook.remove()


def retrieval_factory(j, memory, native_cache, ctx, local_ids, question_rows, trace):
    """Selection uses incoming unrotated Q; final query does not mutate memory."""
    rotary = j.model.model.language_model.rotary_emb
    # Selection and subsequent CPU proof replay use the same FP32 backend.
    # The per-layer host transfer/synchronization is part of new-V timing.
    representatives = memory.representative_layers('cpu')
    source_indices = [block['source_index'] for block in memory.blocks]
    assert local_ids and all(0 <= i < len(source_indices) for i in local_ids)

    def factory(layer_index):
        def forward(att, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
            assert past_key_values is native_cache and hidden_states.shape[0] == 1
            shape = (*hidden_states.shape[:-1], -1, att.head_dim)
            query = att.q_norm(att.q_proj(hidden_states).view(shape)).transpose(1, 2)
            key = att.k_norm(att.k_proj(hidden_states).view(shape)).transpose(1, 2)
            value = att.v_proj(hidden_states).view(shape).transpose(1, 2)
            pooled = mean_query(query, question_rows, key.shape[1])
            remote, scores = remote_selection(pooled.detach().to('cpu'), representatives[layer_index], source_indices, local_ids)
            selected = sorted(set(local_ids + remote), key=lambda index: source_indices[index])
            original_positions = [memory.blocks[index]['positions'] for index in selected]
            packed, end = pack_positions(original_positions, ctx['stance_cache_logical_start'])
            source_keys, source_values = [], []
            for index, positions in zip(selected, packed):
                source_key, source_value, old_positions = memory.layer(index, layer_index, hidden_states.device)
                assert torch.equal(old_positions.to('cpu'), memory.blocks[index]['positions'])
                source_keys.append(rotate_key(source_key, positions, rotary))
                source_values.append(source_value)
            current_positions = suffix_positions(hidden_states.shape[1], end, hidden_states.device)
            cos, sin = rotary(query, current_positions)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            prefix = native_cache.layers[layer_index]
            assert prefix.keys.shape[2] == ctx['stance_cache_tokens']
            all_key = torch.cat([prefix.keys, *source_keys, key], dim=2)
            all_value = torch.cat([prefix.values, *source_values, value], dim=2)
            past = all_key.shape[2] - hidden_states.shape[1]
            mask = causal_mask(hidden_states.shape[1], past, hidden_states.device)
            output = scaled_dot_product_attention(query,
                repeat_kv(all_key, att.num_key_value_groups),
                repeat_kv(all_value, att.num_key_value_groups),
                attn_mask=mask, dropout_p=0., is_causal=False, scale=att.scaling)
            output = output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()
            assert layer_index not in trace
            trace[layer_index] = dict(local_ids=list(local_ids),remote_ids=remote,selected_ids=selected,
                query_vector=pooled.detach().float().to('cpu'),
                similarities=scores.detach().float().to('cpu'),
                source_translations=[int(new.min())-int(old.min()) for old,new in zip(original_positions,packed)],
                question_logical_start=end,
                prefix_tokens=ctx['stance_cache_tokens'],source_tokens=sum(key.shape[2] for key in source_keys),
                suffix_tokens=hidden_states.shape[1],source_times=[memory.blocks[index]['actual_time'] for index in selected])
            return att.o_proj(output), None
        return forward
    return factory
