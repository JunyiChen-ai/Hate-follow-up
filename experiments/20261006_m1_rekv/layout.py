"""Source position translation and rectangular causal attention geometry."""
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half


def pack_positions(block_positions, start):
    """One integer translation for all axes; preserve within-block geometry."""
    packed = []
    cursor = int(start)
    assert cursor >= 0
    for original in block_positions:
        assert original.ndim == 3 and original.shape[:2] == (3, 1) and original.shape[2] > 0
        assert original.dtype == torch.long
        position = original - int(original.min()) + cursor
        assert torch.equal(position - original, torch.full_like(original, cursor - int(original.min())))
        packed.append(position)
        cursor = int(position.max()) + 1
    return packed, cursor


def rotate_key(key, position, rotary):
    """Use stored pre-RoPE keys; no approximate inverse/round-trip rotation."""
    assert key.ndim == 4 and key.shape[0] == 1
    assert position.shape == (3, 1, key.shape[2])
    cos, sin = rotary(key, position.to(key.device))
    assert cos.shape == sin.shape == (1, key.shape[2], key.shape[3])
    return key * cos.unsqueeze(1) + rotate_half(key) * sin.unsqueeze(1)


def suffix_positions(length, start, device):
    assert length > 0 and start >= 0
    return (torch.arange(length, device=device, dtype=torch.long) + int(start))[None, None].expand(3, 1, -1)


def causal_mask(query_length, past_length, device):
    """All past is visible; suffix attends only to earlier/equal suffix rows."""
    assert query_length > 0 and past_length >= 0
    rows = torch.arange(query_length, device=device)[:, None]
    columns = torch.arange(past_length + query_length, device=device)[None, :]
    return (columns <= past_length + rows)[None, None]
