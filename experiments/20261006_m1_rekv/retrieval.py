"""Unscored internal media retrieval; head mapping and ties are explicit."""
import json
from pathlib import Path
import sys
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'CLAUDE.md').is_file())
sys.path.insert(0, str(ROOT))
SPEC = json.loads((Path(__file__).parent / 'spec.json').read_text())


def mean_query(query, rows, kv_heads):
    """Post-q_norm/pre-RoPE [1,Q,L,D] -> concatenated GQA-group mean."""
    assert query.ndim == 4 and query.shape[0] == 1
    heads, length, dimension = query.shape[1:]
    assert heads % kv_heads == 0 and rows
    assert len(set(rows)) == len(rows) and all(0 <= r < length for r in rows)
    grouped = query.float().reshape(1, kv_heads, heads // kv_heads, length, dimension)
    return grouped[:, :, :, rows].mean(2).mean(2).reshape(kv_heads * dimension)


def mean_image_key(key, rows):
    """Post-k_norm/pre-RoPE [1,KV,L,D], only actual image-token rows."""
    assert key.ndim == 4 and key.shape[0] == 1 and rows
    assert len(set(rows)) == len(rows) and all(0 <= r < key.shape[2] for r in rows)
    return key[0, :, rows].float().mean(1).reshape(-1)


def cosine_scores(query, keys):
    assert query.ndim == 1 and keys.ndim == 2 and keys.shape[1] == len(query)
    q, k = query.float(), keys.float()
    denominator = torch.linalg.vector_norm(k, dim=1) * torch.linalg.vector_norm(q)
    numerator = k @ q
    # Avoid a divide-by-zero even in the branch that torch.where discards.
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    scores = torch.where(denominator > 0, numerator / safe, torch.zeros_like(numerator))
    assert torch.isfinite(scores).all()
    return scores


def remote_selection(query, representatives, source_indices, local_ids, budget=None):
    """Indices refer to chronological source blocks, not classes or margins."""
    budget = SPEC['remote_frames_per_layer'] if budget is None else budget
    assert len(source_indices) == len(representatives) and budget >= 0
    assert all(a < b for a, b in zip(source_indices, source_indices[1:]))
    assert all(0 <= i < len(source_indices) for i in local_ids)
    scores = cosine_scores(query, representatives)
    remote = [i for i in range(len(source_indices)) if i not in local_ids]
    if not remote or not budget:
        return [], scores
    # Chronological original indices provide the predeclared deterministic tie.
    ids = torch.tensor(remote, device=scores.device)
    order = torch.argsort(scores[ids], descending=True, stable=True)
    chosen = ids[order[:budget]].tolist()
    return chosen, scores
