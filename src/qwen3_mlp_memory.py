"""Bound temporary feed-forward tensors without splitting attention or its cache."""
from contextlib import contextmanager
import torch


@contextmanager
def chunked_mlp(model,chunk_tokens=4096):
    """Apply each unchanged tokenwise MLP to smaller row batches during prefill."""
    assert chunk_tokens>0
    saved=[]
    def wrapper(original):
        def forward(hidden_states):
            if hidden_states.shape[-2]<=chunk_tokens:
                return original(hidden_states)
            return torch.cat([original(part) for part in hidden_states.split(chunk_tokens,dim=-2)],dim=-2)
        return forward
    try:
        for layer in model.language_model.layers:
            saved.append((layer.mlp,layer.mlp.forward))
            layer.mlp.forward=wrapper(layer.mlp.forward)
        yield
    finally:
        for module,original in saved: module.forward=original
