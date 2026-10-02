"""Window-aligned access masks for frozen Qwen3-VL query layers. No labels."""
from contextlib import contextmanager
import math
import numpy as np
import torch
from src.window_token_regions import span_mask, token_regions

ARMS = ("base", "late", "verdict_only", "all_local", "shifted", "early")






class QueryAccess:
    """Explicit masks in every arm keep SDPA plumbing constant across controls."""
    def __init__(self, judge):
        assert judge.family == "qwen3_vl"
        self.judge=judge
        self.layers=judge.model.model.language_model.layers
        self.cut=len(self.layers)-math.ceil(len(self.layers)/4)
        self.active=None;self.visited=[]
        self.hooks=[layer.self_attn.register_forward_pre_hook(self._hook(i),with_kwargs=True)
                    for i,layer in enumerate(self.layers)]

    def _hook(self, i):
        def hook(module,args,kwargs):
            if self.active is None:return
            self.visited.append(i)
            full, restricted, arm=self.active
            use = (arm == "all_local" or (arm == "early" and i < len(self.layers)-self.cut)
                   or (arm not in ("base","early") and i >= self.cut))
            mask=restricted if use else full
            if "attention_mask" in kwargs:
                kwargs={**kwargs,"attention_mask":mask}
            else:
                args=list(args);assert len(args)>=3;args[2]=mask;args=tuple(args)
            return args,kwargs
        return hook

    @contextmanager
    def restrict(self, prefix_len, query_len, keep, arm):
        assert self.active is None and arm in ARMS
        assert len(keep)==prefix_len
        device,dtype=self.judge.device,self.judge.dtype
        keys=torch.arange(prefix_len+query_len,device=device)
        queries=torch.arange(query_len,device=device)+prefix_len
        allowed=keys[None,:] <= queries[:,None]
        full=torch.zeros((query_len,prefix_len+query_len),device=device,dtype=dtype)
        full.masked_fill_(~allowed,torch.finfo(dtype).min)
        local=full.clone()
        local[:,:prefix_len].masked_fill_(~torch.as_tensor(keep,device=device)[None,:],torch.finfo(dtype).min)
        self.active=(full[None,None],local[None,None],arm);self.visited=[]
        try:yield
        finally:self.active=None
        assert self.visited == list(range(len(self.layers))),self.visited

    def margin(self, cache, ids, keep, arm, copy_cache=False):
        n=cache.get_seq_length()
        with self.restrict(n,len(ids),keep,arm):
            value=self.judge.cached_margin(cache,ids,in_place=not copy_cache)
        if not copy_cache:cache.crop(n)
        assert cache.get_seq_length()==n
        return value

    def close(self):
        for hook in self.hooks:hook.remove()


def window_keep(regions, i, cached_len, arm):
    P=regions["prefix_len"]
    assert cached_len>=P
    keep=np.ones(cached_len,bool)
    if arm != "base":keep[P:]=False
    if arm in ("late","all_local","shifted","early"):
        local=regions["shifted" if arm == "shifted" else "local"][i]
        keep[:P]=regions["scaffolding"]|local
    return keep
