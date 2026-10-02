"""Online, label-free routing of window-selective Qwen3-VL attention heads."""
from contextlib import contextmanager
import math
import numpy as np
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

ARMS=("base","select","all_heads","permuted_heads","shifted_support")


class HeadSelector:
    def __init__(self,judge):
        assert judge.family=="qwen3_vl"
        self.judge=judge
        self.layers=judge.model.model.language_model.layers
        self.attentions=[layer.self_attn for layer in self.layers]
        self.original=ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.active=None
        self.trace=[]
        self.visited=[]
        ALL_ATTENTION_FUNCTIONS.register("sdpa",self.forward)

    def close(self):
        assert self.active is None
        ALL_ATTENTION_FUNCTIONS.register("sdpa",self.original)

    def forward(self,module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
        if self.active is None or module not in self.attentions:
            return self.original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,**kwargs)
        a=self.active;layer=module.layer_idx;self.visited.append(layer)
        batch,heads,qlen,dim=query.shape
        assert batch==1 and qlen==a["query_len"] and key.shape[-2]==a["cached_len"]+qlen
        assert dropout==0 and kwargs.get("position_bias") is None
        full=a["full"].expand(batch,heads,qlen,key.shape[-2])
        if a["arm"] in ("permuted_heads","shifted_support"):
            selected=a["replay"][layer][None,:]
            if a["arm"]=="permuted_heads":selected=torch.roll(selected,heads//2,dims=-1)
        elif a["arm"]=="all_heads":
            selected=torch.full((batch,heads),bool(a["n_local"] and a["n_remote"]),device=query.device,dtype=torch.bool)
        elif a["arm"]=="select" and a["n_local"] and a["n_remote"]:
            # One query row only. Keep GQA groups explicit, without repeating K.
            groups=heads//key.shape[1]
            assert heads%key.shape[1]==0
            q=query[:,:,-1,:].float().reshape(batch,key.shape[1],groups,dim)
            k=key[:,:,:a["prefix_len"],:].float()
            affinity=torch.einsum("bhgd,bhkd->bhgk",q,k).reshape(batch,heads,-1)
            affinity=affinity*(scaling if scaling is not None else dim**-.5)
            loc=torch.logsumexp(affinity.masked_fill(~a["local"][None,None,:],-torch.inf),dim=-1)-math.log(a["n_local"])
            rem=torch.logsumexp(affinity.masked_fill(~a["remote"][None,None,:],-torch.inf),dim=-1)-math.log(a["n_remote"])
            selected=loc>rem
        else:
            selected=torch.zeros((batch,heads),device=query.device,dtype=torch.bool)
        self.trace.append(selected[0].detach().clone())
        # The complete observed question selects heads. Its attention edges still
        # respect causality; no claim of autoregressive routing independence.
        blocked=selected[:,:,None,None]&a["blocked"][None,None,None,:]
        mask=full.masked_fill(blocked,torch.finfo(query.dtype).min)
        return self.original(module,query,key,value,mask,dropout=dropout,scaling=scaling,**kwargs)

    @contextmanager
    def route(self,cached_len,query_len,regions,index,arm,replay=None):
        assert self.active is None and arm in ARMS
        P=regions["prefix_len"];assert P<=cached_len
        device,dtype=self.judge.device,self.judge.dtype
        local=np.asarray(regions["local"][index],bool)
        media=regions["visual"]|regions["speech"]
        remote=media&~local
        support=regions["shifted"][index] if arm=="shifted_support" else local
        blocked=np.zeros(cached_len+query_len,bool);blocked[:P]=media&~support
        # Global Q/A, scaffolding and own query tokens are never blocked here.
        keys=torch.arange(cached_len+query_len,device=device)
        queries=torch.arange(query_len,device=device)+cached_len
        full=torch.zeros((1,1,query_len,cached_len+query_len),device=device,dtype=dtype)
        full.masked_fill_((keys[None,:]>queries[:,None])[None,None,:,:],torch.finfo(dtype).min)
        if arm in ("permuted_heads","shifted_support"):
            assert replay is not None and tuple(replay.shape)==(len(self.layers),self.layers[0].self_attn.config.num_attention_heads)
            replay=replay.to(device=device,dtype=torch.bool)
        self.active={"cached_len":cached_len,"query_len":query_len,"prefix_len":P,"arm":arm,
                     "n_local":int(local.sum()),"n_remote":int(remote.sum()),"replay":replay,
                     "local":torch.as_tensor(local,device=device),"remote":torch.as_tensor(remote,device=device),
                     "blocked":torch.as_tensor(blocked,device=device),"full":full}
        self.trace=[];self.visited=[]
        try:yield
        finally:self.active=None
        assert self.visited==list(range(len(self.layers))),self.visited

    def margin(self,cache,ids,regions,index,arm,replay=None,copy_cache=False):
        n=cache.get_seq_length()
        with self.route(n,len(ids),regions,index,arm,replay):
            z=self.judge.cached_margin(cache,ids,in_place=not copy_cache)
        if not copy_cache:cache.crop(n)
        assert cache.get_seq_length()==n
        return z,torch.stack(self.trace)
