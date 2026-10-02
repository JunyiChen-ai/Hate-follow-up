"""Temporal block-causal prefix states; ordinary joint query access thereafter."""
from contextlib import contextmanager
import numpy as np
import torch


def assign_groups(regions):
    local=np.stack(regions['local']).astype(bool)
    media=regions['visual']|regions['speech']
    count=local.sum(axis=0)
    groups=np.full(len(media),-1,dtype=np.int64)
    groups[media]=len(local)
    owned=count>0
    assert np.all(media[owned])
    groups[owned]=local[:,owned].argmax(axis=0)
    return groups,{'shared_tokens':int((count>1).sum()),'unassigned_media':int((media & ~owned).sum()),
        'group_counts':{str(int(g)):int((groups==g).sum()) for g in np.unique(groups)}}


def allowed_edges(groups,device='cpu',native=False):
    g=torch.as_tensor(groups,device=device);p=len(g)
    q=torch.arange(p,device=device)[:,None];k=torch.arange(p,device=device)[None,:]
    causal=k<=q
    if native:return causal
    # A scaffold key is safe because no scaffold row can read any media key.
    return causal & ((g[None,:]<0) | ((g[:,None]>=0) & (g[:,None]==g[None,:])))


class PrefixFactorizer:
    def __init__(self,judge):
        assert judge.family=='qwen3_vl'
        self.judge=judge;self.layers=judge.model.model.language_model.layers
        self.active=None;self.visited=[]
        self.hooks=[layer.self_attn.register_forward_pre_hook(self.hook(i),with_kwargs=True)
                    for i,layer in enumerate(self.layers)]

    def hook(self,index):
        def before(module,args,kwargs):
            if self.active is None:return
            hidden=kwargs.get('hidden_states',args[0] if args else None)
            assert hidden.shape[1]==self.active.shape[-1], 'prefix mask used on suffix query'
            self.visited.append(index)
            if 'attention_mask' in kwargs:kwargs={**kwargs,'attention_mask':self.active}
            else:
                args=list(args);assert len(args)>=3;args[2]=self.active;args=tuple(args)
            return args,kwargs
        return before

    @contextmanager
    def encoding(self,groups,native=False):
        assert self.active is None
        allowed=allowed_edges(groups,self.judge.device,native)
        mask=torch.zeros(allowed.shape,dtype=self.judge.dtype,device=self.judge.device)
        mask.masked_fill_(~allowed,torch.finfo(mask.dtype).min)
        self.active=mask[None,None];self.visited=[]
        try:yield
        finally:self.active=None
        assert self.visited==list(range(len(self.layers))),self.visited

    def close(self):
        assert self.active is None
        for hook in self.hooks:hook.remove()
