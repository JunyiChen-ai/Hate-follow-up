"""PAI-style last-query-row attention intervention; no training or labels."""
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def contrast_margin(amplified, reference, n_yes, gamma=1.1):
    logits=gamma*amplified-(gamma-1)*reference
    assert torch.isfinite(logits).all()
    return float(torch.logsumexp(logits[:n_yes],0)-torch.logsumexp(logits[n_yes:],0))


def amplify_row(query,key,value,mask,scaling,visual,alpha,groups):
    """GQA last-row computation, preserving the source BF16 matmul/FP32 softmax."""
    key=key.repeat_interleave(groups,dim=1)
    value=value.repeat_interleave(groups,dim=1)
    weights=torch.matmul(query[:,:,-1:],key.transpose(2,3))*scaling
    if mask is not None:
        mask=mask[:,:,-1:,:key.shape[-2]]
        if mask.dtype==torch.bool:weights=weights.masked_fill(~mask,torch.finfo(weights.dtype).min)
        else:weights=weights+mask
    selected=weights[...,visual]
    assert torch.isfinite(selected).all() and (selected>torch.finfo(weights.dtype).min/2).all()
    before=weights.softmax(-1,dtype=torch.float32)[...,visual].sum(-1).mean()
    weights[...,visual]=selected+alpha*selected.abs()
    prob=weights.softmax(-1,dtype=torch.float32)
    after=prob[...,visual].sum(-1).mean()
    row=torch.matmul(prob.to(query.dtype),value).transpose(1,2).contiguous()
    return row,torch.stack((before,after))


class ImageAmplifier:
    def __init__(self,judge):
        assert judge.family=='qwen3_vl'
        self.judge=judge;layers=judge.model.model.language_model.layers
        assert len(layers)>2 and all(l.self_attn.config._attn_implementation=='sdpa' for l in layers)
        self.selected={id(l.self_attn):i for i,l in enumerate(layers) if i>=2}
        self.layer_indices=list(range(2,len(layers)));self.active=None;self.visited=[];self.mass=[]
        self.original=ALL_ATTENTION_FUNCTIONS['sdpa']
        self.wrapper=self.dispatch
        ALL_ATTENTION_FUNCTIONS['sdpa']=self.wrapper

    def dispatch(self,module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
        out,weights=self.original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,**kwargs)
        if self.active is None or id(module) not in self.selected:return out,weights
        visual,alpha,n,q=self.active
        assert not module.training and dropout==0 and kwargs.get('position_bias') is None
        assert query.shape[2]==q and key.shape[2]==n+q and scaling is not None
        # With a cached multi-token suffix, transformers must supply an offset
        # causal mask; a mask-free upper-left causal kernel is not equivalent.
        assert attention_mask is not None or q==1
        row,mass=amplify_row(query,key,value,attention_mask,scaling,visual,alpha,module.num_key_value_groups)
        out=out.clone();out[:,-1:]=row
        self.visited.append(self.selected[id(module)]);self.mass.append(mass)
        return out,weights

    @torch.no_grad()
    def logits(self,cache,ids,visual,alpha):
        assert self.active is None and alpha in (0.,.5)
        n=cache.get_seq_length()
        visual=torch.as_tensor(visual,dtype=torch.bool,device=self.judge.device)
        assert visual.ndim==1 and len(visual)<=n and visual.any()
        visual=torch.cat((visual,torch.zeros(n+len(ids)-len(visual),dtype=torch.bool,device=visual.device)))
        self.active=(visual,alpha,n,len(ids));self.visited=[];self.mass=[]
        try:h=self.judge._step(cache,ids)
        finally:self.active=None;cache.crop(n)
        assert self.visited==self.layer_indices and cache.get_seq_length()==n
        label_ids=torch.tensor(self.judge.yes_ids+self.judge.no_ids,device=self.judge.device)
        logits=self.judge._logits_fp32(h[None],label_ids)[0].cpu()
        mass=torch.stack(self.mass).mean(0).float().tolist();self.mass=[]
        return logits,{'layer_indices':self.visited[:],'image_attention_mass_before_after':mass}

    def close(self):
        assert self.active is None and ALL_ATTENTION_FUNCTIONS['sdpa']==self.wrapper
        ALL_ATTENTION_FUNCTIONS['sdpa']=self.original
