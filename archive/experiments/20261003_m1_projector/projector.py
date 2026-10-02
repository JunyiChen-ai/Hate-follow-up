"""Same-pass orthogonal attention-output guidance; no labels or parameter updates."""
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def orthogonal_update(ordinary,reference,gamma):
    o=ordinary.float();u=reference.float();delta=o-u
    unit=u/(u.norm(dim=-1,keepdim=True)+1e-8)
    correction=delta-(delta*unit).sum(-1,keepdim=True)*unit
    guided=(o+gamma*correction).to(ordinary.dtype)
    diagnostic=torch.stack((o.norm(dim=-1).mean(),u.norm(dim=-1).mean(),
        delta.norm(dim=-1).mean(),correction.norm(dim=-1).mean(),
        (correction*unit).sum(-1).abs().max()))
    return guided,diagnostic


def project_row(query,key,value,mask,scaling,visual,gamma,groups):
    key=key.repeat_interleave(groups,dim=1);value=value.repeat_interleave(groups,dim=1)
    weights=(query[:,:,-1:]@key.transpose(2,3))*scaling
    if mask is not None:
        mask=mask[:,:,-1:,:key.shape[-2]]
        weights=weights.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else weights+mask
    assert visual.ndim==1 and len(visual)==key.shape[-2]
    assert torch.isfinite(weights[...,visual]).all()
    ordinary=weights.softmax(-1,dtype=torch.float32).to(query.dtype)@value
    reference=weights.masked_fill(visual[None,None,None],float('-inf')).softmax(-1,dtype=torch.float32).to(query.dtype)@value
    assert torch.isfinite(ordinary).all() and torch.isfinite(reference).all()
    row,diag=orthogonal_update(ordinary,reference,gamma)
    return row.transpose(1,2).contiguous(),diag


class AttentionProjector:
    def __init__(self,judge):
        assert judge.family=='qwen3_vl'
        self.judge=judge;layers=judge.model.model.language_model.layers
        assert all(l.self_attn.config._attn_implementation=='sdpa' for l in layers)
        self.selected={id(l.self_attn):i for i,l in enumerate(layers)}
        self.layer_indices=list(range(len(layers)));self.active=None;self.visited=[];self.diagnostics=[]
        self.original=ALL_ATTENTION_FUNCTIONS['sdpa'];self.wrapper=self.dispatch
        ALL_ATTENTION_FUNCTIONS['sdpa']=self.wrapper

    def dispatch(self,module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
        out,weights=self.original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,**kwargs)
        if self.active is None or id(module) not in self.selected:return out,weights
        visual,gamma,n,q=self.active
        assert not module.training and dropout==0 and kwargs.get('position_bias') is None
        assert query.shape[2]==q and key.shape[2]==n+q and scaling is not None
        assert attention_mask is not None or q==1
        row,diag=project_row(query,key,value,attention_mask,scaling,visual,gamma,module.num_key_value_groups)
        out=out.clone();out[:,-1:]=row
        self.visited.append(self.selected[id(module)]);self.diagnostics.append(diag)
        return out,weights

    @torch.no_grad()
    def logits(self,cache,ids,visual,gamma):
        assert self.active is None and gamma in (0.,1.4)
        n=cache.get_seq_length();visual=torch.as_tensor(visual,dtype=torch.bool,device=self.judge.device)
        assert visual.ndim==1 and len(visual)<=n and visual.any()
        visual=torch.cat((visual,torch.zeros(n+len(ids)-len(visual),dtype=torch.bool,device=visual.device)))
        self.active=(visual,gamma,n,len(ids));self.visited=[];self.diagnostics=[]
        try:h=self.judge._step(cache,ids)
        finally:self.active=None;cache.crop(n)
        assert self.visited==self.layer_indices and cache.get_seq_length()==n
        label_ids=torch.tensor(self.judge.yes_ids+self.judge.no_ids,device=self.judge.device)
        logits=self.judge._logits_fp32(h[None],label_ids)[0].cpu()
        diag=torch.stack(self.diagnostics).float().cpu().tolist();self.diagnostics=[]
        return logits,{'layer_indices':self.visited[:],'geometry':diag}

    def close(self):
        assert self.active is None and ALL_ATTENTION_FUNCTIONS['sdpa']==self.wrapper
        ALL_ATTENTION_FUNCTIONS['sdpa']=self.original
