"""Paper-defined VAR transfer, with explicit first-token Qwen3 sink channels."""
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def sink_mask(hidden, channels):
    x=hidden.float()
    rms=x.square().mean(-1).sqrt()
    peak=x[...,channels].abs().amax(-1)
    phi=torch.where(rms>0,peak/rms.clamp_min(torch.finfo(x.dtype).tiny),0.)
    return phi>=20


def recycle_attention(query,key,value,mask,scaling,visual,sinks,groups,fraction):
    """All suffix rows, GQA, original BF16 QK/V and FP32 probability algebra."""
    key=key.repeat_interleave(groups,dim=1);value=value.repeat_interleave(groups,dim=1)
    logits=torch.matmul(query,key.transpose(2,3))*scaling
    if mask is not None:
        mask=mask[:,:,:query.shape[-2],:key.shape[-2]]
        if mask.dtype==torch.bool:logits=logits.masked_fill(~mask,torch.finfo(logits.dtype).min)
        else:logits=logits+mask
    prob=logits.softmax(-1,dtype=torch.float32)
    non_sink_visual=visual&~sinks
    visual_mass=prob[...,visual].sum(-1,keepdim=True)
    support_mass=prob[...,non_sink_visual].sum(-1,keepdim=True)
    wasted_mass=prob[...,sinks].sum(-1,keepdim=True)
    selected=(visual_mass>=.2)&(support_mass>=.5*visual_mass)&(support_mass>0)&(wasted_mass>0)
    removed=torch.where(selected,fraction*wasted_mass,0.)
    updated=prob*(1.-fraction*(selected&sinks))
    updated=updated+prob*non_sink_visual*removed/support_mass.clamp_min(torch.finfo(prob.dtype).tiny)
    # The masks preserve zero-probability causal entries, including future suffix keys.
    stats=torch.stack((visual_mass.mean(),support_mass.mean(),wasted_mass.mean(),
        selected.float().mean(),removed.mean(),(updated.sum(-1)-1).abs().max()))
    output=torch.matmul(updated.to(query.dtype),value).transpose(1,2).contiguous()
    return output,stats


class AttentionRecycler:
    def __init__(self,judge):
        assert judge.family=='qwen3_vl'
        self.judge=judge;layers=judge.model.model.language_model.layers
        assert len(layers)>1 and all(l.self_attn.config._attn_implementation=='sdpa' for l in layers)
        self.layer_indices=list(range(len(layers)-1))
        self.selected={id(layers[i].self_attn):i for i in self.layer_indices}
        self.capture=False;self.active=None;self.channels={};self.cached={};self.current={}
        self.visited=[];self.stats=[]
        self.hooks=[layers[i].register_forward_pre_hook(self.make_hook(i),with_kwargs=True) for i in self.layer_indices]
        self.original=ALL_ATTENTION_FUNCTIONS['sdpa'];self.wrapper=self.dispatch
        ALL_ATTENTION_FUNCTIONS['sdpa']=self.wrapper

    def make_hook(self,i):
        def hook(module,args,kwargs):
            if not self.capture and self.active is None:return
            hidden=args[0] if args else kwargs['hidden_states']
            assert hidden.ndim==3 and hidden.shape[0]==1
            if self.capture:
                if i not in self.channels:
                    # Stable descending sort leaves tied channels in increasing index order.
                    self.channels[i]=torch.argsort(hidden[0,0].float().abs(),descending=True,stable=True)[:2]
                mask=sink_mask(hidden,self.channels[i])[0]
                self.cached[i]=torch.cat((self.cached[i],mask)) if i in self.cached else mask
            else:
                assert i in self.channels and i not in self.current
                self.current[i]=sink_mask(hidden,self.channels[i])[0]
        return hook

    def begin_capture(self):
        assert not self.capture and self.active is None
        self.channels={};self.cached={};self.current={};self.capture=True

    def end_capture(self,n):
        assert self.capture and set(self.cached)==set(self.layer_indices)
        assert all(len(v)==n for v in self.cached.values())
        self.capture=False

    def dispatch(self,module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
        if self.active is None or id(module) not in self.selected:
            return self.original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,**kwargs)
        visual,fraction,n,q=self.active;i=self.selected[id(module)]
        assert not module.training and dropout==0 and kwargs.get('position_bias') is None
        assert query.shape[2]==q and key.shape[2]==n+q and scaling is not None
        assert attention_mask is not None or q==1
        assert len(self.cached[i])==n and len(self.current[i])==q
        sinks=torch.cat((self.cached[i],self.current[i]))
        out,stats=recycle_attention(query,key,value,attention_mask,scaling,visual,sinks,module.num_key_value_groups,fraction)
        self.visited.append(i);self.stats.append(stats)
        return out,None

    @torch.no_grad()
    def margin(self,cache,ids,visual,fraction):
        assert self.active is None and not self.capture and fraction in (0.,.6)
        n=cache.get_seq_length();visual=torch.as_tensor(visual,dtype=torch.bool,device=self.judge.device)
        assert visual.ndim==1 and len(visual)<=n and visual.any()
        visual=torch.cat((visual,torch.zeros(n+len(ids)-len(visual),dtype=torch.bool,device=visual.device)))
        self.active=(visual,fraction,n,len(ids));self.visited=[];self.stats=[];self.current={}
        try:h=self.judge._step(cache,ids)
        finally:self.active=None;cache.crop(n)
        assert self.visited==self.layer_indices and cache.get_seq_length()==n
        label_ids=torch.tensor(self.judge.yes_ids+self.judge.no_ids,device=self.judge.device)
        logits=self.judge._logits_fp32(h[None],label_ids)[0];ny=len(self.judge.yes_ids)
        z=float(torch.logsumexp(logits[:ny],0)-torch.logsumexp(logits[ny:],0))
        stats=torch.stack(self.stats).float().cpu().tolist();self.stats=[];self.current={}
        return z,{'layer_indices':self.visited[:],'statistics':stats}

    def close(self):
        assert self.active is None and not self.capture and ALL_ATTENTION_FUNCTIONS['sdpa']==self.wrapper
        ALL_ATTENTION_FUNCTIONS['sdpa']=self.original
        for h in self.hooks:h.remove()
