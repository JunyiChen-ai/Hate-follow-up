"""Paper-defined VSV residual steering and normalized preceding-layer readout."""
import torch


def preserve_norm_update(hidden,direction,strength=.17):
    h=hidden.float();u=h+strength*direction.float()
    hn=h.norm(dim=-1,keepdim=True);un=u.norm(dim=-1,keepdim=True)
    candidate=u*hn/un.clamp_min(torch.finfo(torch.float32).tiny)
    return torch.where(un>0,candidate,h).to(hidden.dtype)


def margin(logits,n_yes):
    assert torch.isfinite(logits).all()
    return float(logits[:n_yes].logsumexp(0)-logits[n_yes:].logsumexp(0))


class ResidualReinforcer:
    def __init__(self,judge):
        assert judge.family=='qwen3_vl'
        self.judge=judge;self.lm=judge.model.model.language_model
        self.layers=self.lm.layers;assert len(self.layers)>=6
        self.layer_indices=list(range(len(self.layers)))
        self.sla_indices=list(range(len(self.layers)-6,len(self.layers)-1))
        self.active=False;self.directions=None;self.states=[];self.visited=[]
        self.norm_input=None;self.norm_output=None
        self.hooks=[layer.register_forward_hook(self.hook(i)) for i,layer in enumerate(self.layers)]
        self.hooks.append(self.lm.norm.register_forward_pre_hook(self.capture_norm_input))
        self.hooks.append(self.lm.norm.register_forward_hook(self.capture_norm_output))

    def capture_norm_input(self,module,args):
        if self.active:self.norm_input=args[0][0,-1].detach().clone()

    def capture_norm_output(self,module,args,output):
        if self.active:self.norm_output=output[0,-1].detach().clone()

    def hook(self,i):
        def after(module,args,output):
            if not self.active:return
            h=output[0] if isinstance(output,tuple) else output
            assert h.ndim==3 and h.shape[0]==1
            if self.directions is not None:
                h=preserve_norm_update(h,self.directions[i][None,None])
            self.states.append(h[0,-1].detach().clone());self.visited.append(i)
            if self.directions is not None:
                return (h,*output[1:]) if isinstance(output,tuple) else h
        return after

    @torch.no_grad()
    def read(self,cache,ids,directions=None,capture_states=True):
        assert not self.active
        if directions is not None:
            directions=torch.as_tensor(directions,dtype=torch.float32,device=self.judge.device)
            assert directions.shape==(len(self.layers),self.lm.config.hidden_size) and torch.isfinite(directions).all()
        n=cache.get_seq_length();self.active=True;self.directions=directions;self.states=[];self.visited=[]
        self.norm_input=None;self.norm_output=None
        try:h=self.judge._step(cache,ids)
        finally:self.active=False;self.directions=None;cache.crop(n)
        assert self.visited==self.layer_indices and len(self.states)==len(self.layers)
        states=torch.stack(self.states);self.states=[]
        assert self.norm_input is not None and self.norm_output is not None
        assert torch.equal(states[-1],self.norm_input),'last residual differs from actual finalNorm input'
        assert torch.equal(h,self.norm_output),'model output differs from actual finalNorm output'
        self.norm_input=None;self.norm_output=None
        labels=torch.tensor(self.judge.yes_ids+self.judge.no_ids,device=self.judge.device)
        final=self.judge._logits_fp32(h[None],labels)[0]
        # Intermediate states retain the original dtype for the shared final RMSNorm.
        previous=self.judge._logits_fp32(self.lm.norm(states[self.sla_indices]),labels).mean(0)
        # Same residual can round differently when normalized as one row rather
        # than in the native full-query tensor. The actual boundaries above must
        # still match bitwise; recomputation is diagnostic, never a score source.
        reconstructed_hidden=self.lm.norm(states[-1:])
        reconstructed=self.judge._logits_fp32(reconstructed_hidden,labels)[0]
        assert torch.isfinite(reconstructed).all()
        norm_diagnostic={'hidden_max':float((h-reconstructed_hidden[0]).abs().max()),
            'logit_max':float((final-reconstructed).abs().max())}
        mixed=.7*final+.3*previous
        return {'native_margin':margin(final,len(self.judge.yes_ids)),
            'full_margin':margin(mixed,len(self.judge.yes_ids)),
            'states':states.float().cpu() if capture_states else None,'final':final.cpu(),'previous':previous.cpu(),
            'layer_indices':self.visited[:],'sla_indices':self.sla_indices[:],
            'cache_length':n,'normalization_recompute':norm_diagnostic}

    def close(self):
        assert not self.active
        for hook in self.hooks:hook.remove()
