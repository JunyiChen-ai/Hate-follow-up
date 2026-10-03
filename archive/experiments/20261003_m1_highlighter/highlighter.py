"""VGA-derived local visual-value guidance; frozen model, no label access."""
import math
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def salience(hidden, weight, batch=128):
    values=[]
    for h in hidden.split(batch):
        logits=h.float()@weight.float().T
        assert torch.isfinite(logits).all()
        lp=logits.topk(10,dim=-1).values-logits.logsumexp(-1,keepdim=True)
        # Zero probabilities contribute zero: lp itself stays finite.
        values.append(-(lp.exp()*lp).sum(-1)/math.log(10))
    out=torch.cat(values)
    assert torch.isfinite(out).all() and (out>=0).all()
    return out


def local_guidance(input_ids,image_id,counts,frames,wins,scores):
    positions=torch.where(input_ids==image_id)[0]
    assert len(counts)==len(frames) and sum(counts)==len(positions)==len(scores)
    assert (scores>=0).all() and torch.isfinite(scores).all()
    frame_ids=torch.repeat_interleave(torch.arange(len(frames)),torch.tensor(counts))
    for i,count in enumerate(counts):
        assert count>0
        block=positions[frame_ids==i]
        assert torch.equal(block,torch.arange(int(block[0]),int(block[0])+count))
    times=torch.tensor([f[0] for f in frames],dtype=torch.float64)[frame_ids]
    answer=[]
    for i,(start,end) in enumerate(wins):
        mask=(times>=start)&((times<=end) if i==len(wins)-1 else (times<end))
        idx=positions[mask];g=scores[mask].float()
        if len(g):g=g/g.sum() if g.sum()>0 else torch.ones_like(g)/len(g)
        answer.append((idx,g))
    return answer,positions,frame_ids


def guide_output(ordinary,direction,g,coefficient=.2):
    o=ordinary.float();d=direction.float()
    cosine=((o*d).sum(-1)/(o.norm(dim=-1).clamp_min(1e-8)*d.norm(dim=-1).clamp_min(1e-8))).clamp(-1,1)
    similarity=(1+cosine)/2
    total=similarity.sum(1,keepdim=True)
    gamma=torch.where(total>0,(2-similarity*o.shape[1]/total.clamp_min(1e-30)).relu(),torch.ones_like(similarity))
    fraction=(g>1e-8).float().mean()
    update=coefficient*fraction*gamma[...,None]*d
    guided=(o+update).to(ordinary.dtype)
    diagnostic=torch.stack((o.norm(dim=-1).mean(),d.norm(dim=-1).mean(),
        update.norm(dim=-1).mean(),gamma.mean(),gamma.min(),gamma.max(),fraction))
    return guided,diagnostic,gamma


class AttentionHighlighter:
    def __init__(self,judge):
        assert judge.family=='qwen3_vl'
        self.judge=judge;self.lm=judge.model.model.language_model
        layers=self.lm.layers;assert len(layers)==36
        assert all(l.self_attn.config._attn_implementation=='sdpa' for l in layers)
        self.layer_indices=list(range(4,18))
        self.selected={id(layers[i].self_attn):i for i in self.layer_indices}
        self.original=ALL_ATTENTION_FUNCTIONS['sdpa'];self.wrapper=self.dispatch
        ALL_ATTENTION_FUNCTIONS['sdpa']=self.wrapper
        self.active=None;self.visited=[];self.diagnostics=[];self.gammas=[]
        self.W32=judge.model.get_output_embeddings().weight.detach().float()

    @torch.no_grad()
    def prefix_cache(self,enc):
        mask=(enc['input_ids'][0]==self.judge.image_token_id).to(self.judge.device)
        captured=[]
        def capture(module,args,output):
            assert output.shape[:2]==(1,len(mask))
            captured.append(output[0,mask].detach().clone())
        handle=self.lm.norm.register_forward_hook(capture)
        try:cache=self.judge.prefix_cache(enc)
        finally:handle.remove()
        assert len(captured)==1 and len(captured[0])==int(mask.sum())
        return cache,captured[0]

    def dispatch(self,module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
        out,weights=self.original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,**kwargs)
        if self.active is None or id(module) not in self.selected:return out,weights
        indices,g,n,q,coefficient=self.active
        assert not module.training and dropout==0 and query.shape[2]==q and key.shape[2]==n+q
        assert int(indices.max())<n and (attention_mask is not None or q==1)
        # Local prefix positions precede every suffix row and are causally available.
        if attention_mask is not None:
            support=attention_mask[:,:,-1,indices]
            assert bool(support.all()) if support.dtype==torch.bool else bool(torch.isfinite(support).all() and (support==0).all())
        v=value[:,:,indices].float().repeat_interleave(module.num_key_value_groups,dim=1)
        direction=(v*g[None,None,:,None]).sum(2,keepdim=True)
        last=out[:,-1:].transpose(1,2)
        guided,diag,gamma=guide_output(last,direction,g,coefficient)
        out=out.clone();out[:,-1:]=guided.transpose(1,2)
        self.visited.append(self.selected[id(module)]);self.diagnostics.append(diag);self.gammas.append(gamma.squeeze(-1))
        return out,weights

    @torch.no_grad()
    def logits(self,cache,ids,indices,g,coefficient=.2):
        assert self.active is None and coefficient in (0.,.2)
        n=cache.get_seq_length();indices=indices.to(self.judge.device);g=g.to(self.judge.device)
        assert len(indices)==len(g) and (g>=0).all() and torch.isfinite(g).all()
        if len(g):assert abs(float(g.sum())-1)<1e-5
        self.visited=[];self.diagnostics=[];self.gammas=[]
        self.active=(indices,g,n,len(ids),coefficient) if len(g) else None
        try:h=self.judge._step(cache,ids)
        finally:self.active=None;cache.crop(n)
        assert self.visited==(self.layer_indices if len(g) else [])
        labels=torch.tensor(self.judge.yes_ids+self.judge.no_ids,device=self.judge.device)
        logits_gpu=self.judge._logits_fp32(h[None],labels)[0]
        ny=len(self.judge.yes_ids)
        margin=float(logits_gpu[:ny].logsumexp(0)-logits_gpu[ny:].logsumexp(0))
        logits=logits_gpu.cpu()
        diag={'layer_indices':self.visited[:],'margin':margin,
            'geometry':torch.stack(self.diagnostics).cpu().tolist() if self.diagnostics else [],
            'gamma':torch.stack(self.gammas).cpu().tolist() if self.gammas else []}
        self.diagnostics=[];self.gammas=[]
        return logits,diag

    def close(self):
        assert self.active is None and ALL_ATTENTION_FUNCTIONS['sdpa']==self.wrapper
        ALL_ATTENTION_FUNCTIONS['sdpa']=self.original
