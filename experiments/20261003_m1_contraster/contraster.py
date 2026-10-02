"""One-model depth contrast; no labels, parameter edits, or extra forward."""
import math
import torch


def contrast_logits(logits, layers, yes_ids, no_ids):
    """Rows are candidate layers followed by mature; full vocabulary in FP32."""
    assert logits.ndim == 2 and len(logits) == len(layers)+1
    lp = logits.log_softmax(-1)
    mature, early = lp[-1:], lp[:-1]
    mixture = torch.logaddexp(mature, early)-math.log(2.)
    js = .5*((mature.exp()*(mature-mixture)).sum(-1)+
             (early.exp()*(early-mixture)).sum(-1))
    assert torch.isfinite(js).all()
    index = int(js.argmax())
    ids = list(yes_ids)+list(no_ids); ny = len(yes_ids)
    labels = logits[:, ids]
    diff = labels[-1]-labels[index]
    margin = torch.logsumexp(diff[:ny],0)-torch.logsumexp(diff[ny:],0)
    logdiff = lp[-1, ids]-lp[index, ids]
    alternative = torch.logsumexp(logdiff[:ny],0)-torch.logsumexp(logdiff[ny:],0)
    return float(margin), {'layers':list(layers),'selected_layer':layers[index],
        'selected_index':index,'js':js.tolist(),'label_logits':labels.tolist(),
        'log_normalizers':torch.logsumexp(logits,-1).tolist(),
        'logprob_equivalence_error':float(abs(margin-alternative)),
        'yes_ids':list(yes_ids),'no_ids':list(no_ids)}


class DepthContraster:
    def __init__(self,judge):
        assert judge.family=='qwen3_vl'
        self.judge=judge;lm=judge.model.model.language_model
        self.layers=list(range(2,len(lm.layers),2));assert self.layers
        self.norm=lm.norm;self.active=False;self.rows={}
        self.head=judge.model.get_output_embeddings().weight.detach().float()
        self.handles=[lm.layers[n-1].register_forward_hook(self.hook(n)) for n in self.layers]

    def hook(self,n):
        def capture(module,args,output):
            if self.active:
                h=output[0] if isinstance(output,(tuple,list)) else output
                assert n not in self.rows
                self.rows[n]=h[0,-1].detach().clone()
        return capture

    @torch.no_grad()
    def forward(self,cache,ids):
        assert not self.active;self.rows={};self.active=True
        try:h=self.judge._step(cache,ids)
        finally:self.active=False
        assert list(self.rows)==self.layers
        return h

    @torch.no_grad()
    def readout(self,h):
        assert list(self.rows)==self.layers
        early=self.norm(torch.stack([self.rows[n] for n in self.layers]))
        states=torch.cat((early,h[None]),0)
        logits=states.float() @ self.head.T
        self.rows={}
        return contrast_logits(logits,self.layers,self.judge.yes_ids,self.judge.no_ids)

    def close(self):
        assert not self.active
        for h in self.handles:h.remove()
        self.handles=[];self.rows={};self.head=None
