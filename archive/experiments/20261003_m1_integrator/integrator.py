"""Future input visibility for visual prefix rows; suffix attention unchanged."""
from contextlib import contextmanager
import torch


def allowed_edges(visual, device='cpu', native=False, future_keys='all'):
    visual=torch.as_tensor(visual, dtype=torch.bool, device=device)
    assert visual.ndim==1 and visual.any()
    p=len(visual);positions=torch.arange(p,device=device)
    causal=positions[None,:]<=positions[:,None]
    assert future_keys in ('all','text')
    future=visual[:,None] if future_keys=='all' else visual[:,None] & ~visual[None,:]
    return causal if native else causal | future


class PrefixIntegrator:
    def __init__(self, judge, future_keys='all'):
        assert judge.family=='qwen3_vl'
        assert future_keys in ('all','text')
        self.future_keys=future_keys
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
    def encoding(self,visual,native=False):
        allowed=allowed_edges(visual,self.judge.device,native,self.future_keys)
        with self.encoding_edges(allowed):yield

    @contextmanager
    def encoding_edges(self,allowed):
        assert self.active is None
        allowed=torch.as_tensor(allowed,dtype=torch.bool,device=self.judge.device)
        assert allowed.ndim==2 and allowed.shape[0]==allowed.shape[1]
        mask=torch.zeros(allowed.shape,dtype=self.judge.dtype,device=self.judge.device)
        mask.masked_fill_(~allowed,torch.finfo(mask.dtype).min)
        self.active=mask[None,None];self.visited=[]
        try:yield
        finally:self.active=None
        assert self.visited==list(range(len(self.layers))),self.visited

    def close(self):
        assert self.active is None
        for hook in self.hooks:hook.remove()
