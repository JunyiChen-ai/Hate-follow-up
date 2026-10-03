"""PAS-derived visual-query temporal phase intervention in the shared prefix."""
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def temporal_coordinates(rotary):
    half=len(rotary.inv_freq);sections=rotary.mrope_section
    assert sum(sections)==half and rotary.attention_scaling==1
    mask=torch.ones(half,dtype=torch.bool,device=rotary.inv_freq.device)
    mask[1:3*sections[1]:3]=False;mask[2:3*sections[2]:3]=False
    assert int(mask.sum())==sections[0]
    # Native rotary outputs for independently changed T positions must leave
    # exactly the spatial dimensions untouched. Nonzero phases avoid aliasing.
    x=torch.zeros((1,3,2*half),device=mask.device,dtype=torch.float32)
    p=torch.tensor([[[0,1,5]],[[2,7,9]],[[3,4,8]]],device=mask.device)
    c,s=rotary(x,p);p2=p.clone();p2[0]+=1
    c2,s2=rotary(x,p2)
    dc=c[...,:half]*c2[...,:half]+s[...,:half]*s2[...,:half]
    ds=c[...,:half]*s2[...,:half]-s[...,:half]*c2[...,:half]
    a=rotary.inv_freq.float()*mask
    assert torch.allclose(dc,a.cos()[None,None,:].expand_as(dc),atol=1e-6,rtol=1e-5)
    assert torch.allclose(ds,a.sin()[None,None,:].expand_as(ds),atol=1e-6,rtol=1e-5)
    assert torch.equal(c[...,~torch.cat((mask,mask))],c2[...,~torch.cat((mask,mask))])
    assert torch.equal(s[...,~torch.cat((mask,mask))],s2[...,~torch.cat((mask,mask))])
    return mask.nonzero().flatten()


def phase_query(query,rows,pairs,inv_freq,deltas):
    """Only nonzero-phase heads, image rows and native split-half pairs change."""
    half=query.shape[-1]//2;heads=torch.where(deltas!=0)[0]
    if not len(heads) or not len(rows):return query,torch.zeros(4,device=query.device)
    ix=(slice(None),heads[:,None,None],rows[None,:,None],pairs[None,None,:])
    iy=(slice(None),heads[:,None,None],rows[None,:,None],pairs[None,None,:]+half)
    x=query[ix].float();y=query[iy].float()
    a=deltas[heads,None]*inv_freq[pairs][None,:]
    c=a.cos()[None,:,None,:];s=a.sin()[None,:,None,:]
    u=x*c-y*s;v=x*s+y*c
    new=query.clone();new[ix]=u.to(query.dtype);new[iy]=v.to(query.dtype)
    diag=torch.stack(((u.square()+v.square()-x.square()-y.square()).abs().max(),
        torch.maximum((new[ix].float()-x).abs().max(),(new[iy].float()-y).abs().max()),
        (x.square()+y.square()).mean(),(new[ix].float().square()+new[iy].float().square()).mean()))
    return new,diag


class PrefixStabilizer:
    def __init__(self,judge):
        assert judge.family=='qwen3_vl'
        self.judge=judge;self.lm=judge.model.model.language_model
        assert len(self.lm.layers)==36
        assert all(l.self_attn.config._attn_implementation=='sdpa' for l in self.lm.layers)
        self.selected={id(l.self_attn):i for i,l in enumerate(self.lm.layers)}
        self.pairs=temporal_coordinates(self.lm.rotary_emb)
        self.inv_freq=self.lm.rotary_emb.inv_freq.detach().float().clone()
        self.original=ALL_ATTENTION_FUNCTIONS['sdpa'];self.wrapper=self.dispatch
        ALL_ATTENTION_FUNCTIONS['sdpa']=self.wrapper
        self.active=None;self.visited=[];self.geometry=[]

    def dispatch(self,module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
        if self.active is not None and id(module) in self.selected:
            rows,n,deltas=self.active
            assert query.shape[0]==1 and query.shape[2]==key.shape[2]==value.shape[2]==n
            assert query.shape[-1]==2*len(self.inv_freq) and query.shape[1]==len(deltas)
            assert not module.training and dropout==0
            query,diag=phase_query(query,rows,self.pairs,self.inv_freq,deltas)
            self.visited.append(self.selected[id(module)]);self.geometry.append(diag)
        return self.original(module,query,key,value,attention_mask,dropout=dropout,scaling=scaling,**kwargs)

    @torch.no_grad()
    def prefix_cache(self,enc,phase='stable'):
        assert self.active is None and phase in ('stable','zero')
        ids=enc['input_ids'][0];rows=torch.where(ids==self.judge.image_token_id)[0].to(self.judge.device)
        assert len(rows)>0 and torch.all(enc['attention_mask']==1)
        H=self.lm.layers[0].self_attn.config.num_attention_heads
        deltas=(torch.arange(H,device=self.judge.device)%2).float()*(.5 if phase=='stable' else 0.)
        self.visited=[];self.geometry=[];self.active=(rows,len(ids),deltas)
        try:cache=self.judge.prefix_cache(enc)
        finally:self.active=None
        assert self.visited==list(range(36))
        diag={'layers':self.visited[:],'geometry':torch.stack(self.geometry).cpu().tolist(),
            'deltas':deltas.cpu().tolist(),'temporal_pairs':self.pairs.cpu().tolist(),
            'inv_freq':self.inv_freq.cpu().tolist(),'image_positions':rows.cpu().tolist()}
        self.geometry=[]
        return cache,diag

    def close(self):
        assert self.active is None and ALL_ATTENTION_FUNCTIONS['sdpa']==self.wrapper
        ALL_ATTENTION_FUNCTIONS['sdpa']=self.original
