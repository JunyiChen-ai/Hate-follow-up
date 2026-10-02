#!/usr/bin/env python3
"""CPU probability and real suffix/cache tests, no media labels."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from recycler import AttentionRecycler,recycle_attention,sink_mask
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


class Judge:
    family='qwen3_vl';device=torch.device('cpu');yes_ids=[0,1];no_ids=[2,3]
    def __init__(self,lm,head):
        self.model=SimpleNamespace(model=SimpleNamespace(language_model=lm));self.lm=lm;self.head=head
    def _step(self,cache,ids):return self.lm(input_ids=torch.tensor([ids]),past_key_values=cache,use_cache=True).last_hidden_state[0,-1]
    def _logits_fp32(self,h,ids):return h.float()@self.head[ids].float().T


def main():
    torch.set_num_threads(1);torch.manual_seed(0);checks=[]
    for dtype in (torch.float32,torch.bfloat16):
        hidden=torch.zeros(1,3,4096,dtype=dtype);hidden[0,1,12]=100;hidden[0,2,28]=100
        channels=torch.tensor([12,3]);assert sink_mask(hidden,channels).tolist()==[[False,True,False]]
        q=torch.randn(1,4,3,8,dtype=dtype);k=torch.randn(1,2,9,8,dtype=dtype);v=torch.randn_like(k)
        visual=torch.tensor([0,1,1,0,1,0,0,0,0],dtype=torch.bool)
        sinks=torch.tensor([1,0,0,0,1,0,0,0,1],dtype=torch.bool);scale=8**-.5
        mask=torch.zeros(1,1,3,9,dtype=dtype)
        for i in range(3):mask[:,:,i,7+i:]=torch.finfo(dtype).min
        out,stats=recycle_attention(q,k,v,mask,scale,visual,sinks,2,.6)
        p=(q@k.repeat_interleave(2,1).transpose(2,3)*scale+mask).softmax(-1,dtype=torch.float32)
        adjusted=p.clone();selected=0
        for h in range(4):
            for i in range(3):
                a=p[0,h,i];vm=a[visual].sum();nv=visual&~sinks;sm=a[nv].sum();wm=a[sinks].sum()
                if vm>=.2 and sm>=.5*vm and sm>0 and wm>0:
                    selected+=1;adjusted[0,h,i,sinks]=.4*a[sinks]
                    adjusted[0,h,i,nv]+=a[nv]*(.6*wm/sm)
        oracle=(adjusted.to(dtype)@v.repeat_interleave(2,1)).transpose(1,2)
        assert torch.allclose(out,oracle,atol=.008 if dtype==torch.bfloat16 else 2e-6,rtol=0)
        assert selected>0 and stats[4]>0 and stats[5]<2e-6
        assert torch.allclose(adjusted.sum(-1),torch.ones(1,4,3),atol=2e-6)
        assert torch.equal(adjusted[mask.expand_as(p)<0],p[mask.expand_as(p)<0])
        # Small language model cannot have phi>=20 (sqrt64<20), so the real
        # cache test explicitly injects synthetic masks after testing capture.
        cfg=Qwen3VLTextConfig(vocab_size=128,hidden_size=64,intermediate_size=128,
            num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
            max_position_embeddings=128,rope_scaling={'rope_type':'default','mrope_section':[2,3,3]})
        cfg._attn_implementation='sdpa';lm=Qwen3VLTextModel(cfg).eval().to(dtype)
        for p in lm.parameters():p.requires_grad_(False)
        j=Judge(lm,torch.randn(4,64,dtype=dtype));engine=AttentionRecycler(j)
        with torch.no_grad():
            engine.begin_capture();cache=lm(input_ids=torch.arange(6)[None],use_cache=True).past_key_values
            j._step(cache,[6]);engine.end_capture(7)
            assert all(v.shape==(7,) and not v.any() for v in engine.cached.values())
            for layer in engine.layer_indices:engine.cached[layer][0]=True
            vis=torch.tensor([0,1,1,1,1,0,0],dtype=torch.bool);original=copy.deepcopy(cache)
            native=j._step(copy.deepcopy(cache),[20,21,22]).clone()
            eager,ed=engine.margin(cache,[20,21,22],vis,0.)
            z,d=engine.margin(cache,[20,21,22],vis,.6)
            again,ad=engine.margin(cache,[20,21,22],vis,.6)
            assert z==again and z!=eager and d==ad
            assert d['layer_indices']==[0,1,2]
            assert max(v[4] for v in d['statistics'])>0
            assert torch.equal(j._step(copy.deepcopy(cache),[20,21,22]),native)
            for a,b in zip(cache.layers,original.layers):assert torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values)
        engine.close()
        checks.append({'dtype':str(dtype),'sink_predicate4096':True,'GQA_oracle':True,'probability_conservation':True,
            'causal_entries_exact':True,'native_restoration':True,'cache_exact':True,'synthetic_sink_delta':z-eager})
    out=ROOT/'runs/20261003_m1_recycler/selfcheck';out.mkdir(parents=True,exist_ok=True)
    (out/'checks.json').write_text(json.dumps({'no_GT':True,'checks':checks},indent=2)+'\n');print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
