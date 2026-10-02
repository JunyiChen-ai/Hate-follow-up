#!/usr/bin/env python3
"""CPU algebra and actual small Qwen suffix/cache checks, no dataset inputs."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from amplifier import ImageAmplifier,amplify_row,contrast_margin
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


class Judge:
    family='qwen3_vl';device=torch.device('cpu');yes_ids=[0,1];no_ids=[2,3]
    def __init__(self,lm,head,dtype):
        self.model=SimpleNamespace(model=SimpleNamespace(language_model=lm));self.lm=lm;self.head=head;self.dtype=dtype
    def _step(self,cache,ids):
        self.hidden=self.lm(input_ids=torch.tensor([ids]),past_key_values=cache,use_cache=True).last_hidden_state
        return self.hidden[0,-1]
    def _logits_fp32(self,h,ids):return h.float()@self.head[ids].float().T


def main():
    torch.set_num_threads(1);torch.manual_seed(0);checks=[]
    for dtype in (torch.float32,torch.bfloat16):
        q=torch.randn(1,4,3,8,dtype=dtype);k=torch.randn(1,2,9,8,dtype=dtype);v=torch.randn_like(k)
        visual=torch.tensor([0,1,1,0,1,0,0,0,0],dtype=torch.bool);scale=8**-.5
        mask=torch.zeros(1,1,3,9,dtype=dtype)
        for i in range(3):mask[:,:,i,7+i:]=torch.finfo(dtype).min
        row,mass=amplify_row(q,k,v,mask,scale,visual,.5,2)
        logits=q[:,:,-1:]@k.repeat_interleave(2,1).transpose(2,3)*scale+mask[:,:,-1:]
        for position in torch.where(visual)[0]:logits[...,position]+=.5*logits[...,position].abs()
        expected=(logits.softmax(-1,dtype=torch.float32).to(dtype)@v.repeat_interleave(2,1)).transpose(1,2)
        assert torch.equal(row,expected) and mass[1]>=mass[0]
        cfg=Qwen3VLTextConfig(vocab_size=128,hidden_size=64,intermediate_size=128,
            num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
            max_position_embeddings=128,rope_scaling={'rope_type':'default','mrope_section':[2,3,3]})
        cfg._attn_implementation='sdpa';lm=Qwen3VLTextModel(cfg).eval().to(dtype)
        for p in lm.parameters():p.requires_grad_(False)
        j=Judge(lm,torch.randn(4,64,dtype=dtype),dtype);engine=ImageAmplifier(j)
        ids=torch.arange(9)[None];suffix=[20,21,22];original=copy.deepcopy(lm.state_dict())
        with torch.no_grad():
            prefix=lm(input_ids=ids,use_cache=True).past_key_values
            native=j._step(copy.deepcopy(prefix),suffix).clone()
            cache=copy.deepcopy(prefix)
            zero,d0=engine.logits(cache,suffix,visual,0.);hidden0=j.hidden.clone()
            amplified,da=engine.logits(cache,suffix,visual,.5);hidden1=j.hidden.clone()
            assert torch.equal(hidden0[:,:-1],hidden1[:,:-1]) and not torch.equal(hidden0[:,-1],hidden1[:,-1])
            for a,b in zip(prefix.layers,cache.layers):
                assert torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values)
            assert torch.equal(j._step(copy.deepcopy(prefix),suffix),native)
            assert d0['layer_indices']==da['layer_indices']==[2,3]
            assert da['image_attention_mass_before_after'][1]>da['image_attention_mass_before_after'][0]
            assert all(torch.equal(p,original[n]) for n,p in lm.state_dict().items())
        engine.close()
        # Full-vocabulary normalizers cancel only after contrast, before class aggregation.
        full=torch.randn(2,128);a,b=full;lp=full.log_softmax(-1)
        margin=contrast_margin(a[:4],b[:4],2)
        equivalent=contrast_margin(lp[0,:4],lp[1,:4],2)
        assert abs(margin-equivalent)<2e-6
        assert abs(contrast_margin(a[:4],b[:4],2,1.)-float(a[:2].logsumexp(0)-a[2:4].logsumexp(0)))<1e-7
        checks.append({'dtype':str(dtype),'GQA_row_oracle':True,'only_last_query_row_changed':True,
            'cached_prefix_exact':True,'native_restored':True,'weights_exact':True,'mass':da,
            'answer_token_max_delta':float((amplified-zero).abs().max()),'normalizer_equivalence':abs(margin-equivalent)})
    out=ROOT/'runs/20261003_m1_amplifier/selfcheck';out.mkdir(parents=True,exist_ok=True)
    (out/'checks.json').write_text(json.dumps({'no_GT':True,'checks':checks},indent=2)+'\n');print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
