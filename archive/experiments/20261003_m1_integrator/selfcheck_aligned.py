#!/usr/bin/env python3
"""Direct ASR matching and actual cached visual-state witness; synthetic/no GT."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from aligned import aligned_edges
from integrator import PrefixIntegrator
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


def main():
    torch.set_num_threads(1);torch.manual_seed(0)
    visual=np.array([0,1,1,0,0,1,1,0,0,0,0,0],bool)
    speech=np.array([0,0,0,0,0,0,0,0,0,1,1,0],bool)
    local=np.zeros((2,12),bool);local[0,[1,2,9]]=True;local[1,[5,6,10]]=True
    edges=aligned_edges(visual,speech,local)
    expected=np.zeros((12,12),bool)
    for i in range(12):
        for j in range(12):
            window=np.flatnonzero(local[:,i])[0] if visual[i] else None
            expected[i,j]=(j<=i) or (visual[i] and not visual[j] and (not speech[j] or local[window,j]))
    assert np.array_equal(edges,expected)
    assert edges[1,9] and not edges[1,10] and edges[5,10] and not edges[5,9]
    assert edges[1,11] and not edges[1,5] # policy is global; later image is unavailable
    for invalid in (np.zeros_like(local),np.ones_like(local)):
        try:aligned_edges(visual,speech,invalid)
        except AssertionError:pass
        else:raise AssertionError('unassigned/duplicate image mapping accepted')
    cfg=Qwen3VLTextConfig(vocab_size=128,hidden_size=64,intermediate_size=128,
        num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
        max_position_embeddings=128,rope_scaling={'rope_type':'default','mrope_section':[2,3,3]})
    cfg._attn_implementation='sdpa';checks=[]
    for dtype in (torch.float32,torch.bfloat16):
        lm=Qwen3VLTextModel(cfg).eval().to(dtype)
        for p in lm.parameters():p.requires_grad_(False)
        judge=SimpleNamespace(family='qwen3_vl',device=torch.device('cpu'),dtype=dtype,
            model=SimpleNamespace(model=SimpleNamespace(language_model=lm)))
        engine=PrefixIntegrator(judge,'text');ids=torch.arange(12)[None]
        own=ids.clone();own[0,9]+=50;other=ids.clone();other[0,10]+=50
        with torch.no_grad():
            native=lm(input_ids=ids,use_cache=True)
            with engine.encoding_edges(edges):original=lm(input_ids=ids,use_cache=True)
            with engine.encoding_edges(edges):changed_own=lm(input_ids=own,use_cache=True)
            with engine.encoding_edges(edges):changed_other=lm(input_ids=other,use_cache=True)
            val=original.past_key_values.layers[1].values[:,:,1:3]
            own_delta=float((val-changed_own.past_key_values.layers[1].values[:,:,1:3]).abs().max())
            other_delta=float((val-changed_other.past_key_values.layers[1].values[:,:,1:3]).abs().max())
            assert own_delta>0 and other_delta==0
            # At deeper layers indirect paths may carry unmatched ASR; no isolation claim.
            suffix=lm(input_ids=torch.tensor([[40,41]]),past_key_values=original.past_key_values,use_cache=True)
            suffix_own=lm(input_ids=torch.tensor([[40,41]]),past_key_values=changed_own.past_key_values,use_cache=True)
            assert not torch.equal(suffix.last_hidden_state,suffix_own.last_hidden_state)
            assert torch.equal(lm(input_ids=ids,use_cache=True).last_hidden_state,native.last_hidden_state)
        engine.close();checks.append({'dtype':str(dtype),'layer1_own_ASR_visual_KV_delta':own_delta,
            'layer1_unmatched_ASR_visual_KV_delta':other_delta,'suffix_sensitive':True,'native_restored':True})
    out=ROOT/'runs/20261003_m1_integrator/selfcheck';out.mkdir(parents=True,exist_ok=True)
    result={'no_GT':True,'exact_mask_oracle':True,'reject_bad_image_mapping':True,'checks':checks}
    (out/'aligned_visibility.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
