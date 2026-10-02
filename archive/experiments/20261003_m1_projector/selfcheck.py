#!/usr/bin/env python3
"""Deterministic geometric and small multimodal model checks; no GT."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from projector import AttentionProjector,orthogonal_update,project_row
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


def main():
    torch.set_num_threads(1);torch.manual_seed(0)
    o=torch.randn(1,4,1,16);u=torch.randn_like(o)
    guided,diag=orthogonal_update(o,u,1.4);unit=u/(u.norm(dim=-1,keepdim=True)+1e-8)
    expected=o+1.4*((o-u)-((o-u)*unit).sum(-1,keepdim=True)*unit)
    assert torch.equal(guided,expected) and diag[-1]<2e-6
    assert torch.equal(orthogonal_update(o,u,0.)[0],o)
    assert torch.equal(orthogonal_update(o,o,1.4)[0],o)
    assert torch.allclose(orthogonal_update(o,torch.zeros_like(o),1.4)[0],2.4*o)
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=256,hidden_size=64,intermediate_size=128,
        num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
        max_position_embeddings=512,rope_scaling={'rope_type':'default','mrope_section':[2,3,3]}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,
            spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,
            deepstack_visual_indexes=[0,1]),image_token_id=127,video_token_id=124,
        vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa';cfg.vision_config._attn_implementation='sdpa'
    checks=[]
    for dtype in (torch.float32,torch.bfloat16):
        q=torch.randn(1,4,3,16).to(dtype);k=torch.randn(1,2,7,16).to(dtype);v=torch.randn_like(k)
        mask=torch.ones(1,1,3,7,dtype=torch.bool);mask[...,0]=False
        visual=torch.tensor([0,1,0,1,0,0,0],dtype=torch.bool)
        row,_=project_row(q,k,v,mask,.25,visual,1.4,2)
        scores=(q[:,:,-1:]@k.repeat_interleave(2,1).transpose(2,3))*.25;scores=scores.masked_fill(~mask[:,:,-1:],float('-inf'))
        oc=scores.softmax(-1,dtype=torch.float32).to(dtype)@v.repeat_interleave(2,1)
        uc=scores.masked_fill(visual[None,None,None],float('-inf')).softmax(-1,dtype=torch.float32).to(dtype)@v.repeat_interleave(2,1)
        assert torch.equal(row,orthogonal_update(oc,uc,1.4)[0].transpose(1,2))
        model=Qwen3VLModel(cfg).eval().to(dtype);head=torch.nn.Linear(64,12,bias=False).to(dtype)
        for p in [*model.parameters(),*head.parameters()]:p.requires_grad_(False)
        j=SimpleNamespace(family='qwen3_vl',model=SimpleNamespace(model=model),device=torch.device('cpu'),yes_ids=list(range(6)),no_ids=list(range(6,12)))
        j._step=lambda c,ids:model(input_ids=torch.tensor([ids]),past_key_values=c,use_cache=True).last_hidden_state[0,-1]
        j._logits_fp32=lambda rows,ids:rows.float()@head.weight[ids].float().T
        eng=AttentionProjector(j)
        enc={'input_ids':torch.tensor([[10,125,127,126,11,125,127,126,12]]),
            'pixel_values':torch.randn(8,24).to(dtype),'image_grid_thw':torch.tensor([[1,2,2],[1,2,2]])}
        with torch.no_grad():
            model.rope_deltas=None;cache=model(**enc,use_cache=True).past_key_values;snapshot=copy.deepcopy(cache);rope=model.rope_deltas.clone();n=cache.get_seq_length()
            h=j._step(cache,[30,31,32]);cache.crop(n);native=j._logits_fp32(h[None],torch.arange(12))[0]
            eager,ed=eng.logits(cache,[30,31,32],enc['input_ids'][0]==127,0.)
            changed,cd=eng.logits(cache,[30,31,32],enc['input_ids'][0]==127,1.4)
            assert ed['layer_indices']==cd['layer_indices']==list(range(36)) and not torch.equal(changed,eager)
            restored=j._step(cache,[30,31,32]);cache.crop(n);assert torch.equal(h,restored)
            assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,snapshot.layers))
            assert torch.equal(model.rope_deltas,rope)
            checks.append({'dtype':str(dtype),'last_row_GQA_oracle_exact':True,'native_restore_exact':True,
                'all36layers':True,'eager_native_delta':float((eager-native).abs().max()),'project_eager_delta':float((changed-eager).abs().max()),
                'max_absolute_projection':max(d[-1] for d in cd['geometry'])})
        eng.close()
    out=ROOT/'runs/20261003_m1_projector/selfcheck';out.mkdir(parents=True,exist_ok=True)
    result={'no_GT':True,'algebra_exact':True,'checks':checks};(out/'numerics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
