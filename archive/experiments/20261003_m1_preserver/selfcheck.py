#!/usr/bin/env python3
"""Actual Qwen prefix/suffix witness, without GPU, dataset labels or trained weights."""
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from preserver import AttentionPreserver, blend
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


def main():
    torch.set_num_threads(1);torch.manual_seed(0)
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=256,hidden_size=64,intermediate_size=128,
        num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,head_dim=128,
        max_position_embeddings=512,rope_scaling={'rope_type':'default','mrope_section':[24,20,20]}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,
            spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,
            deepstack_visual_indexes=[0,1]),image_token_id=127,video_token_id=124,
        vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa';cfg.vision_config._attn_implementation='sdpa'
    result=[]
    for dtype in (torch.float32,torch.bfloat16):
        for N in (20,18):
            model=Qwen3VLModel(cfg).eval().to(dtype)
            for p in model.parameters():p.requires_grad_(False)
            W=torch.randn(12,64,dtype=dtype)
            j=SimpleNamespace(family='qwen3_vl',model=SimpleNamespace(model=model),yes_ids=list(range(6)),no_ids=list(range(6,12)))
            j._step=lambda c,ids:model(input_ids=torch.tensor([ids]),past_key_values=c,use_cache=True).last_hidden_state[0,-1]
            j._logits_fp32=lambda h,ids:h.float()@W[ids].float().T
            eng=AttentionPreserver(j)
            ids=torch.tensor([[10]+[v for i in range(N) for v in [11+i,125,127,126]]+[60,61]])
            ref={'input_ids':ids,'attention_mask':torch.ones_like(ids),'pixel_values':torch.randn(4*N,24).to(dtype),
                'image_grid_thw':torch.tensor([[1,2,2]]*N)}
            full={**ref,'input_ids':torch.cat((ids,torch.tensor([[62,63,64,65,66,67]])),1)}
            full['attention_mask']=torch.ones_like(full['input_ids'])
            if 'mm_token_type_ids' in inspect.signature(model.forward).parameters:
                for enc in (ref,full):enc['mm_token_type_ids']=(enc['input_ids']==127).long()
            with torch.no_grad():
                model.rope_deltas=None;rc=model(**ref,use_cache=True).past_key_values;rr=model.rope_deltas.clone()
                model.rope_deltas=None;fc=model(**full,use_cache=True).past_key_values;fr=model.rope_deltas.clone()
                rs=copy.deepcopy(rc);fs=copy.deepcopy(fc);query=[70,71,72,73,74]
                model.rope_deltas=fr.clone();native=j._step(fc,query).clone();fc.crop(fs.get_seq_length())
                rz,rl,_=eng.run(rc,query,rr,'capture')
                references=[x.clone() for x in eng.reference]
                pz,pl,g=eng.run(fc,query,fr,'mix')
                assert len(g)==36 and any(x[2]>0 for x in g)
                assert all(torch.equal(a,b) for a,b in zip(references,eng.reference))
                zz,zl,zg=eng.run(fc,query,fr,'mix',alpha=0)
                nl=j._logits_fp32(native[None],j.yes_ids+j.no_ids)[0].numpy()
                assert (zl==nl).all() and all(x[2]==0 for x in zg)
                assert not (pl==nl).all()
                for current,saved in ((fc,fs),(rc,rs)):
                    assert current.get_seq_length()==saved.get_seq_length()
                    assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(current.layers,saved.layers))
                x=references[0];assert torch.equal(blend(x,x,.5),x) and blend(x,x,0) is x
                eng.clear();model.rope_deltas=fr.clone()
                restored=j._step(fc,query);assert torch.equal(restored,native)
                result.append({'dtype':str(dtype),'frames':N,'zero_native_exact':True,'same_reference_identity':True,
                    'both_caches_restore_exact':True,'all36layers':True,'mixed_final_changed':True,'reference_readonly':True})
            eng.close()
    out=ROOT/'runs/20261003_m1_preserver/selfcheck';out.mkdir(parents=True,exist_ok=True)
    (out/'numerics.json').write_text(json.dumps({'no_GT':True,'checks':result},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
