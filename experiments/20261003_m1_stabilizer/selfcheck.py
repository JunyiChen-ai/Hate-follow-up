#!/usr/bin/env python3
"""Native rotary and actual multimodal prefix checks without labels."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from stabilizer import PrefixStabilizer,phase_query
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


def main():
    torch.set_num_threads(1);torch.manual_seed(0)
    pairs=torch.tensor(list(range(0,60,3))+[60,61,62,63]);freq=1e6**(-torch.arange(64)/64)
    rows=torch.tensor([3,7,19]);deltas=(torch.arange(32)%2).float()*.5
    algebra=[]
    for dtype in (torch.float32,torch.bfloat16):
        q=torch.randn(1,32,41,128).to(dtype);original=q.clone()
        got,diag=phase_query(q,rows,pairs,freq,deltas)
        oracle=q.clone();allowed=torch.zeros_like(q,dtype=torch.bool)
        for h in range(1,32,2):
            for r in rows.tolist():
                for d in pairs.tolist():
                    x,y=q[0,h,r,d].float(),q[0,h,r,d+64].float();a=.5*freq[d]
                    oracle[0,h,r,d]=x*a.cos()-y*a.sin();oracle[0,h,r,d+64]=x*a.sin()+y*a.cos()
                    allowed[0,h,r,d]=True;allowed[0,h,r,d+64]=True
        assert torch.equal(got,oracle) and torch.equal(got[~allowed],q[~allowed]) and torch.equal(q,original)
        assert float(diag[0])<1e-4 and torch.equal(phase_query(q,rows,pairs,freq,deltas*0)[0],q)
        algebra.append({'dtype':str(dtype),'scalar_rotation_exact':True,'untouched_coordinates_exact':True,'max_FP32_norm_error':float(diag[0])})
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=256,hidden_size=64,intermediate_size=128,
        num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
        max_position_embeddings=512,rope_scaling={'rope_type':'default','mrope_section':[4,2,2]}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,
            spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,
            deepstack_visual_indexes=[0,1]),image_token_id=127,video_token_id=124,
        vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa';cfg.vision_config._attn_implementation='sdpa'
    checks=[]
    for dtype in (torch.float32,torch.bfloat16):
        model=Qwen3VLModel(cfg).eval().to(dtype)
        for p in model.parameters():p.requires_grad_(False)
        j=SimpleNamespace(family='qwen3_vl',model=SimpleNamespace(model=model),device=torch.device('cpu'),image_token_id=127)
        j.prefix_cache=lambda enc:model(**enc,use_cache=True).past_key_values
        eng=PrefixStabilizer(j)
        ids=torch.tensor([[10]+[v for i in range(20) for v in [11+i,125,127,126]]+[60]])
        enc={'input_ids':ids,'attention_mask':torch.ones_like(ids),'pixel_values':torch.randn(80,24).to(dtype),
            'image_grid_thw':torch.tensor([[1,2,2]]*20)}
        with torch.no_grad():
            model.rope_deltas=None;base=j.prefix_cache(enc);rope=model.rope_deltas.clone()
            model.rope_deltas=None;stable,diag=eng.prefix_cache(enc)
            assert diag['layers']==list(range(36)) and diag['deltas']==[0.,.5,0.,.5]
            assert all(r[1]>0 for r in diag['geometry']) and torch.equal(model.rope_deltas,rope)
            assert torch.equal(base.layers[0].keys,stable.layers[0].keys) and torch.equal(base.layers[0].values,stable.layers[0].values)
            assert not torch.equal(base.layers[-1].keys,stable.layers[-1].keys)
            snapshot=copy.deepcopy(stable);n=stable.get_seq_length()
            h=model(input_ids=torch.tensor([[70,71,72]]),past_key_values=stable,use_cache=True).last_hidden_state
            stable.crop(n);assert eng.active is None and eng.visited==list(range(36))
            assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(stable.layers,snapshot.layers))
            model.rope_deltas=None;zero,zd=eng.prefix_cache(enc,'zero')
            assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(zero.layers,base.layers))
            model.rope_deltas=None;restored=j.prefix_cache(enc)
            assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(restored.layers,base.layers))
            assert torch.isfinite(h).all()
            checks.append({'dtype':str(dtype),'all36layers':True,'20_images':True,'zero_and_restore_native_KV_exact':True,
                'layer0_KV_exact':True,'later_KV_changed':True,'suffix_disabled_and_crop_exact':True})
        eng.close()
    out=ROOT/'runs/20261003_m1_stabilizer/selfcheck';out.mkdir(parents=True,exist_ok=True)
    result={'no_GT':True,'algebra':algebra,'checks':checks};(out/'numerics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
