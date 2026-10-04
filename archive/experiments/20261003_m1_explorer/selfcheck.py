#!/usr/bin/env python3
"""Real 36-layer Qwen, multimodal cache extension and deterministic acquisition checks."""
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from explorer import FrameAttention,binary_entropy,choose_frames
from measure import rope_positions
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


def main():
    torch.set_num_threads(1);torch.manual_seed(0)
    assert binary_entropy(0)==np.log(2) and binary_entropy(1000)==0 and binary_entropy(-3)==binary_entropy(3)
    c=[{'time':float(x)} for x in (1,2,3,4)]
    assert choose_frames(c,[0,5],[1,1],2)==[1,0]
    assert choose_frames(c,[0,0,5],[0,2,1],2)==choose_frames(c,[0,5],[1,1],2)
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=256,hidden_size=64,intermediate_size=128,
        num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,head_dim=128,
        max_position_embeddings=512,rope_scaling={'rope_type':'default','mrope_section':[24,20,20]}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,
            spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,
            deepstack_visual_indexes=[0,1]),image_token_id=127,video_token_id=124,
        vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa';cfg.vision_config._attn_implementation='sdpa'
    results=[]
    for dtype in (torch.float32,torch.bfloat16):
        for N in (18,20):
            model=Qwen3VLModel(cfg).eval().to(dtype)
            for p in model.parameters():p.requires_grad_(False)
            j=SimpleNamespace(model=SimpleNamespace(model=model),device='cpu',image_token_id=127)
            ids=torch.tensor([[10]+[v for i in range(N) for v in (11+i,125,127,126)]+[60,61,62]])
            pixels=torch.randn(4*N,24).to(dtype);grids=torch.tensor([[1,2,2]]*N)
            enc={'input_ids':ids,'pixel_values':pixels,'image_grid_thw':grids,'attention_mask':torch.ones_like(ids)}
            modern='mm_token_type_ids' in inspect.signature(model.forward).parameters
            if modern:enc['mm_token_type_ids']=(ids==127).long()
            engine=FrameAttention(j)
            with torch.no_grad():
                model.rope_deltas=None;engine.start('prefix',ids[0]==127)
                cache=model(**enc,use_cache=True).past_key_values;engine.stop()
                native=copy.deepcopy(cache);native_rope=model.rope_deltas.clone();n=cache.get_seq_length()
                prefix_pos,_=rope_positions(j,ids,grids)
                for new_count in (2,4):
                    suffix=torch.tensor([[80]+[v for i in range(new_count) for v in (81+i,125,127,126)]+[91,92,93]])
                    newpixels=torch.randn(4*new_count,24).to(dtype);newgrid=torch.tensor([[1,2,2]]*new_count)
                    fullids=torch.cat((ids,suffix),1);fullgrid=torch.cat((grids,newgrid),0)
                    pos,delta=rope_positions(j,fullids,fullgrid)
                    assert torch.equal(pos[:,:,:n],prefix_pos)
                    kw={'input_ids':suffix,'pixel_values':newpixels,'image_grid_thw':newgrid,
                        'position_ids':pos[:,:,n:],'attention_mask':torch.ones_like(fullids),'past_key_values':cache,'use_cache':True}
                    if modern:kw['mm_token_type_ids']=(suffix==127).long()
                    if 'cache_position' in inspect.signature(model.forward).parameters:kw['cache_position']=torch.arange(n,len(fullids[0]))
                    model.rope_deltas=delta;engine.start('query',suffix[0]==127)
                    h=model(**kw).last_hidden_state[0,-1];engine.stop();cache.crop(n)
                    prior=engine.prior([1]*(N+new_count))
                    assert len(prior)==N+new_count and np.isfinite(prior).all() and abs(prior.sum()-1)<1e-6
                    assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,native.layers))
                    full={'input_ids':fullids,'pixel_values':torch.cat((pixels,newpixels)),
                        'image_grid_thw':fullgrid,'attention_mask':torch.ones_like(fullids),'position_ids':pos,'use_cache':True}
                    if modern:full['mm_token_type_ids']=(fullids==127).long()
                    fresh=model(**full).last_hidden_state[0,-1]
                    tol=2e-5 if dtype==torch.float32 else .06
                    assert torch.allclose(h.float(),fresh.float(),atol=tol,rtol=tol),(dtype,(h-fresh).abs().max())
                    model.rope_deltas=native_rope.clone()
                    results.append({'dtype':str(dtype),'original_frames':N,'new_frames':new_count,
                        'prefix_positions_exact':True,'all36_pre_rope_capture':True,'cache_restore_exact':True,
                        'cached_vs_fresh_max_abs':float((h-fresh).abs().max()),'tolerance':tol})
            engine.close()
    out=ROOT/'runs/20261003_m1_explorer/selfcheck';out.mkdir(parents=True,exist_ok=True)
    (out/'numerics.json').write_text(json.dumps({'no_GT':True,'checks':results},indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
