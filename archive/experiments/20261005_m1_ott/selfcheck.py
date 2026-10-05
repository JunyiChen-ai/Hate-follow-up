#!/usr/bin/env python3
"""Real Qwen3 36-layer BF16/DeepStack equivalence and source-bound sparse layouts."""
import copy
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from ott import ROOT,select,capture,pack,aggregate,prefill,dense_plan
from src.mllm_judge import Judge
from src.stance_cache import positions


def fixture(model,F,dtype):
    head=torch.nn.Linear(model.config.text_config.hidden_size,256,bias=False).to(dtype)
    j=Judge.__new__(Judge);j.model=SimpleNamespace(model=model,config=model.config,get_output_embeddings=lambda:head);j.device=torch.device('cpu');j.dtype=dtype
    j.image_token_id=127;j.forward_params=set(inspect.signature(model.forward).parameters)
    frames=[(float(i),Path(f'{i}.jpg')) for i in range(F)]
    pixels=torch.randn(36,24).repeat(F,1).to(dtype)
    pixels[:36]=-2*pixels[:36]  # One changed frame, then actual repeated identical image inputs.
    grids=torch.tensor([[1,6,6]]*F)
    def messages(frames,segments):return frames,[p for _,p in frames]
    def encode(msgs,files):
        order=[int(t) for t,_ in msgs]
        ids=torch.tensor([[10]+[v for f in order for v in ([11+f,125]+[127]*9+[126])]+[50,51,52]])
        enc=dict(input_ids=ids,pixel_values=pixels.reshape(F,36,24)[order].reshape(-1,24),image_grid_thw=grids[order],attention_mask=torch.ones_like(ids))
        if 'mm_token_type_ids' in j.forward_params:enc['mm_token_type_ids']=(ids==127).long()
        return 'fixtureprefix',enc
    j.prefix_messages=messages;j.encode_prefix=encode
    j.branch_ids=lambda *a,**k:([60,61],'Q');j.answer_ids=lambda *a,**k:([62],'A');j.turn=lambda role,text:dict(role=role,content=text)
    j.yes_ids=[70];j.no_ids=[71];j.softcap=None
    return j,frames,pixels,grids


def main():
    torch.set_num_threads(1);torch.manual_seed(0);checks=[]
    from ott import budgets,matching,components,sinkhorn
    x=torch.eye(9).repeat(3,1);grids=torch.tensor([[1,6,6]]*3);weights=torch.ones(27)
    plan=select(x,weights,grids,2)
    assert plan['spatial_budget']==6 and plan['temporal_budget']==4 and sum(plan['budgets'])==4
    assert plan['retained_count']==14 and not plan['pruned_sources'] and any(len(m)>1 for m in plan['members'])
    # Strong-pruned root owns an entire low-cost descendant chain.
    selected=[[0,1]]*4;matches=[[],[[0,0]],[[0,0]]];costs=torch.zeros(3,2,2);costs[1,0,0]=.8
    roots,members,removed,edges=components(selected,matches,costs,2)
    assert 4 in removed and 6 in removed and 4 not in roots and 6 not in roots
    assert matching(torch.tensor([[.9,.8],[.2,.1]]),2)==[[0,0],[0,1]]
    bb,left=budgets(torch.tensor([0.,20.,20.]),5,2);assert bb==[2,2,1] and left==0
    bb,left=budgets(torch.zeros(2),5,2);assert bb==[2,2] and left==1
    changed=x.clone();changed[9:18]*=-1
    q=select(changed,weights,grids,2);assert q['pruned_sources']
    checks.append(dict(selection='coverage/LOO/budgets/many-to-one/fullchain merge and prune PASS'))
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=36,
        num_attention_heads=32,num_key_value_heads=8,head_dim=128,max_position_embeddings=512,
        rope_scaling={'rope_type':'default','mrope_section':[24,20,20]}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,
            temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1]),
        image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa'
    with torch.no_grad():
        for dtype in (torch.float32,torch.bfloat16):
            for F in (18,20):
                model=Qwen3VLModel(cfg).eval().to(dtype)
                j,frames,pixels,grids=fixture(model,F,dtype);_,enc=j.encode_prefix(frames,[])
                model.rope_deltas=None
                with capture(j) as c:cache=j.prefix_cache(enc)
                original_cache=copy.deepcopy(cache);native_delta=model.rope_deltas.clone()
                j.extend_cache(cache,[60,61]);j.extend_cache(cache,[62]);n=cache.get_seq_length()
                native=j.cached_margin(cache,[63,64],in_place=True);cache.crop(n)
                native_ctx=dict(global_margin=1.,stance='Yes')
                plan=select(c['features'].cpu(),c['saliency'].cpu(),grids,2)
                assert plan['retained_count']<plan['original_count'],'sparse production path not exercised'
                msgs,text=frames,'fixtureprefix';dense=dense_plan(plan);packed,evidence=pack(j,c,dense['roots'],c['features'],c['deepstack'])
                assert evidence['packed_ids']==enc['input_ids'][0].tolist()
                manual,ctx=prefill(j,msgs,text,packed,native_ctx)
                assert manual.get_seq_length()==n and torch.equal(ctx['rope'],native_delta)
                assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,manual.layers))
                z=j.cached_margin(manual,[63,64],in_place=True);manual.crop(n);assert z==native
                merged=aggregate(c['features'],plan['members']);deep=[aggregate(v,plan['members']) for v in c['deepstack']];packed,evidence=pack(j,c,plan['roots'],merged,deep)
                assert all(torch.equal(v,m.to(v.dtype)) for v,m in zip(packed['deepstack_visual_embeds'],deep))
                compressed,ctx=prefill(j,msgs,text,packed,native_ctx);length=compressed.get_seq_length()
                clone=copy.deepcopy(compressed);model.rope_deltas=ctx['rope'].clone()
                z1=j.cached_margin(compressed,[63,64],in_place=True);compressed.crop(length)
                model.rope_deltas=ctx['rope'].clone();z2=j.cached_margin(clone,[63,64],in_place=True);clone.crop(length)
                assert z1==z2 and all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(compressed.layers,clone.layers))
                assert evidence['visual_source_positions_exact']
                assert evidence['text_and_boundaries_preserved'] and compressed.get_seq_length()==ctx['stance_cache_tokens']
                checks.append(dict(dtype=str(dtype),frames=F,layers=36,deepstack_layers=2,native_manual_all_KV_exact=True,
                    native_manual_margin_exact=True,compressed_clone_all_KV_exact=True,source_positions_exact=True,
                    retained=plan['retained_count'],original=plan['original_count'],compressed_cache_tokens=length))
                del model,j,cache,original_cache,manual,compressed,clone,c,packed
    out=ROOT/'runs/20261005_m1_ott/cpu_checks';out.mkdir(parents=True,exist_ok=True)
    (out/'run.pid').write_text(str(os.getpid()))
    (out/'summary.json').write_text(json.dumps(dict(GT_read=False,pass_all=True,checks=checks),indent=2)+'\n')
    print(json.dumps(checks,indent=2));print('CPU_CHECKS_PASS',flush=True)


if __name__=='__main__':main()
