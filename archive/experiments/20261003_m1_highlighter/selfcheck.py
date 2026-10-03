#!/usr/bin/env python3
"""Deterministic salience/guidance and small multimodal model checks; no GT."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from highlighter import AttentionHighlighter,salience,local_guidance,guide_output
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


def main():
    torch.set_num_threads(1);torch.manual_seed(0)
    hidden=torch.randn(19,64);weight=torch.randn(256,64)
    got=salience(hidden,weight,batch=7)
    probs=(hidden@weight.T).softmax(-1).topk(10,dim=-1).values
    oracle=-torch.special.xlogy(probs,probs).sum(-1)/torch.log(torch.tensor(10.))
    assert torch.allclose(got,oracle,atol=1e-6,rtol=1e-5)
    assert torch.isfinite(salience(hidden*10000,weight)).all()
    ids=torch.tensor([10,127,127,126,11,127,126,12,127,126])
    frames=[(0.,''),(8.,''),(24.,'')];wins=[(0.,8.),(8.,16.),(16.,24.)]
    guide,pos,fid=local_guidance(ids,127,[2,1,1],frames,wins,torch.tensor([1.,3.,2.,0.]))
    assert guide[0][1].tolist()==[.25,.75] and guide[1][0].tolist()==[5]
    assert guide[2][0].tolist()==[8] and guide[2][1].tolist()==[1.]
    empty,_,_=local_guidance(ids,127,[2,1,1],frames,[(16.,20.)],torch.ones(4))
    assert len(empty[0][0])==0
    o=torch.randn(1,4,1,16);d=torch.randn_like(o);g=torch.tensor([.2,.8])
    got,diag,gamma=guide_output(o,d,g)
    sim=(1+torch.nn.functional.cosine_similarity(o,d,dim=-1).clamp(-1,1))/2
    expected_gamma=(2-4*sim/sim.sum(1,keepdim=True)).relu()
    assert torch.allclose(gamma,expected_gamma,atol=1e-6)
    assert torch.allclose(got,o+.2*expected_gamma[...,None]*d,atol=1e-6)
    assert torch.equal(guide_output(o,d,g,0.)[0],o)
    assert torch.equal(guide_output(o,torch.zeros_like(d),g)[0],o)
    assert torch.isfinite(guide_output(o,-o,g)[0]).all()
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
        model=Qwen3VLModel(cfg).eval().to(dtype);head=torch.nn.Linear(64,256,bias=False).to(dtype)
        for p in [*model.parameters(),*head.parameters()]:p.requires_grad_(False)
        j=SimpleNamespace(family='qwen3_vl',model=SimpleNamespace(model=model,get_output_embeddings=lambda:head),device=torch.device('cpu'),yes_ids=list(range(6)),no_ids=list(range(6,12)))
        j.image_token_id=127
        j.prefix_cache=lambda enc:model(**enc,use_cache=True).past_key_values
        j._step=lambda c,ids:model(input_ids=torch.tensor([ids]),past_key_values=c,use_cache=True).last_hidden_state[0,-1]
        j._logits_fp32=lambda rows,ids:rows.float()@head.weight[ids].float().T
        eng=AttentionHighlighter(j)
        enc={'input_ids':torch.tensor([[10,125,127,126,11,125,127,126,12]]),
            'pixel_values':torch.randn(8,24).to(dtype),'image_grid_thw':torch.tensor([[1,2,2],[1,2,2]])}
        with torch.no_grad():
            model.rope_deltas=None;cache,hidden=eng.prefix_cache(enc);scores=salience(hidden,eng.W32);snapshot=copy.deepcopy(cache);rope=model.rope_deltas.clone();n=cache.get_seq_length()
            h=j._step(cache,[30,31,32]);cache.crop(n);native=j._logits_fp32(h[None],torch.arange(12))[0]
            positions=torch.where(enc['input_ids'][0]==127)[0];g=scores/scores.sum()
            identity,ed=eng.logits(cache,[30,31,32],positions,g,0.)
            changed,cd=eng.logits(cache,[30,31,32],positions,g)
            absent,ad=eng.logits(cache,[30,31,32],positions[:0],g[:0])
            assert torch.equal(identity,native) and torch.equal(absent,native)
            assert ed['layer_indices']==cd['layer_indices']==list(range(4,18)) and ad['layer_indices']==[]
            assert not torch.equal(changed,native)
            restored=j._step(cache,[30,31,32]);cache.crop(n);assert torch.equal(h,restored)
            assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,snapshot.layers))
            assert torch.equal(model.rope_deltas,rope)
            checks.append({'dtype':str(dtype),'native_restore_exact':True,'zero_and_empty_native_exact':True,
                'layers4to17':True,'guided_native_delta':float((changed-native).abs().max()),
                'salience_finite':True,'local_mapping_exact':True,'all_prefix_KV_exact':True})
        eng.close()
    out=ROOT/'runs/20261003_m1_highlighter/selfcheck';out.mkdir(parents=True,exist_ok=True)
    result={'no_GT':True,'algebra_exact':True,'checks':checks};(out/'numerics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
