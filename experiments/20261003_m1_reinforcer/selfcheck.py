#!/usr/bin/env python3
"""Small actual multimodal Qwen numerical witnesses; no real data or GT."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from reinforcer import ResidualReinforcer,preserve_norm_update
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


def main():
    torch.set_num_threads(1);torch.manual_seed(0)
    h=torch.randn(2,4,64);d=torch.randn(64)
    x=preserve_norm_update(h,d);u=h+.17*d
    assert torch.allclose(x,u/u.norm(dim=-1,keepdim=True)*h.norm(dim=-1,keepdim=True),atol=2e-6)
    assert torch.allclose(x.norm(dim=-1),h.norm(dim=-1),atol=2e-6)
    assert torch.equal(preserve_norm_update(torch.zeros_like(h),d),torch.zeros_like(h))
    exact=torch.ones(1,1,64)*.17
    assert torch.equal(preserve_norm_update(exact,-torch.ones(64)),exact)
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=256,hidden_size=64,intermediate_size=128,
        num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
        max_position_embeddings=512,rope_scaling={'rope_type':'default','mrope_section':[2,3,3]}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,
            spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,
            deepstack_visual_indexes=[0,1]),image_token_id=127,video_token_id=124,
        vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa';cfg.vision_config._attn_implementation='sdpa'
    results=[]
    for dtype in (torch.float32,torch.bfloat16):
        model=Qwen3VLModel(cfg).eval().to(dtype);head=torch.nn.Linear(64,256,bias=False).to(dtype)
        for p in [*model.parameters(),*head.parameters()]:p.requires_grad_(False)
        judge=SimpleNamespace(family='qwen3_vl',model=SimpleNamespace(model=model),device=torch.device('cpu'),yes_ids=[1,2,3],no_ids=[4,5,6])
        judge._step=lambda c,ids:model(input_ids=torch.tensor([ids]),past_key_values=c,use_cache=True).last_hidden_state[0,-1]
        judge._logits_fp32=lambda rows,ids:rows.float()@head.weight[ids].float().T
        eng=ResidualReinforcer(judge);assert eng.sla_indices==[30,31,32,33,34]
        enc={'input_ids':torch.tensor([[10,125,127,126,11,12]]),
            'pixel_values':torch.randn(4,24).to(dtype),'image_grid_thw':torch.tensor([[1,2,2]])}
        query=[30,31,32];labels=torch.tensor(judge.yes_ids+judge.no_ids)
        with torch.no_grad():
            model.rope_deltas=None;out=model(**enc,use_cache=True);cache=out.past_key_values;rope=model.rope_deltas.clone();snapshot=copy.deepcopy(cache)
            n=cache.get_seq_length();native=judge._step(cache,query);cache.crop(n)
            ordinary=eng.read(cache,query);assert torch.equal(ordinary['final'],judge._logits_fp32(native[None],labels)[0])
            model.rope_deltas=None;reference=model(input_ids=torch.tensor([[10,11,12]]),use_cache=True).past_key_values
            ref=eng.read(reference,query);direction=ordinary['states']-ref['states']
            assert direction.shape==(36,64) and direction.abs().max()>0
            model.rope_deltas=rope.clone();full=eng.read(cache,query,direction)
            assert not torch.equal(full['final'],ordinary['final']) and full['layer_indices']==list(range(36))
            # Explicit same-model projection verifies answer-column reduction and layer averaging.
            all_previous=model.language_model.norm(full['states'][30:35].to(dtype)).float()@head.weight.float().T
            all_final=model.language_model.norm(full['states'][-1:].to(dtype)).float()@head.weight.float().T
            all_mixed=.7*all_final[0]+.3*all_previous.mean(0)
            expected=float(all_mixed[judge.yes_ids].logsumexp(0)-all_mixed[judge.no_ids].logsumexp(0))
            assert abs(full['full_margin']-expected)<2e-5
            restored=eng.read(cache,query);assert torch.equal(restored['final'],ordinary['final'])
            assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(snapshot.layers,cache.layers))
            assert cache.get_seq_length()==n and torch.equal(model.rope_deltas,rope)
            results.append({'dtype':str(dtype),'layers':full['layer_indices'],'sla_indices':eng.sla_indices,
                'native_exact':True,'cache_and_mRoPE_restored':True,'direction_nonzero':True,
                'steered_final_delta':float((full['final']-ordinary['final']).abs().max()),
                'token_first_margin_error':abs(full['full_margin']-expected)})
        eng.close()
    out=ROOT/'runs/20261003_m1_reinforcer/selfcheck';out.mkdir(parents=True,exist_ok=True)
    result={'no_GT':True,'norm_algebra':True,'zero_norm_cases':True,'actual_multimodal_Qwen':results}
    (out/'numerics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
