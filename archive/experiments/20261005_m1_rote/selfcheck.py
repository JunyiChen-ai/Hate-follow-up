#!/usr/bin/env python3
"""Numerical integral, independent bias oracle, actual36-layer Qwen/cache checks."""
import copy
import inspect
import json
import math
import socket
from types import SimpleNamespace
import torch
from rote import ROOT, SPEC, coefficients, rotate_pairs, temporal_pairs, IntervalAttention
from src.mllm_judge import Judge
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel


def scalar_coeff(intervals, frequencies):
    cos=[];sin=[]
    for start,end in intervals:
        center=4*(start+end)/2;radius=4*(end-start)/2
        sinc=[math.sin(float(w)*radius)/(float(w)*radius) if radius else 1. for w in frequencies]
        norm=sum(sinc)/len(sinc)
        cos.append([s*math.cos(float(w)*center)/norm for s,w in zip(sinc,frequencies)])
        sin.append([s*math.sin(float(w)*center)/norm for s,w in zip(sinc,frequencies)])
    return torch.tensor(cos),torch.tensor(sin)


def manual_rotate(x,cos,sin):
    n=x.shape[-1]//2
    result=torch.empty_like(x,dtype=torch.float32)
    for i in range(n):
        result[...,i]=x[...,i].float()*cos[...,i]-x[...,i+n].float()*sin[...,i]
        result[...,i+n]=x[...,i].float()*sin[...,i]+x[...,i+n].float()*cos[...,i]
    return result


def fixture(dtype,frames):
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=36,
        num_attention_heads=32,num_key_value_heads=8,head_dim=128,max_position_embeddings=1024,
        rope_parameters={'rope_type':'default','rope_theta':5000000.,'mrope_section':[24,20,20],'mrope_interleaved':True}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,
            temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1]),
        image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa'
    model=Qwen3VLModel(cfg).eval().to(dtype)
    # Module.to(BF16) also casts this random fixture's positional buffer. HF's
    # native default rotary construction uses explicit FP32 inverse frequencies.
    rotary=model.language_model.rotary_emb
    native_frequency,_=rotary.compute_default_rope_parameters(cfg.text_config)
    rotary.inv_freq=native_frequency
    rotary.original_inv_freq=native_frequency.clone()
    head=torch.nn.Linear(64,256,bias=False).to(dtype)
    j=Judge.__new__(Judge);j.device=torch.device('cpu');j.dtype=dtype;j.image_token_id=127
    j.model=SimpleNamespace(model=model,config=cfg,get_output_embeddings=lambda:head)
    j.forward_params=set(inspect.signature(model.forward).parameters)
    j.yes_ids=[170];j.no_ids=[171];j.softcap=None
    ids=torch.tensor([[10]+[v for f in range(frames) for v in [11,125,127,127,127,127,126]]+[40,41,42,43,44,45,46]])
    enc=dict(input_ids=ids,pixel_values=torch.randn(frames*16,24).to(dtype),image_grid_thw=torch.tensor([[1,4,4]]*frames),attention_mask=torch.ones_like(ids))
    if 'mm_token_type_ids' in j.forward_params:enc['mm_token_type_ids']=(ids==127).long()
    return j,enc


def main():
    torch.set_num_threads(1);torch.manual_seed(0);checks=[]
    # Integrate complex rotations using a dense trapezoid, independently of sinc.
    freq=torch.tensor([1.,.41,.12,.007],dtype=torch.float32)
    intervals=[[0.,0.],[1.,2.5],[.1,3.],[7.,7.01]]
    c,s,norm=coefficients(intervals,freq)
    for i,(start,end) in enumerate(intervals):
        points=torch.linspace(4*start,4*end,16385,dtype=torch.float64)
        phase=points[:,None]*freq.double()[None]
        if start==end:cc=phase[0].cos();ss=phase[0].sin()
        else:
            cc=torch.trapezoid(phase.cos(),points,dim=0)/(4*(end-start))
            ss=torch.trapezoid(phase.sin(),points,dim=0)/(4*(end-start))
        assert torch.max(torch.abs(c[i].double()-cc/norm[i]))<1e-6
        assert torch.max(torch.abs(s[i].double()-ss/norm[i]))<1e-6
    x=torch.randn(2,3,4,8)
    assert torch.allclose(rotate_pairs(x,c[None,None],s[None,None]),manual_rotate(x,c[None,None],s[None,None]),atol=1e-7)
    assert torch.equal(coefficients([[3.,3.]],freq)[2],torch.ones(1,dtype=torch.float64))
    try:coefficients([[2.,1.]],freq)
    except AssertionError:pass
    else:raise AssertionError('negative interval accepted')
    checks.append(dict(integral_quadrature=True,point_limit=True,invalid_interval_rejected=True))
    with torch.no_grad():
        for dtype in (torch.float32,torch.bfloat16):
            for frames in (18,20):
                j,enc=fixture(dtype,frames);model=j.model.model
                model.rope_deltas=None
                baseline=j.prefix_cache(enc);j.extend_cache(baseline,[60,61]);j.extend_cache(baseline,[62])
                rope=model.rope_deltas.clone();length=baseline.get_seq_length()
                old=j.cached_margin(baseline,[63,64,65,66,67,68],in_place=True);baseline.crop(length)
                model.rope_deltas=None
                ctrl=IntervalAttention(j)
                raw_q={};raw_k={};captured_keys={};hooks=[]
                for number,layer in enumerate(ctrl.layers):
                    hooks.append(layer.self_attn.q_norm.register_forward_hook(lambda m,i,o,n=number:raw_q.__setitem__(n,o.detach().transpose(1,2))))
                    hooks.append(layer.self_attn.k_norm.register_forward_hook(lambda m,i,o,n=number:raw_k.__setitem__(n,o.detach().transpose(1,2))))
                prefix_length=enc['input_ids'].shape[1]
                index=[prefix_length-6,prefix_length-4,prefix_length-2]
                ctrl.prefix_indices=torch.tensor(index);ctrl.prefix_intervals=[[1.,4.],[4.,7.],[7.,9.]]
                ctrl.capture=True;cache=j.prefix_cache(enc);ctrl.capture=False
                for n in raw_k:captured_keys[n]=raw_k[n][:,:,index].clone()
                j.extend_cache(cache,[60,61]);j.extend_cache(cache,[62])
                assert torch.equal(model.rope_deltas,rope)
                assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,baseline.layers))
                model.rope_deltas=rope.clone();native=j.cached_margin(cache,[63,64,65,66,67,68],in_place=True);cache.crop(length)
                assert native==old
                mapping=[dict(token=0,source_ids=[0]),dict(token=2,source_ids=[1])]
                segments=[(1.,4.,'a'),(7.,9.,'b')]
                intervals=ctrl.prefix_intervals+[[1.,4.],[7.,9.]]
                independent_cos,independent_sin=scalar_coeff(intervals,ctrl.frequencies)
                phase=torch.tensor([4*6*float(w) for w in ctrl.frequencies])
                independent_channels=torch.tensor([v for v in range(64) if v%3==0 or v>=60]+[64+v for v in range(64) if v%3==0 or v>=60])
                assert torch.equal(independent_channels,ctrl.channels)
                expected_key_indices=torch.tensor(index+[length,length+2])
                actual_attention=ctrl.native_attention;oracle_calls=[0];maximum_error=[0.]
                def oracle(module,query,key,value,attention_mask,**kw):
                    if 'position_bias' in kw:
                        layer=module.layer_idx
                        qraw=raw_q[layer][...,independent_channels]
                        kraw=torch.cat((captured_keys[layer][...,independent_channels],raw_k[layer][:,:,[0,2]][...,independent_channels]),2)
                        nq=manual_rotate(qraw,phase.cos(),phase.sin())
                        nk=manual_rotate(kraw,independent_cos[None,None],independent_sin[None,None]).repeat_interleave(4,dim=1)
                        oq=query[...,independent_channels].float()
                        ok=key[:,:,expected_key_indices][...,independent_channels].float().repeat_interleave(4,dim=1)
                        difference=(torch.einsum('bhtd,bhkd->bhtk',nq,nk)-torch.einsum('bhtd,bhkd->bhtk',oq,ok))*module.scaling
                        expected=torch.zeros_like(kw['position_bias']);expected[:,:,:,expected_key_indices]=difference.to(expected.dtype)
                        error=float((expected.float()-kw['position_bias'].float()).abs().max());maximum_error[0]=max(maximum_error[0],error)
                        assert error<=SPEC['cpu_reference_tolerance']['bf16' if dtype==torch.bfloat16 else 'fp32']
                        other=torch.ones(key.shape[2],dtype=torch.bool);other[expected_key_indices]=False
                        assert torch.count_nonzero(kw['position_bias'][...,other])==0
                        assert attention_mask.shape[-2:]==(query.shape[2],key.shape[2])
                        # Future suffix keys stay invisible despite the ASR bias.
                        for i in range(query.shape[2]-1):
                            future=attention_mask[...,i,length+i+1:]
                            assert ((~future).all() if future.dtype==torch.bool else (future< -1e20).all())
                        oracle_calls[0]+=1
                    return actual_attention(module,query,key,value,attention_mask,**kw)
                ctrl.native_attention=oracle
                model.rope_deltas=rope.clone()
                with ctrl.speech(length,mapping,segments,4.,8.):new=j.cached_margin(cache,[63,64,65,66,67,68],in_place=True)
                cache.crop(length);assert oracle_calls[0]==36
                clone=copy.deepcopy(cache);model.rope_deltas=rope.clone()
                with ctrl.speech(length,mapping,segments,4.,8.):repeat=j.cached_margin(clone,[63,64,65,66,67,68],in_place=True)
                clone.crop(length);assert new==repeat and new!=native
                assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,clone.layers))
                model.rope_deltas=rope.clone();after=j.cached_margin(cache,[63,64,65,66,67,68],in_place=True);cache.crop(length);assert after==native
                for hook in hooks:hook.remove()
                ctrl.close();assert j.model.config.text_config._attn_implementation=='sdpa'
                checks.append(dict(dtype=str(dtype),frames=frames,layers=36,oracle_calls=oracle_calls[0],bias_max_error=maximum_error[0],
                    native_allKV_margin_exact=True,clone_allKV_margin_exact=True,nonASR_bias_zero=True,causal_visibility_preserved=True,new_margin_changed=True))
    out=ROOT/'runs/20261005_m1_rote/cpu_checks';out.mkdir(parents=True,exist_ok=True)
    result=dict(GT_read=False,host=socket.gethostname(),pass_all=True,checks=checks)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('CPU_CHECKS_PASS',flush=True)


if __name__=='__main__':main()
