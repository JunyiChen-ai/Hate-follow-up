#!/usr/bin/env python3
"""No-GT CPU causal-isolation checks on an actual tiny Qwen3-VL text model."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from factorizer import PrefixFactorizer,assign_groups,allowed_edges
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


def main():
    torch.set_num_threads(1);torch.manual_seed(0)
    cfg=Qwen3VLTextConfig(vocab_size=128,hidden_size=64,intermediate_size=128,
      num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
      max_position_embeddings=512,rope_scaling={'rope_type':'default','mrope_section':[2,3,3]})
    cfg._attn_implementation='sdpa'
    # Shared membership at token 4 must belong only to the earliest window.
    region={'visual':np.array([0,1,1,0,0,0,0,0,0,0,0,0],bool),
            'speech':np.array([0,0,0,1,1,0,1,1,0,1,0,0],bool),
            'local':[np.array([0,1,1,1,1,0,0,0,0,0,0,0],bool),
                     np.array([0,0,0,0,1,0,1,1,0,0,0,0],bool)]}
    groups,mapping=assign_groups(region)
    assert groups.tolist()==[-1,0,0,0,0,-1,1,1,-1,2,-1,-1]
    edges=allowed_edges(groups).numpy();assert not np.triu(edges,1).any() and np.diag(edges).all()
    # Reachability checks all possible indirect paths, not just the direct mask.
    reach=edges.copy()
    for k in range(len(groups)):reach|=reach[:,k,None] & reach[k,None,:]
    for q in range(len(groups)):
        for k in range(len(groups)):
            if groups[q]<0 and groups[k]>=0:assert not reach[q,k]
            if groups[q]>=0 and groups[k]>=0 and groups[q]!=groups[k]:assert not reach[q,k]
    checks=[]
    for dtype in (torch.float32,torch.bfloat16):
        lm=Qwen3VLTextModel(cfg).eval().to(dtype=dtype)
        for p in lm.parameters():p.requires_grad_(False)
        j=SimpleNamespace(family='qwen3_vl',device=torch.device('cpu'),dtype=dtype,
            model=SimpleNamespace(model=SimpleNamespace(language_model=lm)))
        engine=PrefixFactorizer(j);ids=torch.arange(12)[None]
        original={k:v.clone() for k,v in lm.state_dict().items()}
        with torch.no_grad():
            native=lm(input_ids=ids,use_cache=True)
            with engine.encoding(groups,native=True):explicit=lm(input_ids=ids,use_cache=True)
            tolerance=1e-5 if dtype==torch.float32 else 0.
            parity_error=float((native.last_hidden_state-explicit.last_hidden_state).abs().max())
            parity_kv_error=max(float((getattr(a,kind)-getattr(b,kind)).abs().max())
                for a,b in zip(native.past_key_values.layers,explicit.past_key_values.layers) for kind in ('keys','values'))
            # CPU SDPA chooses different paths for implicit and explicit masks.
            # Record that drift, then hold the kernel fixed to test mask parity.
            with sdpa_kernel(SDPBackend.MATH):
                math_native=lm(input_ids=ids,use_cache=True)
                with engine.encoding(groups,native=True):math_explicit=lm(input_ids=ids,use_cache=True)
            math_error=float((math_native.last_hidden_state-math_explicit.last_hidden_state).abs().max())
            torch.testing.assert_close(math_native.last_hidden_state,math_explicit.last_hidden_state,atol=tolerance,rtol=0.)
            for a,b in zip(math_native.past_key_values.layers,math_explicit.past_key_values.layers):
                torch.testing.assert_close(a.keys,b.keys,atol=tolerance,rtol=0.)
                torch.testing.assert_close(a.values,b.values,atol=tolerance,rtol=0.)
            with engine.encoding(groups):base=lm(input_ids=ids,use_cache=True)
            changed_ids=ids.clone();changed_ids[0,1:5]+=30
            with engine.encoding(groups):changed=lm(input_ids=changed_ids,use_cache=True)
            unaffected=torch.tensor(groups!=0);affected=torch.tensor(groups==0)
            assert torch.equal(base.last_hidden_state[:,unaffected],changed.last_hidden_state[:,unaffected])
            assert not torch.equal(base.last_hidden_state[:,affected],changed.last_hidden_state[:,affected])
            for a,b in zip(base.past_key_values.layers,changed.past_key_values.layers):
                assert torch.equal(a.keys[:,:,unaffected],b.keys[:,:,unaffected])
                assert torch.equal(a.values[:,:,unaffected],b.values[:,:,unaffected])
            changed_native=lm(input_ids=changed_ids,use_cache=True)
            assert not torch.equal(native.last_hidden_state[:,6:],changed_native.last_hidden_state[:,6:])
            # At query time the model is again allowed to use all groups.
            a=lm(input_ids=torch.tensor([[40,41]]),past_key_values=base.past_key_values,use_cache=True)
            b=lm(input_ids=torch.tensor([[40,41]]),past_key_values=changed.past_key_values,use_cache=True)
            assert not torch.equal(a.last_hidden_state,b.last_hidden_state)
            restored=lm(input_ids=ids,use_cache=True)
            assert torch.equal(native.last_hidden_state,restored.last_hidden_state)
        assert all(torch.equal(v,original[k]) for k,v in lm.state_dict().items())
        engine.close();checks.append({'dtype':str(dtype),'native_mask_max_abs_diff':parity_error,
          'fixed_math_kernel_mask_atol':tolerance,
          'native_mask_KV_max_abs_diff':parity_kv_error,
          'fixed_math_kernel_mask_max_abs_diff':math_error,
          'all_layer_other_group_KV_exact':True,'scaffold_hidden_exact':True,'native_is_sensitive':True,
          'query_context_remains_sensitive':True,'weights_and_native_restored':True})
    out=ROOT/'runs/20261003_m1_factorizer/selfcheck';out.mkdir(parents=True,exist_ok=True)
    result={'no_GT':True,'mapping':mapping,'transitive_no_bridge':True,'checks':checks}
    (out/'invariance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
