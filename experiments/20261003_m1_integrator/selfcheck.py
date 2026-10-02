#!/usr/bin/env python3
"""CPU witness that later input reaches visual memory via the new edges."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from integrator import PrefixIntegrator,allowed_edges
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


def main():
    torch.set_num_threads(1);torch.manual_seed(0)
    cfg=Qwen3VLTextConfig(vocab_size=128,hidden_size=64,intermediate_size=128,
        num_hidden_layers=4,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
        max_position_embeddings=512,rope_scaling={'rope_type':'default','mrope_section':[2,3,3]})
    cfg._attn_implementation='sdpa'
    visual=torch.tensor([0,1,1,0,0,1,1,0,0,0,0,0],dtype=torch.bool)
    edges=allowed_edges(visual);causal=allowed_edges(visual,native=True)
    assert torch.equal(edges[~visual],causal[~visual]) and edges[visual].all()
    assert (edges | causal).equal(edges) and torch.diag(edges).all()
    checks=[]
    for dtype in (torch.float32,torch.bfloat16):
        lm=Qwen3VLTextModel(cfg).eval().to(dtype=dtype)
        for p in lm.parameters():p.requires_grad_(False)
        judge=SimpleNamespace(family='qwen3_vl',device=torch.device('cpu'),dtype=dtype,
            model=SimpleNamespace(model=SimpleNamespace(language_model=lm)))
        engine=PrefixIntegrator(judge);ids=torch.arange(12)[None]
        original={k:v.clone() for k,v in lm.state_dict().items()}
        with torch.no_grad():
            native=lm(input_ids=ids,use_cache=True)
            with engine.encoding(visual,native=True):explicit=lm(input_ids=ids,use_cache=True)
            drift=float((native.last_hidden_state-explicit.last_hidden_state).abs().max())
            with sdpa_kernel(SDPBackend.MATH):
                math_native=lm(input_ids=ids,use_cache=True)
                with engine.encoding(visual,native=True):math_explicit=lm(input_ids=ids,use_cache=True)
            torch.testing.assert_close(math_native.last_hidden_state,math_explicit.last_hidden_state,atol=1e-5 if dtype==torch.float32 else 0,rtol=0)
            with engine.encoding(visual):future=lm(input_ids=ids,use_cache=True)
            altered=ids.clone();altered[0,10]+=50
            changed_native=lm(input_ids=altered,use_cache=True)
            with engine.encoding(visual):changed_future=lm(input_ids=altered,use_cache=True)
            assert torch.equal(native.last_hidden_state[:,visual],changed_native.last_hidden_state[:,visual])
            assert not torch.equal(future.last_hidden_state[:,visual],changed_future.last_hidden_state[:,visual])
            # Cached K/V change after the first layer, not just the final hidden output.
            cache_delta=max(float((a.values[:,:,visual]-b.values[:,:,visual]).abs().max())
                for a,b in zip(future.past_key_values.layers[1:],changed_future.past_key_values.layers[1:]))
            assert cache_delta>0 and engine.active is None
            visited=list(engine.visited)
            a=lm(input_ids=torch.tensor([[40,41]]),past_key_values=future.past_key_values,use_cache=True)
            b=lm(input_ids=torch.tensor([[40,41]]),past_key_values=changed_future.past_key_values,use_cache=True)
            assert engine.visited==visited and not torch.equal(a.last_hidden_state,b.last_hidden_state)
            restored=lm(input_ids=ids,use_cache=True)
            assert torch.equal(native.last_hidden_state,restored.last_hidden_state)
        assert all(torch.equal(v,original[k]) for k,v in lm.state_dict().items())
        engine.close();checks.append({'dtype':str(dtype),'implicit_explicit_drift':drift,
            'fixed_kernel_parity':True,'native_visual_independent_of_later_text':True,
            'future_visual_sensitive_to_later_text':True,'later_layer_cached_value_change':cache_delta,
            'suffix_unhooked_and_sensitive':True,'native_restored_exact':True,'weights_unchanged':True})
    out=ROOT/'runs/20261003_m1_integrator/selfcheck';out.mkdir(parents=True,exist_ok=True)
    result={'no_GT':True,'visual_only_new_direct_edges':True,'checks':checks}
    (out/'future_visibility.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
