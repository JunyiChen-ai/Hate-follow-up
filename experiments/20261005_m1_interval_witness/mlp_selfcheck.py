"""Observe full 36-layer and long tokenwise MLP equivalence, CPU only."""
import copy
import json
import os
from pathlib import Path
import socket
import sys
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.qwen3_mlp_memory import chunked_mlp
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel


def main():
    torch.set_num_threads(1);torch.manual_seed(0); results=[]
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=36,
        num_attention_heads=32,num_key_value_heads=8,head_dim=128,max_position_embeddings=512,
        rope_scaling={'rope_type':'default','mrope_section':[24,20,20]}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,
            temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1]),
        image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa'
    with torch.no_grad():
        for dtype in (torch.float32,torch.bfloat16):
            model=Qwen3VLModel(cfg).eval().to(dtype)
            ids=torch.randint(1,100,(1,73));original=[l.mlp.forward for l in model.language_model.layers]
            plain=model(input_ids=ids,use_cache=True);model.rope_deltas=None
            with chunked_mlp(model,17): chunk=model(input_ids=ids,use_cache=True)
            assert all(l.mlp.forward==f for l,f in zip(model.language_model.layers,original))
            assert torch.equal(plain.last_hidden_state,chunk.last_hidden_state)
            assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(plain.past_key_values.layers,chunk.past_key_values.layers))
            x=torch.randn(1,4097,64).to(dtype); max_diff=0.
            for layer in model.language_model.layers:
                direct=layer.mlp(x)
                with chunked_mlp(model,4096): small=layer.mlp(x)
                max_diff=max(max_diff,float((direct.float()-small.float()).abs().max()))
                assert torch.equal(direct,small) if dtype==torch.bfloat16 else torch.allclose(direct,small,rtol=0,atol=1e-6)
            try:
                with chunked_mlp(model,17):raise RuntimeError('fixture restoration')
            except RuntimeError:pass
            assert all(l.mlp.forward==f for l,f in zip(model.language_model.layers,original))
            results.append(dict(dtype=str(dtype),layers=36,full_hidden_and_KV_exact=True,actual4097_row_all_MLP_BF16_exact=(dtype==torch.bfloat16),FP32_long_row_tolerance=1e-6,max_difference=max_diff,finally_restored=True))
    out=ROOT/'runs/20261005_m1_interval_witness/prefill_mlp_fix/author';out.mkdir(parents=True,exist_ok=True)
    (out/'run.pid').write_text(str(os.getpid()));(out/'summary.json').write_text(json.dumps(dict(host=socket.gethostname(),GT_read=False,pass_all=True,checks=results),indent=2)+'\n')
    print(json.dumps(results,indent=2));print('MLP_CPU_PASS',flush=True)


if __name__=='__main__':main()
