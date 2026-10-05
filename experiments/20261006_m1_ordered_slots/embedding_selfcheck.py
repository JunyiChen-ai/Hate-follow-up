"""Actual native tokens/36 layers; random CPU weights, no benchmark labels."""
import json
from types import SimpleNamespace
import torch
from embedding import embed,encode_content,validate_embedding
from retrieval import ROOT
from src.mllm_renderer import cpu_renderer
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel


def main():
    j=cpu_renderer();torch.set_num_threads(1);torch.manual_seed(0)
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=len(j.tok),hidden_size=64,intermediate_size=128,num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,head_dim=128,max_position_embeddings=512,rope_scaling={'rope_type':'default','mrope_section':[24,20,20]}),vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1]),image_token_id=j.image_token_id,video_token_id=151656,vision_start_token_id=151652,vision_end_token_id=151653)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa';results=[]
    with torch.no_grad():
        for dtype in (torch.float32,torch.bfloat16):
            model=Qwen3VLModel(cfg).eval().to(dtype);j.model=SimpleNamespace(model=model);j.dtype=dtype;j.forward_calls=0
            hook=model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1));j.model_inputs=lambda enc:enc
            vectors=[]
            for text in ('A cup moves.','A visible sign changes.','Represent the supplied observable video content for semantic retrieval.'):
                encoded,evidence=encode_content(j,text);old=torch.tensor([[19]]);model.rope_deltas=old
                result=embed(j,text);assert model.rope_deltas is old;validate_embedding(j,result,text)
                model.rope_deltas=None;out=model(**encoded,use_cache=False)
                expected=out.last_hidden_state[0,evidence['pooled_token_indices']].float().mean(0);expected=expected/expected.norm()
                assert torch.equal(torch.tensor(result['vector']),expected.cpu());vectors.append(result['vector'])
            assert vectors[0]!=vectors[1]
            hook.remove();results.append(dict(dtype=str(dtype),layers=36,content_pool_exact=True,rope_restore_exact=True,different_content_vectors=True,calls=len(vectors)))
    out=ROOT/'runs/20261006_m1_ordered_slots/embedding_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,scope='actual36layer CPU and native tokenizer pooling; randomweights/noGT/performance',checks=results),indent=2)+'\n')
    print('EMBEDDING_CPU_PASS',flush=True)


if __name__=='__main__':main()
