"""Actual three-grain/full-history/DCP/probe/LOCAL-context Qwen CPU paths."""
import copy
import json
import socket
import numpy as np
from PIL import Image
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from inputs import ROOT
from memory import Memory
from collect import collect
from reader import probe,visual_margin
from src.qwen_synthetic_fixture import fixture
from src.stance_cache import build,margin
from src.mllm_judge import yesno_question


def main():
    torch.set_num_threads(4);torch.manual_seed(0);print(socket.gethostname(),flush=True)
    out=ROOT/'runs/20261007_m1_mukv/model_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    native=out/'native.png';Image.fromarray(np.full((8,8,3),100,np.uint8)).save(native);rng=np.random.default_rng(0);windows=[]
    for w in range(3):
        frames=[]
        for k,t in enumerate((1.,3.,5.,7.)):
            i=4*w+k;path=out/f'source_{i}.png';Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(path)
            frames.append(dict(index=10*i,time=8*w+t,path=str(path.relative_to(ROOT))))
        windows.append(frames)
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=512,hidden_size=64,intermediate_size=128,num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,head_dim=128,
        max_position_embeddings=8192,rope_parameters={'rope_type':'default','rope_theta':5000000.,'mrope_section':[24,20,20],'mrope_interleaved':True}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,
            out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa';cases=[]
    with torch.no_grad():
        for dtype,count in ((torch.float32,18),(torch.bfloat16,20)):
            model=Qwen3VLModel(cfg).eval().to(dtype);model.requires_grad_(False);j,ff,hooks=fixture(model,count,dtype,native);folder=out/(str(dtype).split('.')[-1]+str(count));memory=Memory(folder/'temporary')
            try:
                cache,ctx=build(j,ff,[(0.,24.,'Fixture literal speech')]);initial=copy.deepcopy(cache);oldrope=j.model.model.rope_deltas.clone()
                q=yesno_question(2,3,16.,24.,'Fixture literal speech','visual');s=yesno_question(2,3,16.,24.,'Fixture literal speech','speech')
                beforev=margin(j,cache,ctx,q);befores=margin(j,cache,ctx,s)
                source,cached=collect(j,cache,ctx,memory,windows,folder/'proof');assert source['actual_LM']==18 and source['actual_vision']==12
                assert len(memory.blocks)==18
                for b in memory.blocks:
                    prior=[a['id'] for a in memory.blocks if a['window']<b['window'] and a['grain']==b['grain']]
                    assert b['direct_ancestors']==prior
                    assert all(memory.blocks[a]['window']<b['window'] for a in b['direct_ancestors'])
                selected,retrieval=probe(j,cache,ctx,memory,2,windows[2],cached,'Fixture literal speech')
                assert len(selected)==6 and all(memory.blocks[i]['window']<2 for i in selected)
                value,proof=visual_margin(j,cache,ctx,memory,windows[2],cached,selected,q)
                clone=copy.deepcopy(cache);again,_=visual_margin(j,clone,ctx,memory,windows[2],cached,selected,q);assert value==again
                assert margin(j,cache,ctx,q)==beforev and margin(j,cache,ctx,s)==befores
                assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,initial.layers))
                assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(clone.layers,initial.layers))
                assert torch.equal(oldrope,j.model.model.rope_deltas) and all('forward' not in l.self_attn.__dict__ for l in model.language_model.layers)
                cases.append(dict(dtype=str(dtype),native_frames=count,layers=36,Q_heads=32,KV_heads=8,head_dim=128,source_LM=18,source_vision=12,
                    actual_three_grains_four_spatial_quadrants=True,all_earlier_samegrain_history_sibling_exclusion=True,paid_lastLM_attention_FFT_sourcepruning=True,
                    one_unscored_probe_no_vision=True,coherent6contextblocks=True,uncompressed_LOCAL_all3DeepStack=True,nativeS_allKV_rope_clones_exact=True))
                print('MODEL_CASE_PASS',dtype,count,flush=True)
            finally:
                memory.close(release=True)
                for hook in hooks:hook.remove()
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,pretrained=False,cases=cases,
        scope='actual36layer random Qwen with shared explicit fixture tokenizer/renderer; source/reader computations only, not realprocessor/8B/semantic/effectiveness'),indent=2)+'\n');print('MODEL_CPU_PASS',flush=True)


if __name__=='__main__':main()
