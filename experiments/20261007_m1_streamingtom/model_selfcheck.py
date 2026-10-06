"""Actual 36-layer collector/reader paths, native identity and physical clones."""
import copy
import json
import socket
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from cpu_fixture import fixture
from inputs import ROOT,local_ids
from memory import Memory
from collect import collect
from reader import visual_margin
from src.stance_cache import build,margin
from src.mllm_judge import yesno_question


def main():
    torch.set_num_threads(4);torch.manual_seed(0);print(socket.gethostname(),flush=True)
    out=ROOT/'runs/20261007_m1_streamingtom/model_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    native=out/'native.png';Image.fromarray(np.full((8,8,3),100,np.uint8)).save(native)
    rng=np.random.default_rng(0);frames=[];array=None
    for i in range(7):
        if i%2==0:array=rng.integers(0,256,(32,32,3),dtype=np.uint8)
        path=out/f'source_{i}.png';Image.fromarray(array).save(path)
        frames.append(dict(index=16*i,time=2.*i,path=str(path.relative_to(ROOT))))
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=512,hidden_size=64,intermediate_size=128,num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,head_dim=128,
        max_position_embeddings=4096,rope_parameters={'rope_type':'default','rope_theta':5000000.,'mrope_section':[24,20,20],'mrope_interleaved':True}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,
            out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa';cases=[]
    with torch.no_grad():
        for dtype in (torch.float32,torch.bfloat16):
            for count in (18,20):
                model=Qwen3VLModel(cfg).eval().to(dtype);model.requires_grad_(False);j,ff,hooks=fixture(model,count,dtype,native)
                folder=out/(str(dtype).split('.')[-1]+str(count));m=Memory(folder/'temporary')
                try:
                    cache,ctx=build(j,ff,[(0.,8.,'Fixture literal speech')]);initial=copy.deepcopy(cache);rope=j.model.model.rope_deltas.clone()
                    q= yesno_question(0,2,0,8,'Fixture literal speech','visual');s= yesno_question(0,2,0,8,'Fixture literal speech','speech')
                    nativev=margin(j,cache,ctx,q);natives=margin(j,cache,ctx,s)
                    acquisition=collect(j,cache,ctx,m,frames,folder/'proof')
                    assert acquisition['actual_forwards']==acquisition['actual_vision']==7
                    assert acquisition['records'][0]['plan']['dynamic_budget']==50
                    assert acquisition['records'][1]['plan']['static_budget']==50
                    assert m.blocks[6]['direct_ancestors']==[2,3,4,5] and m.blocks[6]['ancestors']==list(range(6))
                    local=local_ids(frames,dict(start=0,end=8));value,proof=visual_margin(j,cache,ctx,m,frames,local,q)
                    clone=copy.deepcopy(cache);again,other=visual_margin(j,clone,ctx,m,frames,local,q)
                    assert value==again and proof['input']==other['input']
                    for l in range(36):
                        a,b=proof['layers'][l],other['layers'][l]
                        assert torch.equal(a['query_vector'],b['query_vector'])
                        assert {k:v for k,v in a.items() if k!='query_vector'}=={k:v for k,v in b.items() if k!='query_vector'}
                        assert a['remote_ids'] and all(i not in local for i in a['remote_ids'])
                    assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,initial.layers))
                    assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(clone.layers,initial.layers))
                    assert torch.equal(rope,j.model.model.rope_deltas) and margin(j,cache,ctx,q)==nativev and margin(j,cache,ctx,s)==natives
                    assert all('forward' not in layer.self_attn.__dict__ for layer in model.language_model.layers)
                    cases.append(dict(dtype=str(dtype),native_frames=count,layers=36,Q_heads=32,KV_heads=8,dimension=128,
                        source_fullvision=7,source_sparse_LM=7,static_and_dynamic_paths_enter=True,visual_retained50=True,all3DeepStack=True,
                        actual_quantized_history_and_remote=True,source_ancestry_exact=True,one_query_forward_no_probe_or_vision=True,
                        native_G_S_all_KV_rope_exact=True,clones_exact=True,source_and_LOCAL_roles_bound=True))
                    print('MODEL_CASE_PASS',dtype,count,flush=True)
                finally:
                    m.close(release=True)
                    for hook in hooks:hook.remove()
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,pretrained=False,
        scope='actual36-layer random Qwen, explicit software tokenizer/renderer; not real tokenizer/pretrained source semantics or performance',cases=cases),indent=2)+'\n')
    print('MODEL_CPU_PASS',flush=True)


if __name__=='__main__':main()
