"""CPU fixture check of placement_controls (random-weight 36-layer Qwen3-VL shape); no annotations, no pretrained claim."""
import json
import socket
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from inputs import ROOT
from cpu_fixture import fixture
from placement_controls import read,ARMS,LOCAL_LABEL,NATIVE_LABEL


def main():
    torch.set_num_threads(4);torch.manual_seed(0);print(socket.gethostname(),flush=True)
    out=ROOT/'runs/20261007_m1_streamingtom/placement_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    native=out/'native.png';Image.fromarray(np.full((8,8,3),100,np.uint8)).save(native);rng=np.random.default_rng(0);frames=[]
    for i in range(7):
        if i%2==0:image=rng.integers(0,256,(32,32,3),dtype=np.uint8)
        path=out/f'source_{i}.png';Image.fromarray(image).save(path);frames.append(dict(index=16*i,time=2.*i,path=str(path.relative_to(ROOT))))
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=512,hidden_size=64,intermediate_size=128,num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,head_dim=128,
        max_position_embeddings=4096,rope_parameters={'rope_type':'default','rope_theta':5000000.,'mrope_section':[24,20,20],'mrope_interleaved':True}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,
            out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa'
    row=dict(dataset='HateMM',video_id='fixture',duration=24.);segments=[(0.,8.,'Fixture speech'),(8.,16.,'Later speech')]
    source=dict(decode_seconds=0.,fixture=True);cases=[]
    with torch.no_grad():
        for dtype,count,limit in ((torch.float32,20,1e-4),(torch.bfloat16,20,.1)):
            model=Qwen3VLModel(cfg).eval().to(dtype);model.requires_grad_(False);j,ff,hooks=fixture(model,count,dtype,native)
            try:
                with patch('placement_controls.frames_for',return_value=(source,frames)),patch('placement_controls.frame_paths',return_value=ff),patch('placement_controls.SPEC',{**__import__('placement_controls').SPEC,'native_frames':count}):
                    b=read(j,row,segments,None)
                g=b['diagnostic'];assert g['tokens_equal'] and g['branch_ids_equal'] and g['image_counts_equal'],g
                assert g['abs_difference']<=limit,('standalone native far from cached',g['abs_difference'])
                tr=b['traces'];assert len(tr)==3 and [t['LOCAL'] for t in tr]==[[0,1,2,3],[4,5,6],[]]
                assert all(t['native_inside'] for t in tr)
                for t in tr:
                    n=len(t['LOCAL'])
                    for a in ('local_clean_replay','adjacent_local','prefix_local'):
                        assert (t[a] is not None)==bool(n)
                    assert t['adjacent_native'] is not None and t['adjacent_native']['labels']==[NATIVE_LABEL.format(time=x) for x in t['native_inside']]
                    if n:
                        assert t['local_clean_replay']['labels']==[LOCAL_LABEL.format(time=frames[i]['time']) for i in t['LOCAL']]
                        assert t['adjacent_local']['labels']==[NATIVE_LABEL.format(time=frames[i]['time']) for i in t['LOCAL']]
                        assert t['prefix_local']['images']==count+n and len(t['prefix_local']['inserted_positions'])==n
                        # Same images, different label text or position: the fixture read must move.
                        assert t['local_clean_replay']['z']!=t['native_visual'] and t['adjacent_local']['z']!=t['native_visual'] and t['prefix_local']['z']!=t['native_visual']
                    for a in ARMS:
                        z=b['predictions'][a]['extra']['windows'][t['i']]['z_visual'];assert z==(t[a]['z'] if t[a] else t['native_visual'])
                assert b['checks']['actual_forwards']==b['base']['calls']+3*2+3+1 and b['checks']['actual_vision']==1+3*2+3+1
                (out/(str(dtype).split('.')[-1]+'.json')).write_text(json.dumps(b)+'\n')
                cases.append(dict(dtype=str(dtype),native_frames=count,standalone_native_abs=g['abs_difference'],tokens_equal=True,labels_and_counts=True))
                print('PLACEMENT_CASE_PASS',dtype,count,g['abs_difference'],flush=True)
            finally:
                for hook in hooks:hook.remove()
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,pretrained=False,cases=cases,scope='random-weight 36-layer fixture; not real 8B/semantic/performance evidence'),indent=2)+'\n')
    print('PLACEMENT_CPU_PASS',flush=True)


if __name__=='__main__':main()
