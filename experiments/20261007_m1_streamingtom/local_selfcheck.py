"""CPU fixture check of local_controls against production R1 and control #1; no annotations."""
import json
import socket
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from inputs import ROOT,SPEC
from cpu_fixture import fixture
from measure import read_video
from controls import read as controls_read
from local_controls import read


def main():
    torch.set_num_threads(4);torch.manual_seed(0);print(socket.gethostname(),flush=True)
    out=ROOT/'runs/20261007_m1_streamingtom/local_cpu_checks';out.mkdir(parents=True,exist_ok=True)
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
        for dtype,count,limit in ((torch.float32,18,1e-4),(torch.bfloat16,20,.1)):
            model=Qwen3VLModel(cfg).eval().to(dtype);model.requires_grad_(False);j,ff,hooks=fixture(model,count,dtype,native);opened=[]
            folder=out/(str(dtype).split('.')[-1]+str(count))
            try:
                with (patch('measure.frames_for',return_value=(source,frames)),patch('controls.frames_for',return_value=(source,frames)),
                    patch('local_controls.frames_for',return_value=(source,frames)),patch('measure.frame_paths',return_value=ff),
                    patch('controls.frame_paths',return_value=ff),patch('local_controls.frame_paths',return_value=ff),
                    patch('src.native_input_binding.frame_paths',return_value=ff)):
                    r1,memory=read_video(j,row,segments,folder/'r1',True);opened.append(memory);memory.close(release=True);opened.remove(memory)
                    dual,memory=controls_read(j,row,segments,folder/'dualpath','dualpath',r1);opened.append(memory);memory.close(release=True);opened.remove(memory)
                    b=read(j,row,segments,r1,dual)
                    assert all(t['replay_exact'] for t in b['traces']),'no_remote_replay must reproduce control #1'
                    gap=max(abs(t['custom_native']-t['native_visual']) for t in b['traces'])
                    assert gap<=limit,('custom code path far from native',gap)
                    covered=[t for t in b['traces'] if t['LOCAL']];assert covered and len(covered)<len(b['traces'])
                    for t in covered:
                        e=t['local_clean'];r=t['no_remote_replay'];n=len(t['LOCAL'])
                        assert not e['role_in_suffix'] and r['role_in_suffix'] and e['time_labels']==r['time_labels']==n and len(e['image_counts'])==n
                    assert any(t['local_clean']['z']!=t['no_remote_replay']['z'] for t in covered),'role text must enter the fixture read'
                    for t in b['traces']:
                        if not t['LOCAL']:assert t['local_clean'] is None and t['no_remote_replay'] is None
                    for a in ('local_clean','no_remote_replay'):
                        assert [w['z_visual'] for w in b['predictions'][a]['extra']['windows']]==[t[a]['z'] if t['LOCAL'] else t['native_visual'] for t in b['traces']]
                    assert [w['z_visual'] for w in b['predictions']['custom_native']['extra']['windows']]==[t['custom_native'] for t in b['traces']]
                    (folder/'local.json').write_text(json.dumps(b)+'\n')
                    cases.append(dict(dtype=str(dtype),native_frames=count,replay_equals_control1=True,custom_native_max_abs_vs_native=gap,
                        role_text_absent_in_E=True,time_labels_and_images_in_E=True))
                    print('LOCAL_CASE_PASS',dtype,count,gap,flush=True)
            finally:
                for memory in opened:memory.close(release=True)
                for hook in hooks:hook.remove()
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,pretrained=False,cases=cases,
        scope='random-weight 36-layer fixture; not real 8B/semantic/performance evidence'),indent=2)+'\n')
    print('LOCAL_CPU_PASS',flush=True)


if __name__=='__main__':main()
