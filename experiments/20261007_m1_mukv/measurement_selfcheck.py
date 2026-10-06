"""Complete production measurement and permanent-proof CPU replay."""
import copy
import json
import socket
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from inputs import ROOT
from measure import read_video
from validate import validate_bundle
from src.qwen_synthetic_fixture import fixture


def main():
    torch.set_num_threads(4);torch.manual_seed(0);print(socket.gethostname(),flush=True)
    out=ROOT/'runs/20261007_m1_mukv/measurement_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    native=out/'native.png';Image.fromarray(np.full((8,8,3),100,np.uint8)).save(native);rng=np.random.default_rng(0);windows=[]
    for w in range(2):
        media=[]
        for k,t in enumerate((1.,3.,5.,7.)):
            i=w*4+k;path=out/f'source_{i}.png';Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(path)
            media.append(dict(index=i*10,time=w*8+t,path=str(path.relative_to(ROOT))))
        windows.append(media)
    windows.append([]);source=dict(decode_seconds=0.,fixture=True);row=dict(dataset='HateMM',video_id='fixture',duration=24.);segments=[(0.,8.,'Fixture speech')]
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=512,hidden_size=64,intermediate_size=128,num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,head_dim=128,
        max_position_embeddings=8192,rope_parameters={'rope_type':'default','rope_theta':5000000.,'mrope_section':[24,20,20],'mrope_interleaved':True}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,
            out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa';cases=[]
    with torch.no_grad():
        for dtype,count in ((torch.float32,18),(torch.bfloat16,20)):
            model=Qwen3VLModel(cfg).eval().to(dtype);model.requires_grad_(False);j,ff,hooks=fixture(model,count,dtype,native);memory=None;folder=out/(str(dtype).split('.')[-1]+str(count))
            try:
                with (patch('measure.frames_for',return_value=(source,windows)),patch('validate.frames_for',return_value=(source,windows)),
                    patch('measure.frame_paths',return_value=ff),patch('src.native_input_binding.frame_paths',return_value=ff)):
                    bundle,memory=read_video(j,row,segments,folder,True);validate_bundle(j,row,segments,bundle,True)
                    assert bundle['checks']['missing_LOCAL']==1 and bundle['checks']['probe_calls']==2 and bundle['checks']['source_LM']==12 and bundle['checks']['source_vision']==8
                    t=bundle['traces'][-1];assert t['probe'] is None and t['branch'] is None and t['new_visual']==t['native_visual'] and t['native_speech'] is None
                    for kind in ('history','frequency','coherence','question','speech'):
                        bad=copy.deepcopy(bundle)
                        if kind=='history':bad['source_blocks'][-1]['direct_ancestors']=[]
                        elif kind=='frequency':bad['source_acquisition']['records'][0]['frequency'][0]+=1
                        elif kind=='coherence':bad['traces'][1]['probe']['retrieval']['coarse_vector'][0]+=1
                        elif kind=='question':bad['traces'][0]['branch']['input']['question_rows'][0]+=1
                        else:bad['optimized']['extra']['windows'][0]['z_speech']+=1
                        try:validate_bundle(j,row,segments,bad,True)
                        except AssertionError:pass
                        else:raise AssertionError('accepted corrupt '+kind)
                    memory.close(release=True);memory=None;validate_bundle(j,row,segments,bundle,True)
                    (folder/'bundle.json').write_text(json.dumps(bundle)+'\n')
                    cases.append(dict(dtype=str(dtype),native_frames=count,actual36source_and_reader_chain=True,source_LM=12,source_vision=8,
                        probes=2,source_all3grains=True,corruptions_rejected=['history','frequency','coherence','question','speech'],
                        no_LOCAL_nativeV_noProbe_NoneS_exact=True,permanent_proof_replay_after_temporary_release=True))
                    print('MEASUREMENT_CASE_PASS',dtype,count,flush=True)
            finally:
                if memory is not None:memory.close(release=True)
                for hook in hooks:hook.remove()
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,pretrained=False,cases=cases,
        scope='actual36layer production read/strict proof code, explicit synthetic pixels/tokenizer/renderer; not real8B/inputprocessor/semantic/performance'),indent=2)+'\n');print('MEASUREMENT_CPU_PASS',flush=True)


if __name__=='__main__':main()
