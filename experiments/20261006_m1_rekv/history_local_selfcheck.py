"""Actual 36-layer H0 replay and joint deletion CPU checks; not pretrained."""
import copy
import json
import os
import socket
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from model_selfcheck import fixture
from inputs import ROOT,SPEC
from source_frames import acquire
from measure import read_video
from controls import selection_batch,equal_r0
from history_local import validate


def main():
    torch.set_num_threads(4);torch.manual_seed(0)
    out=ROOT/'runs/20261006_m1_rekv/history_local_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    print(socket.gethostname(),flush=True);(out/'run.pid').write_text(str(os.getpid()))
    original=out/'native.png';Image.fromarray(np.full((8,8,3),100,np.uint8)).save(original)
    row=dict(dataset='HateMM',video_id='fixture',duration=12.,video_path=str(ROOT/'runs/20261006_m1_rekv/source_frame_cpu_checks/fixture.mkv'))
    cache=out/'source_cache';source=acquire(row,cache/row['dataset']/row['video_id']/'frames')
    meta=dict(version=SPEC['version'],spec=SPEC,GT_read=False,dataset=row['dataset'],video_id=row['video_id'],source=source)
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=512,hidden_size=64,intermediate_size=128,num_hidden_layers=36,num_attention_heads=32,
        num_key_value_heads=8,head_dim=128,max_position_embeddings=2048,
        rope_parameters={'rope_type':'default','rope_theta':5000000.,'mrope_section':[24,20,20],'mrope_interleaved':True}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,
        temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),
        image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa'
    cases=[]
    with torch.no_grad(),patch('inputs.CACHE',cache):
        for dtype in (torch.float32,torch.bfloat16):
            for count,segments in ((18,[]),(20,[(0.,8.,'fixture spoken statement')])):
                model=Qwen3VLModel(cfg).eval().to(dtype);model.requires_grad_(False);j,frames,hooks=fixture(model,count,dtype,original)
                base=out/(str(dtype).split('.')[-1]+str(count));memory=None
                try:
                    with patch('measure.frame_paths',return_value=frames),patch('binding.frame_paths',return_value=frames):
                        main,memory=read_video(j,row,segments,meta,base/'main',True);memory.close(release=True);memory=None
                        prior,memory=read_video(j,row,segments,meta,base/'prior',False,previous_frames=0);memory.close(release=True);memory=None
                        first=j.forward_calls;vision=j.vision_calls
                        callback=lambda j,c,x,m,r:selection_batch(j,c,x,m,r,row,base/'joint',True,arms=('L0',))
                        bundle,memory=read_video(j,row,segments,meta,base/'joint',True,previous_frames=0,after_read=callback)
                        bundle['intervention']=dict(arm='H0',source_previous_frames=0,native_global_context_kept=True)
                        bundle['batch_actual_forwards']=j.forward_calls-first;bundle['batch_actual_vision']=j.vision_calls-vision
                        bundle['batch_peak_GiB']=0.
                        equal_r0(bundle,prior)
                        def reference(root,row):return main if root.name=='r1_full_main' else prior
                        with patch('history_local.record',side_effect=reference):
                            validate(j,row,segments,meta,bundle,True)
                            for kind in ('ancestry','remote','cost'):
                                bad=copy.deepcopy(bundle)
                                if kind=='ancestry':bad['source_blocks'][1]['ancestors']=[0]
                                elif kind=='remote':bad['controls']['L0']['traces'][0]['branch']['layers'][0]['remote_ids']=[4]
                                else:bad['batch_actual_forwards']+=1
                                try:validate(j,row,segments,meta,bad,True)
                                except AssertionError:pass
                                else:raise AssertionError('accepted corrupt '+kind)
                        assert all('forward' not in layer.self_attn.__dict__ for layer in model.language_model.layers)
                        cases.append(dict(dtype=str(dtype),native_frames=count,nonempty_S=bool(segments),H0_raw_keys_queries_selection_margins_exact=True,
                            HL0_all_LOCAL_no_REMOTE_no_history=True,native_G_S_exact=True,clones=2,corruption_rejected=['ancestry','remote','cost']))
                        print('HL0_CPU_CASE_PASS',dtype,count,flush=True)
                finally:
                    if memory is not None:memory.close(release=True)
                    for hook in hooks:hook.remove()
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,pretrained=False,cases=cases,
        scope='actual36-layer random Qwen 32Q/8KV/3DeepStack orchestration fixture; pretrained semantics/real renderer not asserted'),indent=2)+'\n')
    print('HL0_CPU_PASS',flush=True)


if __name__=='__main__':main()
