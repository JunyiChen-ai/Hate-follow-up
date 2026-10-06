"""Production controls on actual 36-layer random Qwen CPU; no metrics."""
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
from retrieval import ROOT,SPEC
from source_frames import acquire
from measure import read_video
from validate import validate_bundle
from controls import selection_batch,validate_selection,equal_r0,FrozenSelection,validate_variant,time_mapping


def main():
    torch.set_num_threads(4);torch.manual_seed(0)
    out=ROOT/'runs/20261006_m1_rekv/controls_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    print(socket.gethostname(),flush=True);(out/'run.pid').write_text(str(os.getpid()))
    original=out/'native.png';Image.fromarray(np.full((8,8,3),100,np.uint8)).save(original)
    row=dict(dataset='HateMM',video_id='fixture',duration=12.,video_path=str(ROOT/'runs/20261006_m1_rekv/source_frame_cpu_checks/fixture.mkv'))
    cache=out/'source_cache';folder=cache/row['dataset']/row['video_id'];source=acquire(row,folder/'frames')
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
            model=Qwen3VLModel(cfg).eval().to(dtype);model.requires_grad_(False)
            j,frames,hooks=fixture(model,20,dtype,original)
            base=out/str(dtype).split('.')[-1]
            memory=None
            try:
                with patch('measure.frame_paths',return_value=frames),patch('binding.frame_paths',return_value=frames):
                    reference,memory=read_video(j,row,[],meta,base/'reference',True)
                    validate_bundle(j,row,[],meta,reference,True);memory.close(release=True);memory=None
                    callback=lambda j,c,x,m,r:selection_batch(j,c,x,m,r,row,base/'selection',True)
                    bundle,memory=read_video(j,row,[],meta,base/'selection',True,after_read=callback)
                    validate_bundle(j,row,[],meta,bundle,True);equal_r0(bundle,reference)
                    for arm in bundle['controls'].values():validate_selection(bundle,arm,True)
                    corrupt=copy.deepcopy(bundle['controls']['D0']);corrupt['traces'][0]['exposure'][0]['remote_tokens']+=1
                    try:validate_selection(bundle,corrupt,True)
                    except AssertionError:pass
                    else:raise AssertionError('wrong budget proof accepted')
                    memory.close(release=True);memory=None
                    bundle,memory=read_video(j,row,[],meta,base/'history',True,previous_frames=0)
                    bundle['intervention']=dict(arm='H0',source_previous_frames=0,native_global_context_kept=True)
                    validate_variant(j,row,[],meta,bundle,True,'history',reference)
                    corrupt=copy.deepcopy(bundle);corrupt['source_blocks'][1]['ancestors']=[0]
                    try:validate_variant(j,row,[],meta,corrupt,True,'history',reference)
                    except AssertionError:pass
                    else:raise AssertionError('history deletion proof accepted inherited ancestor')
                    memory.close(release=True);memory=None
                    mapping=time_mapping(reference)
                    factory=lambda w,m,local:FrozenSelection(reference['traces'][w['i']]['branch']['layers'])
                    transform=lambda fs:[{**f,'presented_time':t} for f,t in zip(fs,mapping['presented_times'])]
                    bundle,memory=read_video(j,row,[],meta,base/'time',True,source_transform=transform,selector_factory=factory)
                    actual=[b['actual_time'] for b in bundle['source_blocks']];reads=[]
                    for t in bundle['traces']:
                        a,b=t['bounds']
                        if t['branch']:
                            for layer in t['branch']['layers']:
                                ids=layer['selected_ids'];reads.append(dict(reads=len(ids),role_changes=sum((a<=actual[i]<b)!=(a<=mapping['presented_times'][i]<b) for i in ids)))
                    bundle['intervention']=dict(arm='T0',timestamp_donors=mapping['timestamp_donors'],presented_times=mapping['presented_times'],groups=mapping['groups'],
                        cross_window_frames=sum(int(a//8)!=int(b//8) for a,b in zip(actual,mapping['presented_times'])),layer_exposures=reads)
                    validate_variant(j,row,[],meta,bundle,True,'time',reference)
                    assert all('forward' not in layer.self_attn.__dict__ for layer in model.language_model.layers)
                    cases.append(dict(dtype=str(dtype),layers=36,query_heads=32,kv_heads=8,selection_arms=7,history_deletion=True,
                        frozen_timestamp_selection=True,all_native_exact=True,clones_exact=True,wrong_budget_and_ancestor_rejected=True,
                        fixture_timestamp_scope='fixture source encoder omits timestamp semantics; actual processor/token/GPU timestamp changes required separately'))
                    print('CONTROLS_CASE_PASS',dtype,flush=True)
            finally:
                if memory is not None:memory.close(release=True)
                for h in hooks:h.remove()
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,pretrained=False,cases=cases,
        scope='actual36-layer randomweights fixture tokenizer/native/selection/source orchestration; not actual8B effect/metrics'),indent=2)+'\n')
    print('CONTROLS_CPU_PASS',flush=True)


if __name__=='__main__':main()
