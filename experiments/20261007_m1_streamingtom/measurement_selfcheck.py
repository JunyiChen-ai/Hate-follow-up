"""Complete production read/strict proof replay, including uncovered windows."""
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
from cpu_fixture import fixture
from measure import read_video
from validate import validate_bundle


def main():
    torch.set_num_threads(4);torch.manual_seed(0);print(socket.gethostname(),flush=True)
    out=ROOT/'runs/20261007_m1_streamingtom/measurement_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    native=out/'native.png';Image.fromarray(np.full((8,8,3),100,np.uint8)).save(native);rng=np.random.default_rng(0);frames=[]
    for i in range(7):
        if i%2==0:image=rng.integers(0,256,(32,32,3),dtype=np.uint8)
        path=out/f'source_{i}.png';Image.fromarray(image).save(path);frames.append(dict(index=16*i,time=2.*i,path=str(path.relative_to(ROOT))))
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=512,hidden_size=64,intermediate_size=128,num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,head_dim=128,
        max_position_embeddings=4096,rope_parameters={'rope_type':'default','rope_theta':5000000.,'mrope_section':[24,20,20],'mrope_interleaved':True}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,
            out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa';cases=[]
    row=dict(dataset='HateMM',video_id='fixture',duration=24.);segments=[(0.,8.,'Fixture speech'),(8.,16.,'Later speech')]
    source=dict(decode_seconds=0.,fixture=True)
    with torch.no_grad():
        for dtype,count in ((torch.float32,18),(torch.bfloat16,20)):
            model=Qwen3VLModel(cfg).eval().to(dtype);model.requires_grad_(False);j,ff,hooks=fixture(model,count,dtype,native);memory=None
            folder=out/(str(dtype).split('.')[-1]+str(count))
            try:
                with (patch('measure.frames_for',return_value=(source,frames)),patch('validate.frames_for',return_value=(source,frames)),
                    patch('measure.frame_paths',return_value=ff),patch('src.native_input_binding.frame_paths',return_value=ff)):
                    bundle,memory=read_video(j,row,segments,folder,True);validate_bundle(j,row,segments,bundle,True)
                    assert bundle['checks']['missing_LOCAL']==1 and bundle['checks']['diagnostic_forwards']==2
                    last=bundle['traces'][-1];assert last['branch'] is None and last['new_visual']==last['native_visual'] and last['native_speech'] is None
                    for kind in ('question','source_KV_budget','native_speech','group'):
                        bad=copy.deepcopy(bundle)
                        if kind=='question':bad['traces'][0]['branch']['input']['question_rows'][0]+=1
                        elif kind=='source_KV_budget':bad['source_blocks'][0]['storage_bytes']+=1
                        elif kind=='native_speech':bad['optimized']['extra']['windows'][0]['z_speech']+=1
                        else:bad['source_acquisition']['records'][1]['plan']['groups'][0]['weights'][0]+=.1
                        try:validate_bundle(j,row,segments,bad,True)
                        except AssertionError:pass
                        else:raise AssertionError('accepted corrupt '+kind)
                    # Strict replay still succeeds after only its temporary
                    # working payload is released; permanent proofs remain.
                    memory.close(release=True);memory=None;validate_bundle(j,row,segments,bundle,True)
                    (folder/'bundle.json').write_text(json.dumps(bundle)+'\n')
                    cases.append(dict(dtype=str(dtype),native_frames=count,complete_production_reader_and_strict_replay=True,
                        full_projector_3DS_and_weighted_source_plan_recomputed=True,actual_question_rows_3axis_GQA_token_budgets=True,
                        native_S_or_None_and_no_LOCAL_fallback_exact=True,clones=2,corruptions_rejected=['question','source_KV_budget','native_speech','group'],
                        temporary_release_then_permanent_replay=True))
                    print('MEASUREMENT_CASE_PASS',dtype,count,flush=True)
            finally:
                if memory is not None:memory.close(release=True)
                for hook in hooks:hook.remove()
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,pretrained=False,cases=cases,
        scope='actual36layer production read_video + strict proof code with explicit fixture tokenizer/renderer/rawinput provider; real8B/native processor separate'),indent=2)+'\n')
    print('MEASUREMENT_CPU_PASS',flush=True)


if __name__=='__main__':main()
