"""Production read_video and strict validator on actual decoded synthetic sources."""
import json
import os
import socket
from pathlib import Path
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


def main():
    torch.set_num_threads(4);torch.manual_seed(0)
    out=ROOT/'runs/20261006_m1_rekv/reader_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    print(socket.gethostname(),flush=True);(out/'run.pid').write_text(str(os.getpid()))
    (out/'config.json').write_text(json.dumps(dict(host=socket.gethostname(),GT_read=False,pretrained=False,
        scope='productionreader + source binding validator; fixture tokenizer; owned actualdecodedvideo',
        source='experiments/20261006_m1_rekv/reader_selfcheck.py;2026-10-06'),indent=2)+'\n')
    original=out/'native.png';Image.fromarray(np.full((8,8,3),100,dtype=np.uint8)).save(original)
    row=dict(dataset='HateMM',video_id='fixture',duration=12.,
        video_path=str(ROOT/'runs/20261006_m1_rekv/source_frame_cpu_checks/fixture.mkv'))
    cache=out/'source_cache';folder=cache/row['dataset']/row['video_id']
    source=acquire(row,folder/'frames')
    meta=dict(version=SPEC['version'],spec=SPEC,GT_read=False,dataset=row['dataset'],video_id=row['video_id'],source=source)
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=512,hidden_size=64,intermediate_size=128,num_hidden_layers=36,
        num_attention_heads=32,num_key_value_heads=8,head_dim=128,max_position_embeddings=2048,
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
            memory=None
            try:
                with patch('measure.frame_paths',return_value=frames),patch('binding.frame_paths',return_value=frames):
                    bundle,memory=read_video(j,row,[],meta,out/str(dtype).split('.')[-1],True)
                    validate_bundle(j,row,[],meta,bundle,True)
                assert bundle['checks']['source_frames']==5 and bundle['checks']['remote_layer_reads']>0
                assert sum(trace.get('clone_exact',False) for trace in bundle['traces'])==1
                assert bundle['base']['extra']['z_video']==bundle['optimized']['extra']['z_video']
                for trace in bundle['traces']:
                    assert trace['native_speech'] is None
                # A changed owned time or wrong selection must be rejected, even
                # when all saved model margins remain untouched.
                import copy
                corrupt=copy.deepcopy(bundle);corrupt['source_blocks'][0]['actual_time']+=.1
                try:validate_bundle(j,row,[],meta,corrupt,True)
                except AssertionError:pass
                else:raise AssertionError('wrong actual source ownership accepted')
                corrupt=copy.deepcopy(bundle);corrupt['traces'][0]['branch']['layers'][0]['remote_ids']=[]
                try:validate_bundle(j,row,[],meta,corrupt,True)
                except AssertionError:pass
                else:raise AssertionError('wrong retrieved-source proof accepted')
                cases.append(dict(dtype=str(dtype),production_read_video=True,strict_current_source_and_proof=True,
                    source_frames=5,model_layers=36,nativeG_exact=True,independent_clone_exact=True,source_corruption_rejected=True))
                print(dtype,'READER_PASS',flush=True)
            finally:
                if memory is not None:memory.close(release=True)
                for hook in hooks:hook.remove()
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,scope='software/fixture-tokenizer/randomweights; no actual8B/metrics',cases=cases),indent=2)+'\n')
    print('READER_CPU_PASS',flush=True)


if __name__=='__main__':main()
