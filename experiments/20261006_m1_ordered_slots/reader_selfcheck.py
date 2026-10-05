"""Production reader with real36-layer CPU model, all ordered source cases."""
import copy
import json
import os
import socket
import time
import torch
import numpy as np
from PIL import Image
from inputs import ROOT,SPEC
from model_selfcheck import fixture,Qwen3VLConfig,Qwen3VLModel
import measure as m
import analyze


def main():
    out=ROOT/'runs/20261006_m1_ordered_slots/reader_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    print('host',socket.gethostname(),flush=True);(out/'run.pid').write_text(str(os.getpid()))
    (out/'config.json').write_text(json.dumps(dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),code='experiments/20261006_m1_ordered_slots/reader_selfcheck.py',GT_read=False,spec=SPEC),indent=2)+'\n')
    p=out/'reader.png';Image.fromarray(np.random.default_rng(0).integers(0,256,(8,8,3),dtype=np.uint8)).save(p)
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,head_dim=128,max_position_embeddings=1024,rope_scaling={'rope_type':'default','mrope_section':[24,20,20]}),vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1]),image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa'
    torch.set_num_threads(1);torch.manual_seed(0);torch.cuda.reset_peak_memory_stats=lambda:None;torch.cuda.max_memory_allocated=lambda:0;m.clock=lambda *a:time.perf_counter();checks=[]
    for dtype in (torch.float32,torch.bfloat16):
        for native_count in (18,20):
            model=Qwen3VLModel(cfg).eval().to(dtype);j,ff=fixture(model,native_count,dtype,p);m.frame_paths=lambda *a:ff;j.forward_calls=j.vision_calls=0
            hooks=[model.register_forward_pre_hook(lambda *a:setattr(j,'forward_calls',j.forward_calls+1)),model.visual.register_forward_pre_hook(lambda *a:setattr(j,'vision_calls',j.vision_calls+1))]
            row=dict(dataset='synthetic',video_id='reader',duration=24.);segments=[(i*8.,i*8.+7.,'actual speech source '+str(i)) for i in range(3)];outputs=[]
            for mode in ('no_remote','prequel','sequel','both','no_frames'):
                windows=[dict(i=i,start=i*8,end=(i+1)*8,body=segments[i][2],frames=[] if mode=='no_frames' else [dict(id='p'+str(k),time=i*8+2+k*3,path=str(p.relative_to(ROOT))) for k in range(2)]) for i in range(3)]
                packets=[]
                for i in range(3):
                    before=i-1 if i>0 and mode in ('prequel','both') else None;after=i+1 if i<2 and mode in ('sequel','both') else None
                    packets.append(dict(window=i,bindings={role:dict(source_id=sid) for role,sid in [('prequel',before),('current',i),('sequel',after)]},source_ids=[k for k in (before,after) if k is not None]))
                meta=dict(windows=windows,packets=packets,records=[dict(caption='CAPTION_MUST_NOT_ENTER_READER')]*3,slots=[dict(prequel='HYPOTHESIS_MUST_NOT_ENTER_READER')]*3,cost=dict(standalone_seconds=43.,actual_forwards=29,actual_vision_forwards=7,caption_calls=3,slot_calls=1,embedding_calls=9),peak_GiB=0.)
                unchanged=copy.deepcopy(meta);snapshots=[];realimage=m.image_margin;realmargin=m.margin
                def wrap(fn):
                    def inner(j,cache,ctx,*a):
                        saved=copy.deepcopy(cache);r=fn(j,cache,ctx,*a)
                        assert all(torch.equal(x.keys,y.keys) and torch.equal(x.values,y.values) for x,y in zip(cache.layers,saved.layers))
                        assert torch.equal(j.model.model.rope_deltas,ctx['rope']);snapshots.append(True);return r
                    return inner
                m.image_margin=wrap(realimage);m.margin=wrap(realmargin)
                try:b=m.read_video(j,row,segments,meta,True)
                finally:m.image_margin=realimage;m.margin=realmargin
                m.validate_bundle(row,b,segments,meta,j,True);assert meta==unchanged and len(snapshots)==14
                assert b['checks']['actual_forwards']==17 and b['checks']['actual_vision']==(1 if mode=='no_frames' else 5)
                assert b['checks']['diagnostic_forwards']==2 and b['optimized']['calls']==b['base']['calls']+29
                for i,t in enumerate(b['traces']):
                    body,paths=m.memory(windows[i],packets[i],windows);serialized=json.dumps(body)+str(t['new_speech_question'])
                    assert 'CAPTION_MUST_NOT_ENTER_READER' not in serialized and 'HYPOTHESIS_MUST_NOT_ENTER_READER' not in serialized
                    if packets[i]['source_ids']:
                        assert 'Actual remote speech context' in t['new_speech_question']
                        for k in packets[i]['source_ids']:assert segments[k][2] in t['new_speech_question']
                    else:assert t['new_speech']==t['native_speech'] and 'Actual remote speech context' not in t['new_speech_question']
                    if mode=='both' and i==1:assert len(t['source_branch']['image_counts'])==6
                outputs.append(b);checks.append(dict(dtype=str(dtype),native_frames=native_count,mode=mode,layers=36,production_reader_PASS=True,all_branch_KV_rope_exact=True,actual_forwards=17,actual_vision=b['checks']['actual_vision'],source_cost_preserved=True,hypothetical_text_excluded=True))
            assert all(x['base']['score_curve']==outputs[0]['base']['score_curve'] and x['base']['extra']['z_video']==outputs[0]['base']['extra']['z_video'] for x in outputs)
            for h in hooks:h.remove()
            print('READER_CASES_PASS',dtype,native_count,flush=True)
    calls=[];analyze.subprocess.run=lambda cmd,**kw:calls.append(cmd);analyze.evaluate(out/'synthetic_main',out/'synthetic_decoded','optimized')
    assert calls[0][1:3]==['-m','src.eval.evaluate_four_datasets']
    for flag,value in [('--transform','nscore'),('--duration','bma'),('--bma-prior','length'),('--min-windows','2'),('--bma-grid','6'),('--arm','m2')]:assert calls[1][calls[1].index(flag)+1]==value
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,scope='randomweights/science interfaces only, no pretrained performance',checks=checks,canonical_commands=calls),indent=2)+'\n');print('READER_CPU_PASS',flush=True)


if __name__=='__main__':main()
