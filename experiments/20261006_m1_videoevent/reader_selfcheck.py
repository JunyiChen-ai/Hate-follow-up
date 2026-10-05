"""Actual36-layer production reader, native/empty/current/maximum event packets."""
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
    out=ROOT/'runs/20261006_m1_videoevent/reader_cpu_checks';out.mkdir(parents=True,exist_ok=True);print('host',socket.gethostname(),flush=True);(out/'run.pid').write_text(str(os.getpid()))
    p=out/'reader.png';Image.fromarray(np.random.default_rng(0).integers(0,256,(8,8,3),dtype=np.uint8)).save(p)
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,head_dim=128,max_position_embeddings=1024,rope_scaling={'rope_type':'default','mrope_section':[24,20,20]}),vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1]),image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa';torch.set_num_threads(1);torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats=lambda:None;torch.cuda.max_memory_allocated=lambda:0;m.clock=lambda *a:time.perf_counter();checks=[]
    for dtype in (torch.float32,torch.bfloat16):
        for native_count in (18,20):
            model=Qwen3VLModel(cfg).eval().to(dtype);j,ff=fixture(model,native_count,dtype,p);m.frame_paths=lambda *a:ff;j.forward_calls=j.vision_calls=0
            hooks=[model.register_forward_pre_hook(lambda *a:setattr(j,'forward_calls',j.forward_calls+1)),model.visual.register_forward_pre_hook(lambda *a:setattr(j,'vision_calls',j.vision_calls+1))]
            row=dict(dataset='synthetic',video_id='reader',duration=40.);segments=[(i*8.,i*8+7.,'actual speech source '+str(i)) for i in range(5)];outputs=[]
            for mode in ('no_event','current_event','maximum_event','no_frames'):
                windows=[dict(i=i,start=i*8,end=(i+1)*8,body=segments[i][2],frames=[] if mode=='no_frames' else [dict(id='p'+str(k),time=i*8+2+k*3,path=str(p.relative_to(ROOT))) for k in range(2)]) for i in range(5)]
                packets=[];backgrounds={}
                for i in range(5):
                    chosen=[] if mode in ('no_event','no_frames') else [i] if mode=='current_event' else list(range(4));rep=chosen[0] if chosen else None
                    packets.append(dict(selected_windows=chosen,representative=rep,source_ids=[k for k in chosen if k!=i]))
                    if rep is not None:backgrounds[str(rep)]=dict(description='UNVERIFIED_BACKGROUND',frame=windows[rep]['frames'][0])
                meta=dict(windows=windows,packets=packets,records=['UNVERIFIED_CAPTION']*5,backgrounds=backgrounds,cost=dict(standalone_seconds=43.,actual_forwards=29,actual_vision_forwards=7,caption_calls=5,relevance_calls=5,background_calls=len(backgrounds)),peak_GiB=0.)
                unchanged=copy.deepcopy(meta);realimage=m.image_margin;realmargin=m.margin;restored=[]
                def wrap(fn):
                    def inner(j,cache,ctx,*a):
                        saved=copy.deepcopy(cache);value=fn(j,cache,ctx,*a)
                        assert all(torch.equal(x.keys,y.keys) and torch.equal(x.values,y.values) for x,y in zip(cache.layers,saved.layers));assert torch.equal(j.model.model.rope_deltas,ctx['rope']);restored.append(True);return value
                    return inner
                m.image_margin=wrap(realimage);m.margin=wrap(realmargin)
                try:b=m.read_video(j,row,segments,meta,True)
                finally:m.image_margin=realimage;m.margin=realmargin
                m.validate_bundle(row,b,segments,meta,j,True);assert meta==unchanged
                assert b['checks']['actual_forwards']==25 and b['checks']['actual_vision']==(1 if mode in ('no_event','no_frames') else 7)
                assert b['checks']['diagnostic_forwards']==2 and b['optimized']['calls']==b['base']['calls']+29
                for i,t in enumerate(b['traces']):
                    if mode in ('no_event','no_frames'):assert t['source_branch'] is None and t['new_visual']==t['native_visual']
                    if mode=='current_event':assert t['new_speech']==t['native_speech']
                    if t['source_branch']:
                        content,paths=m.memory(windows[i],packets[i],meta);text=' '.join(c.get('text','') for c in content);assert 'unverified_caption' in text and 'unverified_background' in text and len(paths)<=11
                    assert 'UNVERIFIED' not in (t['new_speech_question'] or '') and 'relevance' not in (t['new_speech_question'] or '')
                if mode=='maximum_event':assert len(b['traces'][4]['source_branch']['image_counts'])==11
                outputs.append(b);checks.append(dict(dtype=str(dtype),native_frames=native_count,mode=mode,layers=36,PASS=True,allKV_rope_restored=True,forwards=b['checks']['actual_forwards'],vision=b['checks']['actual_vision'],source_cost_retained=True))
            assert all(b['base']['score_curve']==outputs[0]['base']['score_curve'] and b['base']['extra']['z_video']==outputs[0]['base']['extra']['z_video'] for b in outputs)
            for h in hooks:h.remove()
            print('READER_CASES_PASS',dtype,native_count,flush=True)
    calls=[];analyze.subprocess.run=lambda cmd,**kw:calls.append(cmd);analyze.evaluate(out/'synthetic_main',out/'synthetic_decoded','optimized')
    assert calls[0][1:3]==['-m','src.eval.evaluate_four_datasets']
    for flag,value in [('--transform','nscore'),('--duration','bma'),('--bma-prior','length'),('--min-windows','2'),('--bma-grid','6'),('--arm','m2')]:assert calls[1][calls[1].index(flag)+1]==value
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,scope='randomweights/software only; no pretrained semantics',checks=checks,canonical_commands=calls),indent=2)+'\n');print('READER_CPU_PASS',flush=True)


if __name__=='__main__':main()
