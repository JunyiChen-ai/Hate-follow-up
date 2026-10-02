#!/usr/bin/env python3
"""Paired VAR transfer collection; no annotations enter scoring."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import socket
import sys
import time
import numpy as np
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.mllm_judge import Judge,MODEL,VIDEO_QUESTION,yesno_question
from src.video_inputs import FPS,frame_paths,load_asr,load_manifest,fixed_windows,window_text
from recycler import AttentionRecycler


def synced_time():
    torch.cuda.synchronize();return time.perf_counter()


def prefix(j,frames,segments):
    msgs,files=j.prefix_messages(frames,segments)
    text,enc=j.encode_prefix(msgs,files)
    if hasattr(j.model.model,'rope_deltas'):j.model.model.rope_deltas=None
    cache=j.prefix_cache(enc);P=cache.get_seq_length()
    ids,btext=j.branch_ids(msgs,VIDEO_QUESTION)
    global_z=j.cached_margin(cache,ids,in_place=True)
    stance='Yes' if global_z>0 else 'No'
    ids,atext=j.answer_ids(msgs,VIDEO_QUESTION,stance);j.extend_cache(cache,ids)
    history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',stance)]
    return cache,msgs,text+btext+atext,history,global_z,stance,enc,P


def queries(j,msgs,head,history,wins,texts,kind):
    return [j.branch_ids(msgs,yesno_question(i,len(wins),a,b,txt,kind),history,head_text=head)[0]
        for i,((a,b),txt) in enumerate(zip(wins,texts))]


def native_reads(j,cache,ids,available):
    n=cache.get_seq_length();values=[]
    for tokens,use in zip(ids,available):
        values.append(j.cached_margin(cache,tokens,in_place=True) if use else None)
        cache.crop(n)
    return values


def read_video(j,engine,row,segments,smoke=False):
    ds,vid,duration=row['dataset'],row['video_id'],float(row['duration'])
    frames=frame_paths(ds,vid,20,'k20');assert frames
    wins=fixed_windows(duration,8);texts=[window_text(segments,a,b) for a,b in wins]
    V=len(wins);available=[bool(t.strip()) for t in texts];B=V+sum(available)
    initial=j.forward_calls;torch.cuda.reset_peak_memory_stats();tick=synced_time()
    engine.begin_capture()
    cache,msgs,head,history,zv,stance,enc,P=prefix(j,frames,segments)
    n=cache.get_seq_length();engine.end_capture(n)
    visual=(enc['input_ids'][0]==j.image_token_id).numpy()
    assert visual.sum()==sum(j.img_tokens) and len(j.img_tokens)==len(frames)
    assert torch.all(enc['attention_mask']==1)
    prefix_seconds=synced_time()-tick
    vis_ids=queries(j,msgs,head,history,wins,texts,'visual')
    speech_ids=queries(j,msgs,head,history,wins,texts,'speech')
    tick=synced_time();bv=native_reads(j,cache,vis_ids,[True]*V);native_visual_seconds=synced_time()-tick
    tick=synced_time();speech=native_reads(j,cache,speech_ids,available);speech_seconds=synced_time()-tick
    eager=[];recycle=[];diag=[];tick=synced_time()
    for ids in vis_ids:
        z,check=engine.margin(cache,ids,visual,0.)
        eager.append(z);diag.append({'eager':check})
    eager_seconds=synced_time()-tick;tick=synced_time()
    for i,ids in enumerate(vis_ids):
        z,check=engine.margin(cache,ids,visual,.6)
        recycle.append(z);diag[i]['recycle']=check
    recycle_seconds=synced_time()-tick
    assert cache.get_seq_length()==n and engine.active is None and not engine.capture
    saved={'channels':torch.stack([engine.channels[i] for i in engine.layer_indices]).cpu().numpy(),
        'cached_sinks':torch.stack([engine.cached[i] for i in engine.layer_indices]).cpu().numpy(),
        'input_ids':enc['input_ids'][0].numpy(),'visual':visual}
    del cache
    verify={};diagnostic_seconds=0.
    if smoke:
        tick=synced_time()
        rc,rm,rh,rhi,rz,rs,re,rP=prefix(j,frames,segments)
        rv=native_reads(j,rc,queries(j,rm,rh,rhi,wins,texts,'visual'),[True]*V)
        rsp=native_reads(j,rc,queries(j,rm,rh,rhi,wins,texts,'speech'),available)
        assert rz==zv and rs==stance and rv==bv and rsp==speech
        assert all(torch.equal(enc[k],re[k]) for k in enc if torch.is_tensor(enc[k]))
        assert rP==P;del rc;diagnostic_seconds=synced_time()-tick
        verify={'native_global_exact':True,'native_windows_exact':True,'inputs_exact':True,'extra_forwards':3+B}
    actual=j.forward_calls-initial;assert actual==3+B+2*V+verify.get('extra_forwards',0)
    length=int(math.ceil(duration*FPS));index=np.clip(((np.arange(length)+.5)/FPS//8).astype(int),0,V-1)
    records={}
    for arm,vs,seconds in (
            ('base',bv,prefix_seconds+native_visual_seconds+speech_seconds),
            ('eager',eager,prefix_seconds+eager_seconds+speech_seconds),
            ('recycle',recycle,prefix_seconds+recycle_seconds+speech_seconds)):
        windows=[]
        for i,((a,b),v,s) in enumerate(zip(wins,vs,speech)):
            w={'i':i,'start':a,'end':b,'z_visual':v,'z':max(v,s) if s is not None else v}
            if s is not None:w['z_speech']=s
            windows.append(w)
        scores=np.asarray([w['z'] for w in windows]);assert np.isfinite(scores).all()
        records[arm]={'schema_version':1,'method':'m1_recycler_'+arm,'dataset':ds,'video_id':vid,
            'duration':duration,'native_rate':FPS,'score_curve':scores[index].tolist(),'intervals':[],'error':None,
            'seed':0,'code_path':str(Path(__file__).relative_to(ROOT)),'calls':3+B,
            'extra':{'z_video':zv,'stance':stance,'prefix_tokens':P,'standalone_seconds':seconds,
                'n_branches':B,'windows':windows}}
    check={'dataset':ds,'video_id':vid,'native_global':zv,'answer':stance,'prefix_tokens':P,
        'cached_tokens':n,'visual_tokens':int(visual.sum()),'visual_queries':V,'native_branches':B,
        'actual_forwards':actual,'verify':verify,'diagnostic_seconds':diagnostic_seconds,
        'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'attention':diag,
        'seconds':{'native_prefix_with_capture':prefix_seconds,'native_visual':native_visual_seconds,
            'speech':speech_seconds,'eager':eager_seconds,'recycle':recycle_seconds}}
    return records,check,saved


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261003_m1_recycler'/('r1_smoke' if a.smoke else 'r1_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    torch.manual_seed(0)
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',['HateMM','HateClipSeg'])
    if a.smoke:
        rows=[r for ds in ('HateMM','HateClipSeg') for r in [v for v in rows if v['dataset']==ds][:2]]+[
            r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
        assert len(rows)==5
    asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};j=Judge(MODEL);engine=AttentionRecycler(j)
    j.forward_calls=0
    def count(*_):j.forward_calls+=1
    counter=j.model.model.register_forward_pre_hook(count)
    import transformers
    config={'date':time.strftime('%Y-%m-%d'),'host':socket.gethostname(),'smoke':a.smoke,'seed':0,
        'model':MODEL,'torch':torch.__version__,'transformers':transformers.__version__,'GT_in_reader':False,
        'code':'experiments/20261003_m1_recycler/{measure,recycler}.py + src/{video_inputs,mllm_judge}.py; sources2026-10-03',
        'sink_threshold':20,'redistributed_fraction':.6,'visual_mass_threshold':.2,'non_sink_fraction':.5,
        'channels':'top2 abs first prefix token residual at each block input, tie smaller index',
        'layers':engine.layer_indices,'intervention':'all visual-question rows; native prefix and global QA',
        'frames':20,'window_seconds':8,'fps':FPS,'video_question':VIDEO_QUESTION,
        'global_and_answer':'original native','speech':'original native',
        'timing_note':'base prefix includes sink-mask capture overhead shared by paired arms'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');handles={};done={}
    for arm in ('base','eager','recycle'):
        d=out/arm;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**config,'arm':arm},indent=2)+'\n')
        p=d/'predictions.jsonl';done[arm]={(r['dataset'],r['video_id']) for r in map(json.loads,p.open())} if p.exists() else set()
        handles[arm]=p.open('a')
    cp=out/'checks.jsonl';seen={(r['dataset'],r['video_id']) for r in map(json.loads,cp.open())} if cp.exists() else set()
    assert done['base']==done['eager']==done['recycle']==seen,'incomplete paired record'
    checked=cp.open('a');started=time.time()
    for k,row in enumerate(rows):
        key=row['dataset'],row['video_id']
        if key in seen:continue
        recs,check,saved=read_video(j,engine,row,asr[key[0]].get(key[1],[]),a.smoke)
        dest=out/'masks'/key[0];dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/(key[1]+'.npz'),**saved)
        for arm,r in recs.items():handles[arm].write(json.dumps(r)+'\n');handles[arm].flush()
        checked.write(json.dumps(check)+'\n');checked.flush()
        logging.info('progress %d/%d %s %s elapsed=%.1f peak_GiB=%.2f',k+1,len(rows),*key,time.time()-started,check['peak_GiB'])
    for f in handles.values():f.close()
    checked.close();counter.remove();engine.close();logging.info('RUN_DONE videos=%d elapsed=%.1f',len(rows),time.time()-started)


if __name__=='__main__':main()
