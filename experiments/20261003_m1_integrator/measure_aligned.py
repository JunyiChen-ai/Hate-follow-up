#!/usr/bin/env python3
"""R4: ASR-aligned future visual memory, original speech; no annotations."""
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
from integrator import PrefixIntegrator
from aligned import aligned_edges
from src.window_token_regions import token_regions


def fresh_prefix(j,enc):
    if hasattr(j.model.model,'rope_deltas'):j.model.model.rope_deltas=None
    return j.prefix_cache(enc)


def window_reads(j,cache,msgs,prefix,btext,stance,wins,texts,kinds=('visual','speech')):
    torch.cuda.synchronize();tick=time.perf_counter()
    answer,atext=j.answer_ids(msgs,VIDEO_QUESTION,stance);j.extend_cache(cache,answer)
    torch.cuda.synchronize();times={'answer':time.perf_counter()-tick}
    history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',stance)]
    head=prefix+btext+atext;B=cache.get_seq_length();values=[{} for _ in wins];branches=0
    for kind in kinds:
        torch.cuda.synchronize();tick=time.perf_counter()
        for i,((a,b),txt) in enumerate(zip(wins,texts)):
            if kind=='speech' and not txt.strip():continue
            ids,_=j.branch_ids(msgs,yesno_question(i,len(wins),a,b,txt,kind),history,head_text=head)
            values[i]['z_'+kind]=j.cached_margin(cache,ids,in_place=True);cache.crop(B);branches+=1
        torch.cuda.synchronize();times[kind]=time.perf_counter()-tick
    return values,branches,times


def read_video(j,engine,row,segments,smoke=False):
    ds,vid,dur=row['dataset'],row['video_id'],float(row['duration'])
    frames=frame_paths(ds,vid,20,'k20');assert frames,(ds,vid)
    wins=fixed_windows(dur,8);texts=[window_text(segments,a,b) for a,b in wins]
    initial_calls=j.forward_calls
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
    msgs,files=j.prefix_messages(frames,segments);prefix,enc=j.encode_prefix(msgs,files)
    cache=fresh_prefix(j,enc);P=cache.get_seq_length()
    bids,btext=j.branch_ids(msgs,VIDEO_QUESTION);zv=j.cached_margin(cache,bids,in_place=True)
    torch.cuda.synchronize();global_seconds=time.perf_counter()-tick
    stance='Yes' if zv>0 else 'No';tick=time.perf_counter()
    base,B,base_parts=window_reads(j,cache,msgs,prefix,btext,stance,wins,texts)
    torch.cuda.synchronize();base_seconds=time.perf_counter()-tick;del cache
    tick=time.perf_counter()
    visual=(enc['input_ids'][0]==j.image_token_id).numpy()
    assert len(visual)==P and int(visual.sum())==sum(j.img_tokens)
    assert torch.all(enc['attention_mask']==1) and len(j.img_tokens)==len(frames)
    regions=token_regions(j,prefix,enc,frames,segments,wins)
    local=np.asarray(regions['local']);speech_region=regions['speech']
    edges=aligned_edges(visual,speech_region,local)
    mapping={'visual_tokens':int(visual.sum()),'prefix_tokens':P,
        'added_edges':int(np.triu(edges,k=1).sum()),'language_layers':len(engine.layers),
        'future_keys':'aligned','ASR_tokens':int(speech_region.sum()),'read_windows':len(wins)}
    original={k:v.clone() for k,v in enc.items() if torch.is_tensor(v)}
    torch.cuda.synchronize();mapping_seconds=time.perf_counter()-tick
    tick=time.perf_counter()
    with engine.encoding(visual,native=True):cache=fresh_prefix(j,enc)
    cz=j.cached_margin(cache,bids,in_place=True)
    causal,CB,_=window_reads(j,cache,msgs,prefix,btext,stance,wins,texts,kinds=('visual',));assert CB==len(wins)
    for w,bw in zip(causal,base):
        if 'z_speech' in bw:w['z_speech']=bw['z_speech']
    torch.cuda.synchronize();causal_seconds=time.perf_counter()-tick;del cache
    tick=time.perf_counter()
    with engine.encoding_edges(edges):cache=fresh_prefix(j,enc)
    assert cache.get_seq_length()==P and engine.active is None
    fz=j.cached_margin(cache,bids,in_place=True)
    future,FB,_=window_reads(j,cache,msgs,prefix,btext,stance,wins,texts,kinds=('visual',));assert FB==len(wins)
    for w,bw in zip(future,base):
        if 'z_speech' in bw:w['z_speech']=bw['z_speech']
    torch.cuda.synchronize();future_seconds=time.perf_counter()-tick;del cache
    verify={};diagnostic_seconds=0.
    if smoke:
        tick=time.perf_counter()
        cache=fresh_prefix(j,enc)
        rz=j.cached_margin(cache,bids,in_place=True)
        restored,RB,_=window_reads(j,cache,msgs,prefix,btext,stance,wins,texts)
        delta=max(abs(w[m]-v[m]) for w,v in zip(restored,base) for m in w)
        assert rz==zv and delta==0. and RB==B,('native restoration parity',rz,zv,delta)
        del cache;torch.cuda.synchronize();diagnostic_seconds=time.perf_counter()-tick
        verify={'restored_native_global_exact':True,'restored_native_windows_exact':True,'extra_forwards':3+B}
    assert all(torch.equal(enc[k],v) for k,v in original.items())
    measured_calls=j.forward_calls-initial_calls
    assert measured_calls==9+B+2*len(wins)+verify.get('extra_forwards',0)
    L=int(math.ceil(dur*FPS));idx=np.clip(((np.arange(L)+.5)/FPS//8).astype(int),0,len(wins)-1)
    records={}
    for arm,ww,calls,seconds in (('base',base,3+B,base_seconds),
            ('causal',causal,6+B,causal_seconds+mapping_seconds+base_parts['answer']+base_parts['speech']),
            ('future',future,6+B,future_seconds+mapping_seconds+base_parts['answer']+base_parts['speech'])):
        z=np.array([max(w.values()) for w in ww]);assert np.isfinite(z).all()
        records[arm]={'schema_version':1,'method':'m1_integrator_r4_'+arm,'dataset':ds,'video_id':vid,
            'duration':dur,'native_rate':FPS,'score_curve':z[idx].tolist(),'intervals':[],'error':None,
            'seed':0,'code_path':str(Path(__file__).relative_to(ROOT)),'calls':calls,
            'extra':{'z_video':zv,'stance':stance,'prefix_tokens':P,'prefix_seconds':global_seconds,
                'prefix_seconds_note':'includes original whole-video question',
                'branch_seconds':seconds,'branch_seconds_note':'causal/future include native answer+speech, mapping, second prefix/global/forced answer+visual',
                'n_branches':B,'windows':[{'i':i,'start':a,'end':b,'z':float(z[i]),**ww[i]} for i,(a,b) in enumerate(wins)]}}
    check={'dataset':ds,'video_id':vid,'prefix_tokens':P,'mapping':mapping,'native_global':zv,
        'future_global_diagnostic_only':fz,'causal_global_diagnostic_only':cz,
        'causal_window_max_abs_diff':max(abs(w[m]-v[m]) for w,v in zip(causal,base) for m in w),
        'global_seconds':global_seconds,'base_seconds':base_seconds,'mapping_seconds':mapping_seconds,
        'future_seconds':future_seconds,'causal_seconds':causal_seconds,'actual_forwards':measured_calls,'original_inputs_unchanged':True,'native_branches':B,
        'diagnostic_seconds':diagnostic_seconds,'base_parts':base_parts,'verify':verify,'peak_GiB':torch.cuda.max_memory_allocated()/2**30}
    return records,check,{'input_ids':enc['input_ids'][0].numpy(),'visual':visual,'speech':speech_region,'local':local}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    a.run_name='r4_smoke' if a.smoke else 'r4_main'
    torch.manual_seed(0);out=ROOT/'runs/20261003_m1_integrator'/a.run_name;out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',['HateMM','HateClipSeg'])
    if a.smoke:
        selected=[r for ds in ('HateMM','HateClipSeg') for r in [v for v in rows if v['dataset']==ds][:2]]
        selected+=[r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
        assert len({(r['dataset'],r['video_id']) for r in selected})==5
        rows=selected
    asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};j=Judge(MODEL);engine=PrefixIntegrator(j,'text')
    j.forward_calls=0
    def count_forward(*_):j.forward_calls+=1
    counter=j.model.model.register_forward_pre_hook(count_forward)
    import transformers
    config={**vars(a),'date':time.strftime('%Y-%m-%d'),'host':socket.gethostname(),'seed':0,'model':MODEL,
        'torch':torch.__version__,'transformers':transformers.__version__,'GT_in_reader':False,
        'code':'experiments/20261003_m1_integrator/{measure_aligned,aligned,integrator}.py + src/{video_inputs,mllm_judge,window_token_regions}.py; sources2026-10-03',
        'video_question':VIDEO_QUESTION,'frames':20,'window_seconds':8,'fps':FPS,
        'future_keys':'aligned','prefix_mask':'j<=i OR visual(i) AND NOT visual(j) AND (NOT ASR(j) OR same_read_window(i,j)); all36 layers',
        'queries':'ordinary full causal access',
        'global_key':'native original','forced_answer':'native original','speech':'native original'}
    if (out/'config.json').exists():
        previous=json.load((out/'config.json').open())
        assert previous['future_keys']=='aligned' and previous['smoke']==a.smoke
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');handles={};done={}
    for arm in ('base','causal','future'):
        d=out/arm;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**config,'arm':arm},indent=2)+'\n')
        p=d/'predictions.jsonl';done[arm]={(r['dataset'],r['video_id']) for r in map(json.loads,p.open())} if p.exists() else set()
        handles[arm]=p.open('a')
    cp=out/'checks.jsonl'
    checked_done={(r['dataset'],r['video_id']) for r in map(json.loads,cp.open())} if cp.exists() else set()
    assert done['base']==done['causal']==done['future']==checked_done,'partial arm records need repair before resume'
    checked=cp.open('a');started=time.time()
    for k,row in enumerate(rows):
        key=row['dataset'],row['video_id']
        if key in done['base']:continue
        recs,check,tokens=read_video(j,engine,row,asr[key[0]].get(key[1],[]),smoke=a.smoke)
        token_dir=out/'tokens'/key[0];token_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(token_dir/(key[1]+'.npz'),**tokens)
        for arm,r in recs.items():handles[arm].write(json.dumps(r)+'\n');handles[arm].flush()
        checked.write(json.dumps(check)+'\n');checked.flush()
        logging.info('progress %d/%d %s %s elapsed=%.1f peak_GiB=%.2f',k+1,len(rows),*key,time.time()-started,check['peak_GiB'])
    for f in handles.values():f.close()
    checked.close();counter.remove();engine.close();logging.info('RUN_DONE videos=%d elapsed=%.1f',len(rows),time.time()-started)

if __name__=='__main__':main()
