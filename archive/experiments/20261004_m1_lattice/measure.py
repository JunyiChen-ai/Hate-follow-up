#!/usr/bin/env python3
"""Paired exact native and single suffix lattice speech measurements."""
import argparse
import copy
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
from src.mllm_judge import Judge,MODEL,yesno_question
from src.stance_cache import build,margin
from src.video_inputs import FPS,frame_paths,load_asr,fixed_windows,window_text
from extract import CACHE,DATASETS,selected_rows,validate
from lattice import CACHE_VERSION,CONSTANTS,confusion,SCAFFOLD,validate_graph_trace
from reader import structural,compile_branch


def tick():torch.cuda.synchronize();return time.perf_counter()


def record(row,ctx,visual,speech,seconds,method):
    windows=[];duration=float(row['duration']);wins=fixed_windows(duration,8)
    for i,((a,b),v,s) in enumerate(zip(wins,visual,speech)):
        w=dict(i=i,start=a,end=b,z_visual=v,z=max(v,s) if s is not None else v)
        if s is not None:w['z_speech']=s
        windows.append(w)
    idx=np.clip(((np.arange(math.ceil(duration*FPS))+.5)/FPS//8).astype(int),0,len(wins)-1)
    return dict(schema_version=1,method=method,dataset=row['dataset'],video_id=row['video_id'],duration=duration,
        native_rate=FPS,score_curve=np.asarray([w['z'] for w in windows])[idx].tolist(),intervals=[],error=None,seed=0,
        calls=3+len(wins)+sum(s is not None for s in speech),
        code_path=str(Path(__file__).relative_to(ROOT)),extra=dict(z_video=ctx['global_margin'],stance=ctx['stance'],
            stance_cache_tokens=ctx['stance_cache_tokens'],stance_cache_logical_start=ctx['stance_cache_logical_start'],
            prefix_tokens=ctx['prefix_tokens'],standalone_seconds=seconds,n_branches=len(wins)+sum(s is not None for s in speech),windows=windows))


@torch.no_grad()
def smoke_checks(j,cache,ctx,w,nwindows,z):
    # Compare against a cache cloned before the replay, without carrying local state.
    clone=copy.deepcopy(cache)
    try:replay,_=structural(j,clone,ctx,w,nwindows);assert replay==z
    finally:del clone
    # Probability-one degeneration uses exactly the manually compiled IDs.
    unit=copy.deepcopy(w);text=w['confusion']['onebest']
    unit['confusion']=confusion([dict(text=text,score=0.) for _ in range(5)])
    structured,_=structural(j,cache,ctx,unit,nwindows)
    ids,graph,h,t,trace=compile_branch(j,ctx,unit,nwindows)
    n=cache.get_seq_length();j.model.model.rope_deltas=ctx['rope'].clone()
    try:serial=j.cached_margin(cache,ids,in_place=True)
    finally:cache.crop(n);j.model.model.rope_deltas=ctx['rope'].clone()
    # Custom versus automatic SDPA masks can choose different BF16 kernels.
    assert abs(structured-serial)<=.01,(structured,serial)
    return dict(cloned_cache_exact=True,unit_graph_margin=structured,unit_serial_margin=serial,
        unit_difference=structured-serial,unit_tolerance=.01,diagnostic_forwards=3)


@torch.no_grad()
def read_video(j,row,segments,metadata,smoke):
    frames=frame_paths(row['dataset'],row['video_id'],20);wins=fixed_windows(float(row['duration']),8)
    first=j.forward_calls;torch.cuda.reset_peak_memory_stats();start=tick();cache,ctx=build(j,frames,segments)
    prefix_seconds=tick()-start;visual=[];base_speech=[];visual_seconds=reference_speech_seconds=0.
    for i,((a,b),w) in enumerate(zip(wins,metadata['windows'])):
        body=window_text(segments,a,b);assert body==w['native_body']
        start=tick();visual.append(margin(j,cache,ctx,yesno_question(i,len(wins),a,b,body,'visual')))
        visual_seconds+=tick()-start
        if body.strip():
            start=tick();base_speech.append(margin(j,cache,ctx,yesno_question(i,len(wins),a,b,body,'speech')))
            reference_speech_seconds+=tick()-start
        else:base_speech.append(None)
    new_speech=[];traces=[];new_speech_seconds=diagnostic_seconds=0.;diagnostic_forwards=0;checked=False
    for w in metadata['windows']:
        if w['available']:
            start=tick();z,trace=structural(j,cache,ctx,w,len(wins));new_speech_seconds+=tick()-start
            if smoke and not checked:
                start=tick();trace['smoke_checks']=smoke_checks(j,cache,ctx,w,len(wins),z)
                diagnostic_seconds+=tick()-start;diagnostic_forwards+=3;checked=True
            trace['available']=True
        else:z=None;trace=dict(available=False,reason=w['reason'])
        new_speech.append(z);traces.append(trace)
    base=record(row,ctx,visual,base_speech,prefix_seconds+visual_seconds+reference_speech_seconds,'m1_native')
    new=record(row,ctx,visual,new_speech,metadata['standalone_seconds']+prefix_seconds+visual_seconds+new_speech_seconds,'m1_lattice')
    real=j.forward_calls-first;expected=base['calls']+sum(w['available'] for w in metadata['windows'])+diagnostic_forwards
    assert real==expected,(real,expected)
    checks=dict(dataset=row['dataset'],video_id=row['video_id'],GT_read=False,actual_forwards=real,
        diagnostic_forwards=diagnostic_forwards,diagnostic_seconds=diagnostic_seconds,
        standalone_seconds=dict(base=base['extra']['standalone_seconds'],optimized=new['extra']['standalone_seconds']),
        prefix_seconds=prefix_seconds,visual_seconds=visual_seconds,reference_speech_seconds=reference_speech_seconds,
        new_speech_seconds=new_speech_seconds,preprocessing_seconds=metadata['standalone_seconds'],
        input_actual_forwards=metadata['actual_forwards'],peak_GiB=max(torch.cuda.max_memory_allocated()/2**30,metadata['peak_GiB']),
        graph_tokens=sum(t.get('graph_tokens',0) for t in traces),graph_slots=sum(t.get('slot_count',0) for t in traces),
        recognized_windows=sum(w['available'] for w in metadata['windows']))
    detail=dict(dataset=row['dataset'],video_id=row['video_id'],segments=[list(s) for s in segments],
        cache_version=CACHE_VERSION,traces=traces,missing_audio_windows=[w['i'] for w in metadata['windows'] if w.get('reason')=='missing_actual_audio'])
    del cache
    return base,new,checks,detail


def validate_records(row,base,new,check,detail,metadata,tokenizer):
    wins=fixed_windows(float(row['duration']),8)
    for r in (base,new):
        assert (r['dataset'],r['video_id'])==(row['dataset'],row['video_id'])
        assert r['duration']==float(row['duration']) and r['native_rate']==FPS and r['error'] is None
        assert r['extra']['stance']==('Yes' if r['extra']['z_video']>0 else 'No')
        assert len(r['extra']['windows'])==len(wins) and np.isfinite(r['extra']['z_video'])
        for i,(w,(a,b)) in enumerate(zip(r['extra']['windows'],wins)):
            assert w['i']==i and w['start']==a and w['end']==b
            assert w['z']==max(w['z_visual'],w.get('z_speech',float('-inf')))
        idx=np.clip(((np.arange(math.ceil(r['duration']*FPS))+.5)/FPS//8).astype(int),0,len(wins)-1)
        assert np.array_equal(r['score_curve'],np.asarray([w['z'] for w in r['extra']['windows']])[idx])
        assert np.isfinite(r['score_curve']).all()
        assert r['calls']==3+len(wins)+sum('z_speech' in w for w in r['extra']['windows'])
    assert new['extra']['z_video']==base['extra']['z_video'] and new['extra']['stance']==base['extra']['stance']
    for key in ('prefix_tokens','stance_cache_tokens','stance_cache_logical_start'):
        assert new['extra'][key]==base['extra'][key]
    assert detail['cache_version']==CACHE_VERSION and len(detail['traces'])==len(wins)
    count=0;diagnostic=0
    for b,w,m,t in zip(base['extra']['windows'],new['extra']['windows'],metadata['windows'],detail['traces']):
        assert w['z_visual']==b['z_visual'] and ('z_speech' in w)==m['available']==t['available']
        if m['available']:
            count+=1;assert t['margin']==w['z_speech'] and t['mask_forward'] and t['prefix_positions_unchanged']
            validate_graph_trace(t,m['confusion'],tokenizer,base['extra']['stance_cache_tokens'],base['extra']['stance_cache_logical_start'])
            assert t['scaffold']==SCAFFOLD.format(i=m['i']+1,n=len(wins),a=m['start'],b=m['end'])
            assert t['scaffold'] in t['head_text']
            diagnostic+=t.get('smoke_checks',{}).get('diagnostic_forwards',0)
        else:assert t['reason']==m['reason']
    assert check['recognized_windows']==count and check['diagnostic_forwards']==diagnostic
    assert check['actual_forwards']==base['calls']+count+diagnostic


def existing(path,expected):
    rows={}
    if path.exists():
        for line in path.open():
            r=json.loads(line);key=r['dataset'],r['video_id'];assert key in expected and key not in rows
            rows[key]=r
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261004_m1_lattice'/('r1_full_'+('smoke' if a.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    config=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,cache_version=CACHE_VERSION,
        constants=CONSTANTS,GT_in_reader=False,smoke=a.smoke,torch=torch.__version__,transformers=transformers.__version__,
        code='experiments/20261004_m1_lattice/{lattice,reader,extract,measure}.py + src/{stance_cache,mllm_judge,video_inputs}.py; sources2026-10-05',
        diagnostic_unit_tolerance=.01,command='python -u '+' '.join(sys.argv))
    cp=out/'config.json'
    if cp.exists():
        old=json.loads(cp.read_text());assert {k:v for k,v in old.items() if k!='date'}=={k:v for k,v in config.items() if k!='date'}
    else:cp.write_text(json.dumps(config,indent=2)+'\n')
    rows=selected_rows(a.smoke);expected={(r['dataset'],r['video_id']):r for r in rows}
    done={name:existing(out/name/'predictions.jsonl',expected) for name in ('base','optimized')}
    checks=existing(out/'checks.jsonl',expected);assert done['base'].keys()==done['optimized'].keys()==checks.keys()
    asr={ds:load_asr(ds) for ds in DATASETS}
    if checks:
        from transformers import AutoTokenizer
        resume_tokenizer=AutoTokenizer.from_pretrained(MODEL)
    for key in checks:
        segments=asr[key[0]].get(key[1],[]);metadata=json.loads((CACHE/key[0]/(key[1]+'.json')).read_text())
        validate(metadata,expected[key],segments)
        detail=json.loads((out/'details'/key[0]/(key[1]+'.json')).read_text())
        assert detail['segments']==[list(s) for s in segments]
        validate_records(expected[key],done['base'][key],done['optimized'][key],checks[key],detail,metadata,resume_tokenizer)
    torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=0
    hook=j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1))
    handles={}
    for name in ('base','optimized'):
        d=out/name;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**config,'measurement':name},indent=2)+'\n')
        handles[name]=(d/'predictions.jsonl').open('a',buffering=1)
    ch=(out/'checks.jsonl').open('a',buffering=1)
    for i,row in enumerate(rows,1):
        key=row['dataset'],row['video_id']
        if key in checks:logging.info('%d/%d reuse %s/%s',i,len(rows),*key);continue
        segments=asr[key[0]].get(key[1],[]);metadata=json.loads((CACHE/key[0]/(key[1]+'.json')).read_text())
        validate(metadata,row,segments)
        base,new,check,detail=read_video(j,row,segments,metadata,a.smoke)
        validate_records(row,base,new,check,detail,metadata,j.tok)
        detaildir=out/'details'/key[0];detaildir.mkdir(parents=True,exist_ok=True)
        (detaildir/(key[1]+'.json')).write_text(json.dumps(detail)+'\n')
        for name,r in [('base',base),('optimized',new)]:handles[name].write(json.dumps(r)+'\n')
        ch.write(json.dumps(check)+'\n');logging.info('%d/%d %s/%s %.2fs',i,len(rows),*key,new['extra']['standalone_seconds'])
    for f in handles.values():f.close()
    ch.close();hook.remove();logging.info('DONE coverage=%d',len(rows))


if __name__=='__main__':main()
