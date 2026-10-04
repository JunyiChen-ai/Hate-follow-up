#!/usr/bin/env python3
"""Independent exact-native and executed-evidence local branches."""
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
from handle_inputs import CACHE,DATASETS,selected_rows,validate
from handle_extract import validate_generations
from handle_interface import CACHE_VERSION,CONSTANTS
from program import canonical,branch_record,EVIDENCE_HEADER


def tick():torch.cuda.synchronize();return time.perf_counter()


def question(executed,i,n,a,b,body,kind):
    rec=branch_record(executed,kind)
    return EVIDENCE_HEADER+canonical(rec)+'\n'+yesno_question(i,n,a,b,body,kind),rec


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
            prefix_tokens=ctx['prefix_tokens'],standalone_seconds=seconds,windows=windows))


@torch.no_grad()
def read_video(j,row,segments,metadata,smoke):
    frames=frame_paths(row['dataset'],row['video_id'],20);wins=fixed_windows(float(row['duration']),8)
    first=j.forward_calls;torch.cuda.reset_peak_memory_stats();start=tick();cache,ctx=build(j,frames,segments)
    prefix_seconds=tick()-start;native={'visual':[],'speech':[]};new={'visual':[],'speech':[]}
    times=dict(native_visual=0.,native_speech=0.,new_visual=0.,new_speech=0.,diagnostic=0.)
    traces=[];checked=set();diagnostics=0
    for i,((a,b),w) in enumerate(zip(wins,metadata['windows'])):
        body=window_text(segments,a,b);trace=dict(i=i,start=a,end=b,native_body=body,branches={})
        for kind in ('visual','speech'):
            if kind=='speech' and not body.strip():
                native[kind].append(None);new[kind].append(None);trace['branches'][kind]=dict(available=False,reason='native_no_speech');continue
            start=tick();native[kind].append(margin(j,cache,ctx,yesno_question(i,len(wins),a,b,body,kind)))
            times['native_'+kind]+=tick()-start
            query,rec=question(w['execution'],i,len(wins),a,b,body,kind)
            ids,suffix=j.branch_ids(ctx['msgs'],query,ctx['history'],head_text=ctx['head'])
            start=tick();z=margin(j,cache,ctx,query);times['new_'+kind]+=tick()-start;new[kind].append(z)
            t=dict(available=True,record=rec,question=query,suffix_text=suffix,suffix_ids=ids,margin=z,
                cache_tokens_restored=cache.get_seq_length()==ctx['stance_cache_tokens'],rope_restored=torch.equal(j.model.model.rope_deltas,ctx['rope']))
            assert t['cache_tokens_restored'] and t['rope_restored']
            if smoke and kind not in checked:
                start=tick();clone=copy.deepcopy(cache)
                try:replay=margin(j,clone,ctx,query);assert replay==z and clone.get_seq_length()==ctx['stance_cache_tokens']
                finally:del clone
                times['diagnostic']+=tick()-start;diagnostics+=1;checked.add(kind);t['clone_exact']=True
            trace['branches'][kind]=t
        traces.append(trace)
    base=record(row,ctx,native['visual'],native['speech'],prefix_seconds+times['native_visual']+times['native_speech'],'m1_native')
    optimized=record(row,ctx,new['visual'],new['speech'],metadata['standalone_seconds']+prefix_seconds+times['new_visual']+times['new_speech'],'m1_program_handles')
    forwards=j.forward_calls-first;expected=base['calls']+optimized['calls']-3+diagnostics;assert forwards==expected
    checks=dict(dataset=row['dataset'],video_id=row['video_id'],GT_read=False,actual_forwards=forwards,
        diagnostic_forwards=diagnostics,diagnostic_seconds=times['diagnostic'],prefix_seconds=prefix_seconds,
        native_visual_seconds=times['native_visual'],native_speech_seconds=times['native_speech'],
        new_visual_seconds=times['new_visual'],new_speech_seconds=times['new_speech'],
        preprocessing_seconds=metadata['standalone_seconds'],input_actual_forwards=metadata['actual_forwards'],
        planner_calls=len(metadata['chunks']),module_calls=sum(len(w['execution']['calls']) for w in metadata['windows']),
        valid_programs=sum(w['execution']['status']=='valid' for w in metadata['windows']),
        nonempty_programs=sum(bool(w['execution']['emitted']) for w in metadata['windows']),
        perception_nonUNKNOWN_calls=sum(c['result']['kind']!='UNKNOWN' for w in metadata['windows'] for c in w['execution']['calls']),
        perception_UNKNOWN_calls=sum(c['result']['kind']=='UNKNOWN' for w in metadata['windows'] for c in w['execution']['calls']),
        standalone_seconds=dict(base=base['extra']['standalone_seconds'],optimized=optimized['extra']['standalone_seconds']),
        peak_GiB=max(torch.cuda.max_memory_allocated()/2**30,metadata['peak_GiB']))
    detail=dict(dataset=row['dataset'],video_id=row['video_id'],segments=[list(s) for s in segments],cache_version=CACHE_VERSION,
        traces=traces,native_conversation=dict(msgs=ctx['msgs'],history=ctx['history'],head=ctx['head']))
    del cache
    return base,optimized,checks,detail


def validate_records(row,base,new,check,detail,metadata,renderer):
    wins=fixed_windows(float(row['duration']),8)
    for r in (base,new):
        assert (r['dataset'],r['video_id'])==(row['dataset'],row['video_id'])
        assert r['duration']==float(row['duration']) and r['native_rate']==FPS and r['error'] is None
        assert r['extra']['stance']==('Yes' if r['extra']['z_video']>0 else 'No')
        assert len(r['extra']['windows'])==len(wins) and np.isfinite(r['extra']['z_video'])
        for i,(w,(a,b)) in enumerate(zip(r['extra']['windows'],wins)):
            assert w['i']==i and w['start']==a and w['end']==b and w['z']==max(w['z_visual'],w.get('z_speech',float('-inf')))
        idx=np.clip(((np.arange(math.ceil(r['duration']*FPS))+.5)/FPS//8).astype(int),0,len(wins)-1)
        assert np.array_equal(r['score_curve'],np.asarray([w['z'] for w in r['extra']['windows']])[idx]) and np.isfinite(r['score_curve']).all()
        assert r['calls']==3+len(wins)+sum('z_speech' in w for w in r['extra']['windows'])
    for key in ('z_video','stance','prefix_tokens','stance_cache_tokens','stance_cache_logical_start'):
        assert new['extra'][key]==base['extra'][key]
    assert detail['cache_version']==CACHE_VERSION and len(detail['traces'])==len(wins)
    c=detail['native_conversation'];diagnostics=0
    for i,(t,b,n,m,(a,e)) in enumerate(zip(detail['traces'],base['extra']['windows'],new['extra']['windows'],metadata['windows'],wins)):
        body=window_text(detail['segments'],a,e);assert t['i']==i and t['start']==a and t['end']==e and t['native_body']==body
        for kind in ('visual','speech'):
            branch=t['branches'][kind];available=kind=='visual' or bool(body.strip())
            assert branch['available']==available and ('z_'+kind in n)==available and ('z_'+kind in b)==available
            if available:
                query,rec=question(m['execution'],i,len(wins),a,e,body,kind)
                assert branch['record']==rec and branch['question']==query and branch['margin']==n['z_'+kind]
                ids,text=renderer.branch_ids(c['msgs'],query,c['history'],head_text=c['head'])
                assert branch['suffix_ids']==ids and branch['suffix_text']==text and branch['cache_tokens_restored'] and branch['rope_restored']
                diagnostics+=branch.get('clone_exact',False)
            else:assert branch['reason']=='native_no_speech'
    assert check['diagnostic_forwards']==diagnostics and check['actual_forwards']==base['calls']+new['calls']-3+diagnostics
    assert check['module_calls']==sum(len(w['execution']['calls']) for w in metadata['windows'])
    actual_calls=[c for w in metadata['windows'] for c in w['execution']['calls']]
    assert check['perception_nonUNKNOWN_calls']==sum(c['result']['kind']!='UNKNOWN' for c in actual_calls)
    assert check['perception_UNKNOWN_calls']==sum(c['result']['kind']=='UNKNOWN' for c in actual_calls)
    assert check['planner_calls']==len(metadata['chunks'])
    assert check['valid_programs']==sum(w['execution']['status']=='valid' for w in metadata['windows'])
    assert check['nonempty_programs']==sum(bool(w['execution']['emitted']) for w in metadata['windows'])
    assert check['preprocessing_seconds']==metadata['standalone_seconds']
    assert check['input_actual_forwards']==metadata['actual_forwards']
    assert check['standalone_seconds']['base']==base['extra']['standalone_seconds']
    assert check['standalone_seconds']['optimized']==new['extra']['standalone_seconds']
    assert base['extra']['standalone_seconds']==check['prefix_seconds']+check['native_visual_seconds']+check['native_speech_seconds']
    assert new['extra']['standalone_seconds']==metadata['standalone_seconds']+check['prefix_seconds']+check['new_visual_seconds']+check['new_speech_seconds']


def existing(path,expected):
    rows={}
    if path.exists():
        for line in path.open():
            r=json.loads(line);key=r['dataset'],r['video_id'];assert key in expected and key not in rows;rows[key]=r
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261005_m1_program'/('r1_handles_full_'+('smoke' if a.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,cache_version=CACHE_VERSION,constants=CONSTANTS,
        GT_in_reader=False,smoke=a.smoke,torch=torch.__version__,transformers=transformers.__version__,
        code='experiments/20261005_m1_program/handle_{interface,inputs,decoder,extract,measure}.py + original program.py + src/{stance_cache,mllm_judge,video_inputs}.py; sources2026-10-05',
        command='python -u '+' '.join(sys.argv))
    cp=out/'config.json'
    if cp.exists():
        old=json.loads(cp.read_text());assert {k:v for k,v in old.items() if k!='date'}=={k:v for k,v in cfg.items() if k!='date'}
    else:cp.write_text(json.dumps(cfg,indent=2)+'\n')
    rows=selected_rows(a.smoke);expected={(r['dataset'],r['video_id']):r for r in rows}
    done={name:existing(out/name/'predictions.jsonl',expected) for name in ('base','optimized')};checks=existing(out/'checks.jsonl',expected)
    assert done['base'].keys()==done['optimized'].keys()==checks.keys()
    asr={ds:load_asr(ds) for ds in DATASETS};torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=0
    hook=j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1))
    handles={}
    for name in ('base','optimized'):
        d=out/name;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**cfg,'measurement':name},indent=2)+'\n')
        handles[name]=(d/'predictions.jsonl').open('a',buffering=1)
    ch=(out/'checks.jsonl').open('a',buffering=1)
    for i,row in enumerate(rows,1):
        key=row['dataset'],row['video_id'];segments=asr[key[0]].get(key[1],[])
        metadata=json.loads((CACHE/key[0]/(key[1]+'.json')).read_text());validate(metadata,row,segments);validate_generations(j,metadata)
        if key in checks:
            detail=json.loads((out/'details'/key[0]/(key[1]+'.json')).read_text())
            assert detail['segments']==[list(s) for s in segments]
            validate_records(row,done['base'][key],done['optimized'][key],checks[key],detail,metadata,j)
            logging.info('%d/%d reuse %s/%s',i,len(rows),*key);continue
        base,new,check,detail=read_video(j,row,segments,metadata,a.smoke)
        validate_records(row,base,new,check,detail,metadata,j)
        dd=out/'details'/key[0];dd.mkdir(parents=True,exist_ok=True);(dd/(key[1]+'.json')).write_text(json.dumps(detail)+'\n')
        for name,r in (('base',base),('optimized',new)):handles[name].write(json.dumps(r)+'\n')
        ch.write(json.dumps(check)+'\n');logging.info('%d/%d %s/%s %.2fs',i,len(rows),*key,new['extra']['standalone_seconds'])
    for f in handles.values():f.close()
    ch.close();hook.remove();logging.info('DONE coverage=%d',len(rows))


if __name__=='__main__':main()
