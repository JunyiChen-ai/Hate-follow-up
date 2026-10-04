#!/usr/bin/env python3
"""Paired native and graph-source asymmetric speech readings."""
import argparse
import copy
import json
import logging
import math
import socket
import sys
import time
import numpy as np
import torch

from extract import ROOT,CACHE,DATASETS,selected_rows,validate
from graph import VERSION,CONSTANTS
from reader import structural,compile_branch
from src.mllm_judge import Judge,MODEL,VIDEO_QUESTION,yesno_question
from src.mllm_renderer import cpu_renderer
from src.stance_cache import build,margin
from src.video_inputs import frame_paths,load_asr


def tick():torch.cuda.synchronize();return time.perf_counter()


def prediction(row,ctx,visual,speech,seconds,method):
    ww=[]
    for m,v,s in zip(ctx['windows'],visual,speech):
        w=dict(i=m['i'],start=m['start'],end=m['end'],z_visual=v,z=max(v,s) if s is not None else v)
        if s is not None:w['z_speech']=s
        ww.append(w)
    idx=np.clip(((np.arange(math.ceil(float(row['duration'])*4))+.5)/4//8).astype(int),0,len(ww)-1)
    return dict(schema_version=1,method=method,dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),
        native_rate=4,score_curve=np.asarray([w['z'] for w in ww])[idx].tolist(),intervals=[],error=None,seed=0,
        calls=3+len(ww)+sum(s is not None for s in speech),code_path=str(__file__).replace(str(ROOT)+'/',''),
        extra=dict(z_video=ctx['global_margin'],stance=ctx['stance'],prefix_tokens=ctx['prefix_tokens'],
            stance_cache_tokens=ctx['stance_cache_tokens'],stance_cache_logical_start=ctx['stance_cache_logical_start'],
            standalone_seconds=seconds,windows=ww))


def bind_context(j,row,segments,stance):
    frames=frame_paths(row['dataset'],row['video_id'],20);msgs,files=j.prefix_messages(frames,segments)
    head,encoded=j.encode_prefix(msgs,files);qids,qtext=j.branch_ids(msgs,VIDEO_QUESTION)
    aids,atext=j.answer_ids(msgs,VIDEO_QUESTION,stance)
    return dict(msgs=msgs,files=files,head=head+qtext+atext,
        history=[dict(role='user',content=[dict(type='text',text=VIDEO_QUESTION)]),j.turn('assistant',stance)]),dict(
        head=head+qtext+atext,files=[str(p) for p in files],image_counts=list(j.img_tokens),
        ids=encoded['input_ids'][0].tolist()+qids+aids,prefix_tokens=encoded['input_ids'].shape[1])


def validate_bundle(row,bundle,metadata,segments,j,smoke):
    validate(metadata,row,segments,j)
    assert bundle['version']==VERSION and bundle['GT_read'] is False
    base,new=bundle['base'],bundle['optimized'];windows=metadata['windows']
    ctx,native=bind_context(j,row,segments,base['extra']['stance']);assert native==bundle['native_input']
    n=len(native['ids']);start=base['extra']['stance_cache_logical_start']
    assert n==base['extra']['stance_cache_tokens'] and native['prefix_tokens']==base['extra']['prefix_tokens']
    count=diagnostics=0
    for r in (base,new):
        assert (r['dataset'],r['video_id'])==(row['dataset'],row['video_id']) and r['error'] is None
        assert r['duration']==float(row['duration']) and r['native_rate']==4
        assert r['extra']['stance']==('Yes' if r['extra']['z_video']>0 else 'No')
        for k in ('z_video','stance','prefix_tokens','stance_cache_tokens','stance_cache_logical_start'):
            assert r['extra'][k]==base['extra'][k]
        ww=r['extra']['windows'];assert len(ww)==len(windows)
        for w,m in zip(ww,windows):
            assert (w['i'],w['start'],w['end'])==(m['i'],m['start'],m['end'])
            assert ('z_speech' in w)==bool(m['body'].strip())
            assert w['z_visual']==base['extra']['windows'][m['i']]['z_visual']
            assert w['z']==max(w['z_visual'],w.get('z_speech',float('-inf')))
        idx=np.clip(((np.arange(math.ceil(r['duration']*4))+.5)/4//8).astype(int),0,len(ww)-1)
        assert np.array_equal(r['score_curve'],np.asarray([w['z'] for w in ww])[idx])
        assert np.isfinite(r['score_curve']).all() and np.isfinite(r['extra']['z_video'])
        assert r['calls']==3+len(ww)+sum('z_speech' in w for w in ww)
    assert len(bundle['traces'])==len(windows)
    for w,m,p,t in zip(new['extra']['windows'],windows,metadata['packets'],bundle['traces']):
        if not m['body'].strip():assert t==dict(available=False);continue
        count+=1;ids,expected=compile_branch(j,ctx,windows,p)
        assert all(t[k]==v for k,v in expected.items()) and t['ids']==ids
        assert t['margin']==w['z_speech'] and t['available'] and t['mask_forward']
        assert t['prefix_tokens']==n and t['prefix_logical_start']==start
        assert t['positions']==list(range(start,start+len(ids))) and t['attention_dtype']=='torch.bfloat16'
        if 'clone_margin' in t:assert t['clone_margin']==t['margin'];diagnostics+=1
    c=bundle['checks'];assert diagnostics==int(smoke and count>0)
    assert c['diagnostic_forwards']==diagnostics+2*int(smoke and count>0)
    assert c['actual_forwards']==base['calls']+count+c['diagnostic_forwards']
    assert c['preprocessing_seconds']==metadata['standalone_seconds'] and c['input_actual_forwards']==metadata['actual_forwards']
    assert c['peak_GiB']>=metadata['peak_GiB']
    assert base['extra']['standalone_seconds']==c['prefix_seconds']+c['visual_seconds']+c['reference_speech_seconds']
    assert new['extra']['standalone_seconds']==metadata['standalone_seconds']+c['prefix_seconds']+c['visual_seconds']+c['new_speech_seconds']
    assert all(np.isfinite(v) and v>=0 for k,v in c.items() if k.endswith('seconds') or k=='peak_GiB')
    if smoke and count:
        assert c['contextless_checks']['difference']==c['contextless_checks']['graph_margin']-c['contextless_checks']['serial_margin']
        assert abs(c['contextless_checks']['difference'])<=.01


@torch.no_grad()
def read_video(j,row,segments,metadata,smoke):
    first=j.forward_calls;torch.cuda.reset_peak_memory_stats();start=tick()
    cache,ctx=build(j,frame_paths(row['dataset'],row['video_id'],20),segments);prefix_seconds=tick()-start
    ctx['windows']=metadata['windows'];native_ids=[];image_index=0
    for token in j.tok.encode(ctx['head'],add_special_tokens=False):
        if token==j.image_token_id:native_ids.extend([token]*ctx['image_counts'][image_index]);image_index+=1
        else:native_ids.append(token)
    assert image_index==len(ctx['image_counts']) and len(native_ids)==cache.get_seq_length()
    native_input=dict(head=ctx['head'],files=[str(p) for p in ctx['files']],image_counts=ctx['image_counts'],
        ids=native_ids,prefix_tokens=ctx['prefix_tokens'])
    visual=[];speech=[];new_speech=[];traces=[]
    visual_seconds=reference_seconds=new_seconds=diagnostic_seconds=0.;diagnostics=0;contextless={};checked=False
    for m in metadata['windows']:
        start=tick();visual.append(margin(j,cache,ctx,yesno_question(m['i'],len(ctx['windows']),m['start'],m['end'],m['body'],'visual')))
        visual_seconds+=tick()-start
        if m['body'].strip():
            start=tick();speech.append(margin(j,cache,ctx,yesno_question(m['i'],len(ctx['windows']),m['start'],m['end'],m['body'],'speech')))
            reference_seconds+=tick()-start
        else:speech.append(None)
    for m,p in zip(metadata['windows'],metadata['packets']):
        if not m['body'].strip():new_speech.append(None);traces.append(dict(available=False));continue
        start=tick();z,t=structural(j,cache,ctx,metadata['windows'],p);new_seconds+=tick()-start;t['available']=True
        if smoke and not checked:
            start=tick();clone=copy.deepcopy(cache)
            try:replay,_=structural(j,clone,ctx,metadata['windows'],p);assert replay==z
            finally:del clone
            t['clone_margin']=replay
            empty=copy.deepcopy(p);empty['selected']=[];empty['relations']=[]
            gz,_=structural(j,cache,ctx,metadata['windows'],empty);ids,_=compile_branch(j,ctx,metadata['windows'],empty)
            n=cache.get_seq_length();j.model.model.rope_deltas=ctx['rope'].clone()
            try:sz=j.cached_margin(cache,ids,in_place=True)
            finally:cache.crop(n);j.model.model.rope_deltas=ctx['rope'].clone()
            assert abs(gz-sz)<=.01
            contextless=dict(graph_margin=gz,serial_margin=sz,difference=gz-sz,tolerance=.01)
            diagnostic_seconds+=tick()-start;diagnostics+=3;checked=True
        new_speech.append(z);traces.append(t)
    base=prediction(row,ctx,visual,speech,prefix_seconds+visual_seconds+reference_seconds,'m1_native')
    new=prediction(row,ctx,visual,new_speech,metadata['standalone_seconds']+prefix_seconds+visual_seconds+new_seconds,'m1_quote_graph')
    actual=j.forward_calls-first;assert actual==base['calls']+sum(s is not None for s in new_speech)+diagnostics
    checks=dict(prefix_seconds=prefix_seconds,visual_seconds=visual_seconds,reference_speech_seconds=reference_seconds,
        new_speech_seconds=new_seconds,diagnostic_seconds=diagnostic_seconds,diagnostic_forwards=diagnostics,
        preprocessing_seconds=metadata['standalone_seconds'],input_actual_forwards=metadata['actual_forwards'],actual_forwards=actual,
        peak_GiB=max(torch.cuda.max_memory_allocated()/2**30,metadata['peak_GiB']),contextless_checks=contextless)
    del cache
    return dict(version=VERSION,GT_read=False,native_input=native_input,base=base,optimized=new,traces=traces,checks=checks)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    out=ROOT/'runs/20261005_m1_quote_graph'/('r1_full_'+('smoke' if args.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    import transformers
    config=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,version=VERSION,constants=CONSTANTS,
        GT_read=False,smoke=args.smoke,torch=torch.__version__,transformers=transformers.__version__,
        code='experiments/20261005_m1_quote_graph/{graph,reader,measure}.py + src/stance_cache.py; sources2026-10-05',
        command='python -u '+' '.join(sys.argv))
    cp=out/'config.json'
    if cp.exists():
        old=json.loads(cp.read_text());assert {k:v for k,v in old.items() if k!='date'}=={k:v for k,v in config.items() if k!='date'}
    else:cp.write_text(json.dumps(config,indent=2)+'\n')
    rows=selected_rows(args.smoke);asr={ds:load_asr(ds) for ds in DATASETS};renderer=cpu_renderer();done={}
    for row in rows:
        path=out/'records'/row['dataset']/(row['video_id']+'.json')
        if path.exists():
            m=json.loads((CACHE/row['dataset']/(row['video_id']+'.json')).read_text());b=json.loads(path.read_text())
            validate_bundle(row,b,m,asr[row['dataset']].get(row['video_id'],[]),renderer,args.smoke);done[row['dataset'],row['video_id']]=b
    torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=0
    hook=j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1))
    for i,row in enumerate(rows,1):
        key=row['dataset'],row['video_id']
        if key in done:logging.info('%d/%d reuse %s/%s',i,len(rows),*key);continue
        segments=asr[key[0]].get(key[1],[]);m=json.loads((CACHE/key[0]/(key[1]+'.json')).read_text());validate(m,row,segments,renderer)
        b=read_video(j,row,segments,m,args.smoke);validate_bundle(row,b,m,segments,renderer,args.smoke)
        path=out/'records'/key[0]/(key[1]+'.json');path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.partial');temp.write_text(json.dumps(b)+'\n');temp.replace(path);done[key]=b
        logging.info('%d/%d %s/%s %.2fs',i,len(rows),*key,b['optimized']['extra']['standalone_seconds'])
    hook.remove()
    for name in ('base','optimized'):
        d=out/name;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        with (d/'predictions.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(done[row['dataset'],row['video_id']][name])+'\n')
    logging.info('DONE coverage=%d',len(rows))


if __name__=='__main__':main()
