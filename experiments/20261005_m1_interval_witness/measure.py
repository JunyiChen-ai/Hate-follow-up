#!/usr/bin/env python3
"""Pair unchanged native readings with interval tree global/own stance/local readings."""
import argparse
import json
import logging
import math
import socket
import sys
import time
import numpy as np
import torch
from inputs import ROOT,CACHE,DATASETS,selected_rows,validate
from interface import VERSION,CONSTANTS,SPEC
from reader import read,validate_trace,build_tree,binding
from src.mllm_judge import Judge,MODEL,VIDEO_QUESTION,yesno_question
from src.stance_cache import build,margin
from src.source_generation import clock
from src.video_inputs import frame_paths,load_asr


def native_input(j,row,segments,stance):
    msgs,paths=j.prefix_messages(frame_paths(row['dataset'],row['video_id'],20),segments)
    text,enc=j.encode_prefix(msgs,paths);qid,qtext=j.branch_ids(msgs,VIDEO_QUESTION)
    aid,atext=j.answer_ids(msgs,VIDEO_QUESTION,stance)
    return dict(msgs=msgs,files=paths,history=[dict(role='user',content=[dict(type='text',text=VIDEO_QUESTION)]),j.turn('assistant',stance)],
        head=text+qtext+atext,native_stance_ids=enc['input_ids'][0].tolist()+qid+aid,
        stance_cache_tokens=len(enc['input_ids'][0])+len(qid)+len(aid),prefix_tokens=len(enc['input_ids'][0]))


def prediction(row,ctx,visual,speech,seconds,method):
    windows=[]
    for w,v,s in zip(ctx['windows'],visual,speech):
        value=dict(i=w['i'],start=w['start'],end=w['end'],z_visual=v,z=max(v,s) if s is not None else v)
        if s is not None:value['z_speech']=s
        windows.append(value)
    idx=np.clip(((np.arange(math.ceil(float(row['duration'])*4))+.5)/4//8).astype(int),0,len(windows)-1)
    return dict(schema_version=1,method=method,dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),
        native_rate=4.,score_curve=np.asarray([w['z'] for w in windows])[idx].tolist(),intervals=[],error=None,seed=0,
        calls=3+len(windows)+sum(s is not None for s in speech),
        code_path='experiments/20261005_m1_interval_witness/measure.py',extra=dict(z_video=ctx['global_margin'],stance=ctx['stance'],
            prefix_tokens=ctx['prefix_tokens'],stance_cache_tokens=ctx['stance_cache_tokens'],
            stance_cache_logical_start=ctx['stance_cache_logical_start'],windows=windows,standalone_seconds=seconds))


@torch.no_grad()
def read_video(j,row,segments,m,smoke):
    first=j.forward_calls;first_vision=j.vision_calls;torch.cuda.reset_peak_memory_stats();start=clock(j)
    cache,ctx=build(j,frame_paths(row['dataset'],row['video_id'],20),segments);prefix_seconds=clock(j)-start
    start=clock(j);conversation=native_input(j,row,segments,ctx['stance']);input_binding_seconds=clock(j)-start
    assert conversation['head']==ctx['head'] and conversation['stance_cache_tokens']==ctx['stance_cache_tokens']
    ctx['native_stance_ids']=conversation['native_stance_ids'];ctx['windows']=m['windows']
    native={kind:[] for kind in ('visual','speech')};new={kind:[] for kind in native}
    native_times={kind:[] for kind in native}
    times=dict(native_visual=0.,native_speech=0.,new_visual=0.,new_speech=0.,diagnostic=0.)
    for w in m['windows']:
        for kind in native:
            if kind=='speech' and not w['body'].strip():native[kind].append(None);native_times[kind].append(0.);continue
            start=clock(j);z=margin(j,cache,ctx,yesno_question(w['i'],len(m['windows']),w['start'],w['end'],w['body'],kind))
            seconds=clock(j)-start;native[kind].append(z);native_times[kind].append(seconds);times['native_'+kind]+=seconds
    # Full multimodal branches need no native KV cache; preserve literal native
    # context/own stance while releasing this memory before fresh branch prefill.
    del cache
    start=clock(j);newctx=build_tree(j,frame_paths(row['dataset'],row['video_id'],20),segments,m);new_prefix_seconds=clock(j)-start
    newctx['windows']=m['windows'];traces=[];checked=set();diagnostics=0
    for w in m['windows']:
        t=dict(window=w['i'],branches={},native_seconds={k:native_times[k][w['i']] for k in native})
        for kind in native:
            if kind=='speech' and not w['body'].strip():
                new[kind].append(None);t['branches'][kind]=dict(available=False,reason='native_no_speech');continue
            z,trace=read(j,newctx,m,w['i'],kind);new[kind].append(z);times['new_'+kind]+=trace['seconds'];trace['available']=True
            if smoke and kind not in checked:
                start=clock(j);replay,other=read(j,newctx,m,w['i'],kind)
                assert replay==z and other['input_tokens']==trace['input_tokens'] and other['image_grid']==trace['image_grid']
                seconds=clock(j)-start;trace['repeat_margin']=replay;trace['repeat_exact']=True;trace['repeat_seconds']=seconds
                diagnostics+=1;checked.add(kind);times['diagnostic']+=seconds
            t['branches'][kind]=trace
        traces.append(t)
    base=prediction(row,ctx,native['visual'],native['speech'],prefix_seconds+times['native_visual']+times['native_speech'],'m1_native')
    optimized=prediction(row,newctx,new['visual'],new['speech'],m['standalone_seconds']+new_prefix_seconds+times['new_visual']+times['new_speech'],'m1_interval_witness')
    actual=j.forward_calls-first;assert actual==base['calls']+optimized['calls']+diagnostics
    checks=dict(actual_forwards=actual,actual_vision_forwards=j.vision_calls-first_vision,
        prefix_seconds=prefix_seconds,new_prefix_seconds=new_prefix_seconds,input_binding_seconds=input_binding_seconds,**times,diagnostic_forwards=diagnostics,
        preprocessing_seconds=m['standalone_seconds'],input_actual_forwards=m['actual_forwards'],
        peak_GiB=max(torch.cuda.max_memory_allocated()/2**30,m['peak_GiB']))
    conversation['files']=[str(p.relative_to(ROOT)) for p in conversation['files']]
    tree_input={k:v for k,v in newctx.items() if k not in ('windows','image_counts','stance_cache_logical_start','global_margin','stance')}
    tree_input['files']=[str(p.relative_to(ROOT)) for p in tree_input['files']]
    timings=dict(prefix_seconds=prefix_seconds,new_prefix_seconds=new_prefix_seconds,input_binding_seconds=input_binding_seconds)
    return dict(version=VERSION,GT_read=False,native_input=conversation,tree_input=tree_input,base=base,optimized=optimized,traces=traces,checks=checks,timings=timings)


def validate_bundle(row,b,m,segments,j,smoke):
    validate(m,row,segments,j);assert b['version']==VERSION and b['GT_read'] is False
    expected=native_input(j,row,segments,b['base']['extra']['stance']);expected['files']=[str(p.relative_to(ROOT)) for p in expected['files']]
    assert b['native_input']==expected;newexpected=binding(j,frame_paths(row['dataset'],row['video_id'],20),segments,m,b['optimized']['extra']['stance'])
    newexpected['files']=[str(p.relative_to(ROOT)) for p in newexpected['files']]
    assert b['tree_input']==newexpected;ctx={**newexpected,'files':[ROOT/p for p in newexpected['files']]}
    repeats=0
    for name in ('base','optimized'):
        r=b[name];input_expected=expected if name=='base' else newexpected;assert (r['dataset'],r['video_id'],r['duration'],r['native_rate'],r['error'])==(row['dataset'],row['video_id'],float(row['duration']),4.,None)
        assert r['extra']['stance']==('Yes' if r['extra']['z_video']>0 else 'No') and np.isfinite(r['extra']['z_video'])
        assert r['extra']['prefix_tokens']==input_expected['prefix_tokens'] and r['extra']['stance_cache_tokens']==input_expected['stance_cache_tokens']
        assert len(r['extra']['windows'])==len(m['windows']) and len(b['traces'])==len(m['windows'])
        for w,p in zip(m['windows'],r['extra']['windows']):
            assert (p['i'],p['start'],p['end'])==(w['i'],w['start'],w['end'])
            assert ('z_speech' in p)==bool(w['body'].strip()) and p['z']==max(p['z_visual'],p.get('z_speech',float('-inf')))
        index=np.clip(((np.arange(math.ceil(r['duration']*4))+.5)/4//8).astype(int),0,len(m['windows'])-1)
        assert np.array_equal(r['score_curve'],np.asarray([w['z'] for w in r['extra']['windows']])[index]) and np.isfinite(r['score_curve']).all()
        assert r['calls']==3+len(m['windows'])+sum(bool(w['body'].strip()) for w in m['windows'])
    for w,t,p in zip(m['windows'],b['traces'],b['optimized']['extra']['windows']):
        assert t['window']==w['i']
        for kind in ('visual','speech'):
            trace=t['branches'][kind];available=kind=='visual' or bool(w['body'].strip());assert trace['available']==available
            if available:
                validate_trace(j,ctx,m,w['i'],kind,trace);assert trace['margin']==p['z_'+kind]
                if trace.get('repeat_exact'):
                    assert trace['repeat_margin']==trace['margin'];repeats+=1
            else:assert trace['reason']=='native_no_speech'
    if smoke:assert repeats==1+int(any(w['body'].strip() for w in m['windows']))
    check=b['checks'];assert check['diagnostic_forwards']==repeats
    assert {k:check[k] for k in ('prefix_seconds','new_prefix_seconds','input_binding_seconds')}==b['timings']
    assert check['actual_forwards']==b['base']['calls']+b['optimized']['calls']+repeats
    assert check['actual_vision_forwards']==2+b['optimized']['calls']-3+repeats
    assert check['input_actual_forwards']==m['actual_forwards'] and check['preprocessing_seconds']==m['standalone_seconds']
    assert check['peak_GiB']>=m['peak_GiB'] and np.isfinite(check['peak_GiB'])
    for key in ('prefix_seconds','new_prefix_seconds','input_binding_seconds','native_visual','native_speech','new_visual','new_speech','diagnostic'):
        assert np.isfinite(check[key]) and check[key]>=0
    for kind in ('visual','speech'):
        assert abs(check['new_'+kind]-sum(t['branches'][kind]['seconds'] for t in b['traces'] if t['branches'][kind]['available']))<1e-6
        assert all(np.isfinite(t['native_seconds'][kind]) and t['native_seconds'][kind]>=0 for t in b['traces'])
        assert abs(check['native_'+kind]-sum(t['native_seconds'][kind] for t in b['traces']))<1e-6
    assert abs(check['diagnostic']-sum(t['branches'][kind].get('repeat_seconds',0.) for t in b['traces'] for kind in ('visual','speech')))<1e-6
    for name in ('base','optimized'):
        seconds=check['prefix_seconds' if name=='base' else 'new_prefix_seconds']+check[('native_' if name=='base' else 'new_')+'visual']+check[('native_' if name=='base' else 'new_')+'speech']
        if name=='optimized':seconds+=m['standalone_seconds']
        assert abs(seconds-b[name]['extra']['standalone_seconds'])<1e-6


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261005_m1_interval_witness'/('r1_full_'+('smoke' if a.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    import transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,version=VERSION,constants=CONSTANTS,
        GT_read=False,smoke=a.smoke,torch=torch.__version__,transformers=transformers.__version__,
        code='experiments/20261005_m1_interval_witness/{inputs,interface,reader,measure}.py + src/{stance_cache,mllm_judge}.py; sources2026-10-05',command='python -u '+' '.join(sys.argv))
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');rows=selected_rows(a.smoke)
    asr={ds:load_asr(ds) for ds in DATASETS};torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=j.vision_calls=0
    hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),
        j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    handles={}
    for name in ('base','optimized'):
        folder=out/name;folder.mkdir(exist_ok=True);(folder/'config.json').write_text(json.dumps({**cfg,'measurement':name},indent=2)+'\n')
        handles[name]=(folder/'predictions.jsonl').open('w',buffering=1)
    for number,row in enumerate(rows,1):
        segments=asr[row['dataset']].get(row['video_id'],[]);m=json.loads((CACHE/row['dataset']/row['video_id']/'metadata.json').read_text())
        validate(m,row,segments,j);b=read_video(j,row,segments,m,a.smoke);validate_bundle(row,b,m,segments,j,a.smoke)
        folder=out/'records'/row['dataset'];folder.mkdir(parents=True,exist_ok=True);(folder/(row['video_id']+'.json')).write_text(json.dumps(b)+'\n')
        for name in handles:handles[name].write(json.dumps(b[name])+'\n')
        logging.info('%d/%d %s/%s %.2fs',number,len(rows),row['dataset'],row['video_id'],b['optimized']['extra']['standalone_seconds'])
    for handle in handles.values():handle.close()
    for hook in hooks:hook.remove()
    logging.info('DONE coverage=%d',len(rows))


if __name__=='__main__':main()
