#!/usr/bin/env python3
"""Fresh paired native/interval speech measurements, without annotation access."""
import argparse
import copy
import json
import logging
import math
import os
import socket
import time

import numpy as np
import torch
from rote import ROOT, SPEC, IntervalAttention, prefix_sources, question_sources, source_intervals, coefficients, temporal_pairs
from src.mllm_judge import Judge, MODEL, yesno_question
from src.stance_cache import build, margin
from src.video_inputs import load_manifest, load_asr, frame_paths, fixed_windows, window_text

DATASETS = ('HateMM', 'HateClipSeg')


def selected_rows(smoke=False):
    rows = load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl', DATASETS)
    if smoke:
        rows = [r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]] + [r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def prediction(row, ctx, visual, speech, seconds, name):
    wins=fixed_windows(float(row['duration']),8)
    windows=[]
    for i,((a,b),v,s) in enumerate(zip(wins,visual,speech)):
        w=dict(i=i,start=a,end=b,z_visual=v,z=max(v,s) if s is not None else v)
        if s is not None:w['z_speech']=s
        windows.append(w)
    idx=np.clip(((np.arange(math.ceil(float(row['duration'])*4))+.5)/4//8).astype(int),0,len(wins)-1)
    return dict(schema_version=1,method=name,dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),native_rate=4,
        score_curve=np.asarray([w['z'] for w in windows])[idx].tolist(),intervals=[],error=None,seed=0,
        calls=3+len(wins)+sum(s is not None for s in speech),code_path='experiments/20261005_m1_rote/measure.py',
        extra=dict(z_video=ctx['global_margin'],stance=ctx['stance'],prefix_tokens=ctx['prefix_tokens'],
            stance_cache_tokens=ctx['stance_cache_tokens'],stance_cache_logical_start=ctx['stance_cache_logical_start'],
            windows=windows,standalone_seconds=seconds))


@torch.no_grad()
def read_video(j, row, segments, smoke):
    before=j.language_calls
    vision_before=j.vision_calls
    torch.cuda.reset_peak_memory_stats()
    start=tick()
    controller=IntervalAttention(j)
    try:
        with controller.native_prefix(segments):cache,ctx=build(j,frame_paths(row['dataset'],row['video_id'],20),segments)
        times=dict(shared_prefix=tick()-start,native_visual=0.,native_speech=0.,new_speech=0.,diagnostic=0.)
        native_v=[];native_s=[];new_s=[];traces=[];diagnostic_calls=0
        wins=fixed_windows(float(row['duration']),8)
        for i,(a,b) in enumerate(wins):
            body=window_text(segments,a,b)
            q=yesno_question(i,len(wins),a,b,body,'visual')
            start=tick();v=margin(j,cache,ctx,q);times['native_visual']+=tick()-start
            native_v.append(v)
            trace=dict(i=i,start=a,end=b,body=body,visual=dict(question=q,margin=v))
            if not body.strip():
                native_s.append(None);new_s.append(None);traces.append(trace);continue
            q=yesno_question(i,len(wins),a,b,body,'speech')
            ids,suffix=j.branch_ids(ctx['msgs'],q,ctx['history'],head_text=ctx['head'])
            mapping=question_sources(j,suffix,ids,segments,a,b,body)
            assert mapping
            start=tick();old=margin(j,cache,ctx,q);times['native_speech']+=tick()-start
            n=cache.get_seq_length();start=tick()
            with controller.speech(n,mapping,segments,a,b):z=margin(j,cache,ctx,q)
            times['new_speech']+=tick()-start
            assert cache.get_seq_length()==n and torch.equal(j.model.model.rope_deltas,ctx['rope'])
            trace['speech']=dict(question=q,suffix_ids=ids,suffix_text=suffix,body_sources=mapping,
                source_intervals=source_intervals(mapping,segments),native_margin=old,new_margin=z,
                layer_calls=controller.layer_calls,key_count=len(controller.key_indices),
                normalization_min=float(controller.norm.min()),normalization_max=float(controller.norm.max()),
                cache_restored=True,rope_restored=True)
            if smoke and diagnostic_calls==0:
                start=tick();clone=copy.deepcopy(cache)
                repeat_native=margin(j,clone,ctx,q)
                with controller.speech(n,mapping,segments,a,b):repeat_new=margin(j,clone,ctx,q)
                assert repeat_native==old and repeat_new==z
                assert clone.get_seq_length()==n
                assert all(torch.equal(x.keys,y.keys) and torch.equal(x.values,y.values) for x,y in zip(cache.layers,clone.layers))
                del clone;times['diagnostic']+=tick()-start;diagnostic_calls+=2
                trace['speech']['repeat_native_exact']=trace['speech']['repeat_new_exact']=True
            native_s.append(old);new_s.append(z);traces.append(trace)
        base_seconds=times['shared_prefix']+times['native_visual']+times['native_speech']
        new_seconds=times['shared_prefix']+times['native_visual']+times['new_speech']
        base=prediction(row,ctx,native_v,native_s,base_seconds,'m1_native')
        new=prediction(row,ctx,native_v,new_s,new_seconds,'m1_interval_rote')
        actual=j.language_calls-before
        assert actual==base['calls']+sum(s is not None for s in new_s)+diagnostic_calls
        assert j.vision_calls-vision_before==1
        result=dict(base=base,optimized=new,traces=traces,segments=[list(s) for s in segments],
            prefix_sources=controller.prefix_mapping,prefix_source_intervals=controller.prefix_intervals,
            temporal_pairs=controller.pairs.tolist(),frequencies=controller.frequencies.tolist(),
            native_ctx={k:v for k,v in ctx.items() if k not in ('rope','positions','files')},
            checks=dict(GT_read=False,host=socket.gethostname(),times=times,actual_language_forwards=actual,
                actual_vision_forwards=1,diagnostic_forwards=diagnostic_calls,
                peak_GiB=torch.cuda.max_memory_allocated()/2**30,
                source_key_buffer_bytes=sum(k.numel()*k.element_size() for k in controller.prefix_keys.values())))
        del cache
        return result
    finally:controller.close()


def validate_bundle(row,b,segments,j,smoke):
    assert b['checks']['GT_read'] is False and b['segments']==[list(s) for s in segments]
    ctx=b['native_ctx'];base=b['base'];new=b['optimized'];wins=fixed_windows(float(row['duration']),8)
    msgs,files=j.prefix_messages(frame_paths(row['dataset'],row['video_id'],20),segments)
    text,enc=j.encode_prefix(msgs,files)
    assert ctx['msgs']==msgs and prefix_sources(j,text,enc,segments)==b['prefix_sources']
    assert source_intervals(b['prefix_sources'],segments)==b['prefix_source_intervals']
    from transformers import AutoConfig
    pairs,freq=temporal_pairs(AutoConfig.from_pretrained(MODEL,local_files_only=True).text_config)
    assert pairs.tolist()==b['temporal_pairs'] and freq.tolist()==b['frequencies']
    assert base['extra']['z_video']==new['extra']['z_video'] and base['extra']['stance']==new['extra']['stance']
    assert base['extra']['z_video']==ctx['global_margin']
    assert ctx['stance']==('Yes' if ctx['global_margin']>0 else 'No')
    assert len(b['traces'])==len(wins)==len(base['extra']['windows'])==len(new['extra']['windows'])
    for i,((a,z),t,old,now) in enumerate(zip(wins,b['traces'],base['extra']['windows'],new['extra']['windows'])):
        assert t['body']==window_text(segments,a,z) and (t['i'],t['start'],t['end'])==(i,a,z)
        assert old['z_visual']==now['z_visual']==t['visual']['margin']
        assert t['visual']['question']==yesno_question(i,len(wins),a,z,t['body'],'visual')
        if t['body'].strip():
            s=t['speech'];q=yesno_question(i,len(wins),a,z,t['body'],'speech')
            ids,suffix=j.branch_ids(msgs,q,ctx['history'],head_text=ctx['head'])
            mapping=question_sources(j,suffix,ids,segments,a,z,t['body'])
            assert (q,ids,suffix,mapping)==(s['question'],s['suffix_ids'],s['suffix_text'],s['body_sources'])
            assert source_intervals(mapping,segments)==s['source_intervals']
            _,_,norm=coefficients(np.asarray(b['prefix_source_intervals']+s['source_intervals']).reshape(-1,2),freq)
            assert math.isclose(float(norm.min()),s['normalization_min'],abs_tol=1e-12)
            assert math.isclose(float(norm.max()),s['normalization_max'],abs_tol=1e-12)
            assert s['layer_calls']==36 and s['key_count']==len(mapping)+len(b['prefix_sources'])
            assert s['cache_restored'] and s['rope_restored']
            assert old['z_speech']==s['native_margin'] and now['z_speech']==s['new_margin']
        else:assert 'z_speech' not in old and 'z_speech' not in now and 'speech' not in t
        for w in (old,now):
            assert (w['i'],w['start'],w['end'])==(i,a,z)
            assert w['z']==max(w['z_visual'],w['z_speech']) if 'z_speech' in w else w['z']==w['z_visual']
    times=b['checks']['times']
    assert all(math.isfinite(v) and v>=0 for v in times.values())
    assert abs(base['extra']['standalone_seconds']-sum(times[k] for k in ('shared_prefix','native_visual','native_speech')))<1e-6
    assert abs(new['extra']['standalone_seconds']-sum(times[k] for k in ('shared_prefix','native_visual','new_speech')))<1e-6
    repeats=sum(bool(t.get('speech',{}).get('repeat_new_exact')) for t in b['traces'])
    assert repeats==(1 if smoke else 0) and b['checks']['diagnostic_forwards']==2*repeats
    calls=3+len(wins)+sum(bool(window_text(segments,a,z).strip()) for a,z in wins)
    assert base['calls']==new['calls']==calls
    assert b['checks']['actual_language_forwards']==calls+sum('speech' in t for t in b['traces'])+2*repeats
    assert b['checks']['actual_vision_forwards']==1
    for r in (base,new):
        assert (r['dataset'],r['video_id'],r['duration'],r['native_rate'])==(row['dataset'],row['video_id'],float(row['duration']),4)
        assert r['error'] is None and np.isfinite(r['score_curve']).all()
        assert len(r['score_curve'])==math.ceil(float(row['duration'])*4)
        idx=np.clip(((np.arange(len(r['score_curve']))+.5)/4//8).astype(int),0,len(wins)-1)
        assert np.array_equal(r['score_curve'],np.asarray([w['z'] for w in r['extra']['windows']])[idx])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261005_m1_rote'/('r1_full_smoke' if a.smoke else 'r1_full_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,spec=SPEC,GT_read=False,smoke=a.smoke,
        torch=torch.__version__,transformers=transformers.__version__,code='experiments/20261005_m1_rote/{rote,measure}.py; sources2026-10-05',command='python -u '+' '.join(sys.argv))
    config=out/'config.json'
    if config.exists():assert {k:v for k,v in json.loads(config.read_text()).items() if k!='date'}=={k:v for k,v in cfg.items() if k!='date'}
    else:config.write_text(json.dumps(cfg,indent=2)+'\n')
    torch.manual_seed(0);torch.set_num_threads(4)
    j=Judge(MODEL);j.language_calls=j.vision_calls=0
    hooks=[j.model.model.language_model.register_forward_pre_hook(lambda *_:setattr(j,'language_calls',j.language_calls+1)),
           j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    rows=selected_rows(a.smoke);asr={ds:load_asr(ds) for ds in DATASETS};results={n:[] for n in ('base','optimized')}
    for number,row in enumerate(rows,1):
        p=out/'records'/row['dataset']/(row['video_id']+'.json');p.parent.mkdir(parents=True,exist_ok=True)
        segments=asr[row['dataset']].get(row['video_id'],[])
        if p.exists():b=json.loads(p.read_text())
        else:
            b=read_video(j,row,segments,a.smoke);validate_bundle(row,b,segments,j,a.smoke)
            temporary=p.with_suffix('.partial');temporary.write_text(json.dumps(b)+'\n');temporary.replace(p)
        validate_bundle(row,b,segments,j,a.smoke)
        for name in results:results[name].append(b[name])
        logging.info('%d/%d %s/%s %.2fs',number,len(rows),row['dataset'],row['video_id'],b['optimized']['extra']['standalone_seconds'])
    for name,rr in results.items():
        d=out/name;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**cfg,'measurement':name},indent=2)+'\n')
        (d/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
    for h in hooks:h.remove()
    logging.info('DONE coverage=%d',len(rows))


if __name__=='__main__':
    import sys
    main()
