"""Fresh native readings and actual source-image/crop cached branches; no GT."""
import argparse
import copy
import json
import logging
import math
import os
from pathlib import Path
import socket
import time
import numpy as np
import torch
from inputs import ROOT,SPEC,CACHE,DATASETS,selected_rows
from extract import validate as validate_acquisition
from src.mllm_judge import Judge,MODEL,yesno_question
from src.stance_cache import build,margin
from src.source_image_branch import margin as image_margin,encode_branch
from src.source_generation import clock
from src.video_inputs import frame_paths,load_asr,fixed_windows,window_text
REVISION=int(os.environ.get('READER_REVISION','1'));assert REVISION in (1,2,3)
if REVISION>=2:assert SPEC.get('interface')=='B'


def uses_source(window,packet):
    return bool(window['frames']) and (REVISION==1 or bool(packet['source_ids']))


def remote_speech(windows,packet):
    return [dict(role=role,id=i,bounds=[windows[i]['start'],windows[i]['end']],ASR=windows[i]['body'])
        for role in ('prequel','sequel') for i in [packet['bindings'][role]['source_id']] if i is not None and windows[i]['body'].strip()]


def memory(window,packet,windows):
    content=[dict(type='text',text='Actual current LOCAL observations and actual temporally ordered context. Remote material remains at its own timestamps; it is not evidence of an event occurring in the current window. The role labels indicate source selection, not proven causality.\n')];paths=[]
    for role in SPEC['slot_order']:
        index=packet['bindings'][role]['source_id']
        if index is None:
            content.append(dict(type='text',text=json.dumps(dict(role=role,source='NONE'))));continue
        source=windows[index]
        details=dict(role=role,id=index,bounds=[source['start'],source['end']])
        if REVISION in (1,3):details['literal_ASR']=source['body']
        content.append(dict(type='text',text=json.dumps(details,ensure_ascii=False)))
        for frame in source['frames']:
            content.extend([dict(type='text',text=f"Actual frame {frame['id']} at PTS {frame['time']:.9f}s."),dict(type='image')]);paths.append(ROOT/frame['path'])
    assert len(paths)<=2*(1+SPEC['max_remote_windows'])
    return content,paths


def prediction(row,ctx,visual,speech,seconds,method,source_forwards=0):
    wins=fixed_windows(float(row['duration']),SPEC['window_seconds']);windows=[]
    for i,((a,b),v,s) in enumerate(zip(wins,visual,speech)):
        w=dict(i=i,start=a,end=b,z_visual=v,z=max(v,s) if s is not None else v)
        if s is not None:w['z_speech']=s
        windows.append(w)
    idx=np.clip(((np.arange(math.ceil(float(row['duration'])*4))+.5)/4//SPEC['window_seconds']).astype(int),0,len(wins)-1)
    return dict(schema_version=1,method=method,dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),native_rate=4,
        score_curve=np.asarray([w['z'] for w in windows])[idx].tolist(),intervals=[],error=None,seed=0,calls=3+len(wins)+sum(s is not None for s in speech)+source_forwards,
        code_path='experiments/20261006_m1_ordered_slots/measure.py',extra=dict(z_video=ctx['global_margin'],stance=ctx['stance'],prefix_tokens=ctx['prefix_tokens'],
            stance_cache_tokens=ctx['stance_cache_tokens'],stance_cache_logical_start=ctx['stance_cache_logical_start'],windows=windows,standalone_seconds=seconds))


@torch.no_grad()
def read_video(j,row,segments,m,smoke):
    start=clock(j);before=j.forward_calls;before_vision=j.vision_calls;torch.cuda.reset_peak_memory_stats()
    cache,ctx=build(j,frame_paths(row['dataset'],row['video_id'],SPEC['native_frames']),segments)
    times=dict(prefix=clock(j)-start,native_visual=0.,native_speech=0.,new_visual=0.,new_speech=0.,diagnostic=0.)
    native_visual=[];native_speech=[];new_visual=[];new_speech=[];traces=[];diagnostic_calls=diagnostic_vision=0;speech_clone=False
    for i,((a,b),w) in enumerate(zip(fixed_windows(float(row['duration']),SPEC['window_seconds']),m['windows'])):
        body=window_text(segments,a,b);q=yesno_question(i,len(m['windows']),a,b,body,'visual')
        start=clock(j);v=margin(j,cache,ctx,q);times['native_visual']+=clock(j)-start;native_visual.append(v)
        if body.strip():
            sq=yesno_question(i,len(m['windows']),a,b,body,'speech');start=clock(j);s=margin(j,cache,ctx,sq);times['native_speech']+=clock(j)-start
        else:s=None
        native_speech.append(s)
        start=clock(j)
        if uses_source(w,m['packets'][i]):
            content,paths=memory(w,m['packets'][i],m['windows']);z,evidence=image_margin(j,cache,ctx,q,content,paths)
        else:
            content=paths=None;z=margin(j,cache,ctx,q);evidence=None;assert z==v
        times['new_visual']+=clock(j)-start;new_visual.append(z)
        speech_context=remote_speech(m['windows'],m['packets'][i]) if REVISION==1 else []
        if s is not None:
            new_sq=sq+('\nActual remote speech context at its own times (not current speech):\n'+json.dumps(speech_context,ensure_ascii=False) if speech_context else '')
            start=clock(j);new_s=margin(j,cache,ctx,new_sq);times['new_speech']+=clock(j)-start
            if not speech_context:assert new_s==s
        else:new_s=None;new_sq=None
        new_speech.append(new_s)
        trace=dict(i=i,start=a,end=b,body=body,question=q,native_visual=v,native_speech=s,new_visual=z,new_speech=new_s,new_speech_question=new_sq,source_branch=evidence,
            cache_restored=cache.get_seq_length()==ctx['stance_cache_tokens'],rope_restored=torch.equal(j.model.model.rope_deltas,ctx['rope']))
        assert trace['cache_restored'] and trace['rope_restored']
        if smoke and new_s is not None and not speech_clone:
            start=clock(j);clone=copy.deepcopy(cache)
            try:
                replay=margin(j,clone,ctx,new_sq);assert replay==new_s and clone.get_seq_length()==ctx['stance_cache_tokens']
                trace['speech_clone_exact']=True
            finally:del clone
            diagnostic_calls+=1;speech_clone=True;times['diagnostic']+=clock(j)-start
        if smoke and not any(t.get('clone_exact',False) for t in traces):
            start=clock(j);clone=copy.deepcopy(cache)
            try:
                if evidence is None:replay=margin(j,clone,ctx,q)
                else:replay,replay_evidence=image_margin(j,clone,ctx,q,content,paths);assert replay_evidence==evidence;diagnostic_vision+=1
                assert replay==z and clone.get_seq_length()==ctx['stance_cache_tokens']
                trace['clone_exact']=True
            finally:del clone
            diagnostic_calls+=1;times['diagnostic']+=clock(j)-start
        traces.append(trace)
    native_seconds=sum(times[k] for k in ('prefix','native_visual','native_speech'))
    optimized_seconds=m['cost']['standalone_seconds']+sum(times[k] for k in ('prefix','new_speech','new_visual'))
    base=prediction(row,ctx,native_visual,native_speech,native_seconds,'m1_native')
    new=prediction(row,ctx,new_visual,new_speech,optimized_seconds,'m1_ordered_slots' if REVISION==1 else f'm1_ordered_slots_r{REVISION}',m['cost']['actual_forwards'])
    found=sum(uses_source(w,p) for w,p in zip(m['windows'],m['packets']))
    checks=dict(host=socket.gethostname(),GT_read=False,times=times,source_seconds=m['cost']['standalone_seconds'],source_forwards=m['cost']['actual_forwards'],source_vision=m['cost']['actual_vision_forwards'],
        actual_forwards=j.forward_calls-before,actual_vision=j.vision_calls-before_vision,diagnostic_forwards=diagnostic_calls,diagnostic_vision=diagnostic_vision,
        peak_GiB=max(torch.cuda.max_memory_allocated()/2**30,m['peak_GiB']),caption_calls=m['cost']['caption_calls'],slot_calls=m['cost']['slot_calls'],embedding_calls=m['cost']['embedding_calls'],
        remote_windows=sum(bool(p['source_ids']) for p in m['packets']),prequel_bindings=sum(p['bindings']['prequel']['source_id'] is not None for p in m['packets']),sequel_bindings=sum(p['bindings']['sequel']['source_id'] is not None for p in m['packets']),source_windows=found)
    assert checks['actual_forwards']==base['calls']+len(m['windows'])+sum(v is not None for v in new_speech)+diagnostic_calls
    assert checks['actual_vision']==1+found+diagnostic_vision
    result=dict(base=base,optimized=new,checks=checks,traces=traces,segments=[list(s) for s in segments],
        native_ctx={k:v for k,v in ctx.items() if k not in ('positions','rope','files')},native_rope=ctx['rope'].tolist())
    if REVISION>=2:result['reader_revision']=REVISION
    del cache;return result


def validate_bundle(row,b,segments,m,j,smoke):
    assert b.get('reader_revision',1)==REVISION
    base,new=b['base'],b['optimized'];assert b['segments']==[list(s) for s in segments]
    assert b['checks']['GT_read'] is False and b['checks']['source_seconds']==m['cost']['standalone_seconds']
    wins=fixed_windows(float(row['duration']),SPEC['window_seconds']);assert len(b['traces'])==len(m['windows'])==len(wins)
    for key in ('z_video','stance','prefix_tokens','stance_cache_tokens','stance_cache_logical_start'):assert base['extra'][key]==new['extra'][key]
    assert b['checks']['actual_forwards']==base['calls']+len(wins)+sum(t['native_speech'] is not None for t in b['traces'])+b['checks']['diagnostic_forwards']
    assert b['checks']['diagnostic_forwards']==int(smoke)*(1+int(any(t['native_speech'] is not None for t in b['traces'])))
    assert b['checks']['actual_vision']==1+b['checks']['source_windows']+b['checks']['diagnostic_vision']
    assert base['calls']==3+len(wins)+sum(bool(window_text(segments,a,z).strip()) for a,z in wins)
    assert new['calls']==base['calls']+m['cost']['actual_forwards']
    t=b['checks']['times'];assert abs(base['extra']['standalone_seconds']-sum(t[k] for k in ('prefix','native_visual','native_speech')))<1e-6
    assert abs(new['extra']['standalone_seconds']-m['cost']['standalone_seconds']-sum(t[k] for k in ('prefix','new_speech','new_visual')))<1e-6
    for trace,window,lookup,(a,z),bw,nw in zip(b['traces'],m['windows'],m['packets'],wins,base['extra']['windows'],new['extra']['windows']):
        i=trace['i'];body=window_text(segments,a,z);assert (trace['start'],trace['end'],trace['body'])==(a,z,body)
        assert trace['question']==yesno_question(i,len(wins),a,z,body,'visual')
        assert trace['native_visual']==bw['z_visual'] and trace['new_visual']==nw['z_visual'] and trace['cache_restored'] and trace['rope_restored']
        if not uses_source(window,lookup):assert trace['source_branch'] is None and bw['z_visual']==nw['z_visual']
        else:
            content,paths=memory(window,lookup,m['windows']);_,_,expected=encode_branch(j,b['native_ctx'],trace['question'],content,paths)
            assert expected==trace['source_branch']
        assert bw.get('z_speech')==trace['native_speech'] and nw.get('z_speech')==trace['new_speech']
        context=remote_speech(m['windows'],lookup) if REVISION==1 else []
        expected_speech=yesno_question(i,len(wins),a,z,body,'speech')+('\nActual remote speech context at its own times (not current speech):\n'+json.dumps(context,ensure_ascii=False) if context else '') if body.strip() else None
        assert trace['new_speech_question']==expected_speech
        if not context:assert bw.get('z_speech')==nw.get('z_speech')
    for record in (base,new):
        assert (record['dataset'],record['video_id'],record['duration'],record['native_rate'])==(row['dataset'],row['video_id'],float(row['duration']),4)
        assert record['error'] is None and np.isfinite(record['score_curve']).all()
        ww=record['extra']['windows'];assert all(w['z']==max(w['z_visual'],w.get('z_speech',-math.inf)) for w in ww)
        idx=np.clip(((np.arange(len(record['score_curve']))+.5)/4//SPEC['window_seconds']).astype(int),0,len(wins)-1)
        assert np.array_equal(record['score_curve'],np.asarray([w['z'] for w in ww])[idx])
    if smoke:
        assert sum(t.get('clone_exact',False) for t in b['traces'])==1
        assert sum(t.get('speech_clone_exact',False) for t in b['traces'])==int(any(t['native_speech'] is not None for t in b['traces']))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261006_m1_ordered_slots'/(f'r{REVISION}_full_'+('smoke' if a.smoke else 'main')+('_B' if SPEC.get('interface')=='B' else ''));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,spec=SPEC,GT_read=False,smoke=a.smoke,
        torch=torch.__version__,transformers=transformers.__version__,code='experiments/20261006_m1_ordered_slots/measure.py + src/source_image_branch.py;2026-10-06',command='python -u '+' '.join(__import__('sys').argv))
    if REVISION>=2:cfg['reader_revision']=REVISION
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=j.vision_calls=0
    hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    rows=selected_rows(a.smoke);asr={ds:load_asr(ds) for ds in DATASETS};results={name:[] for name in ('base','optimized')}
    for number,row in enumerate(rows,1):
        segments=asr[row['dataset']].get(row['video_id'],[]);m=json.loads((CACHE/row['dataset']/row['video_id']/'metadata.json').read_text());validate_acquisition(j,m,row,segments,CACHE/row['dataset']/row['video_id'])
        p=out/'records'/row['dataset']/(row['video_id']+'.json');p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():b=json.loads(p.read_text())
        else:
            b=read_video(j,row,segments,m,a.smoke);validate_bundle(row,b,segments,m,j,a.smoke)
            temporary=p.with_suffix('.partial');temporary.write_text(json.dumps(b)+'\n');temporary.replace(p)
        validate_bundle(row,b,segments,m,j,a.smoke)
        for name in results:results[name].append(b[name])
        logging.info('%d/%d %s/%s %.2fs sources=%d',number,len(rows),row['dataset'],row['video_id'],b['optimized']['extra']['standalone_seconds'],b['checks']['source_windows'])
    for name,rr in results.items():
        d=out/name;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**cfg,'measurement':name},indent=2)+'\n');(d/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
    for hook in hooks:hook.remove()
    logging.info('DONE coverage=%d',len(rows))


if __name__=='__main__':main()
