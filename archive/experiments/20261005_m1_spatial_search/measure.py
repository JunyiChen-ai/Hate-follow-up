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


def unavailable_crop(found):
    if found is None:return None
    x0,y0,x1,y1=found['original_pixel_box'];width=x1-x0;height=y1-y0
    assert width>0 and height>0
    # Frozen Qwen2VLImageProcessor.smart_resize support, not a tuned threshold.
    if max(width,height)/min(width,height)>200:
        return dict(reason='intrinsic_processor_aspect_unsupported',width=width,height=height,max_aspect=200)
    return None


def memory(w,found):
    frame=next(f for f in w['window']['frames'] if f['id']==found['frame']);content=[]
    content.append(dict(type='text',text=json.dumps(dict(actual_window=w['window']['i'],actual_PTS=found['time'],original_shape=found['original_shape'],original_pixel_box=found['original_pixel_box'],search_request=w['decision']['target']),separators=(',',':'))+'\nSearch request is a condition, not a witnessed fact. Original source frame:\n'))
    content.append(dict(type='image'))
    content.extend([dict(type='text',text='Exact source crop at the given original location; search request is not a witnessed fact.\n'),dict(type='image')])
    return content,[ROOT/frame['path'],ROOT/found['path']]


def prediction(row,ctx,visual,speech,seconds,method,source_forwards=0):
    wins=fixed_windows(float(row['duration']),SPEC['window_seconds']);windows=[]
    for i,((a,b),v,s) in enumerate(zip(wins,visual,speech)):
        w=dict(i=i,start=a,end=b,z_visual=v,z=max(v,s) if s is not None else v)
        if s is not None:w['z_speech']=s
        windows.append(w)
    idx=np.clip(((np.arange(math.ceil(float(row['duration'])*4))+.5)/4//SPEC['window_seconds']).astype(int),0,len(wins)-1)
    return dict(schema_version=1,method=method,dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),native_rate=4,
        score_curve=np.asarray([w['z'] for w in windows])[idx].tolist(),intervals=[],error=None,seed=0,calls=3+len(wins)+sum(s is not None for s in speech)+source_forwards,
        code_path='experiments/20261005_m1_spatial_search/measure.py',extra=dict(z_video=ctx['global_margin'],stance=ctx['stance'],prefix_tokens=ctx['prefix_tokens'],
            stance_cache_tokens=ctx['stance_cache_tokens'],stance_cache_logical_start=ctx['stance_cache_logical_start'],windows=windows,standalone_seconds=seconds))


@torch.no_grad()
def read_video(j,row,segments,m,smoke):
    start=clock(j);before=j.forward_calls;before_vision=j.vision_calls;torch.cuda.reset_peak_memory_stats()
    cache,ctx=build(j,frame_paths(row['dataset'],row['video_id'],SPEC['native_frames']),segments)
    times=dict(prefix=clock(j)-start,native_visual=0.,native_speech=0.,new_visual=0.,diagnostic=0.)
    native_visual=[];native_speech=[];new_visual=[];traces=[];diagnostic_calls=diagnostic_vision=0
    for i,((a,b),w) in enumerate(zip(fixed_windows(float(row['duration']),SPEC['window_seconds']),m['windows'])):
        body=window_text(segments,a,b);q=yesno_question(i,len(m['windows']),a,b,body,'visual')
        start=clock(j);v=margin(j,cache,ctx,q);times['native_visual']+=clock(j)-start;native_visual.append(v)
        if body.strip():
            sq=yesno_question(i,len(m['windows']),a,b,body,'speech');start=clock(j);s=margin(j,cache,ctx,sq);times['native_speech']+=clock(j)-start
        else:s=None
        native_speech.append(s)
        start=clock(j)
        unavailable=unavailable_crop(w['found'])
        if w['found'] is not None and unavailable is None:
            content,paths=memory(w,w['found']);z,evidence=image_margin(j,cache,ctx,q,content,paths)
        else:
            content=paths=None;z=margin(j,cache,ctx,q);evidence=None;assert z==v
        times['new_visual']+=clock(j)-start;new_visual.append(z)
        trace=dict(i=i,start=a,end=b,body=body,question=q,native_visual=v,native_speech=s,new_visual=z,source_branch=evidence,
            cache_restored=cache.get_seq_length()==ctx['stance_cache_tokens'],rope_restored=torch.equal(j.model.model.rope_deltas,ctx['rope']))
        assert trace['cache_restored'] and trace['rope_restored']
        if unavailable is not None:trace['source_unavailable']=unavailable
        if smoke and diagnostic_calls==0:
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
    optimized_seconds=m['standalone_seconds']+sum(times[k] for k in ('prefix','native_speech','new_visual'))
    base=prediction(row,ctx,native_visual,native_speech,native_seconds,'m1_native')
    new=prediction(row,ctx,new_visual,native_speech,optimized_seconds,'m1_spatial_search',m['actual_forwards'])
    found=sum(w['found'] is not None for w in m['windows']);unsupported=sum(unavailable_crop(w['found']) is not None for w in m['windows']);used=found-unsupported
    checks=dict(host=socket.gethostname(),GT_read=False,times=times,source_seconds=m['standalone_seconds'],source_forwards=m['actual_forwards'],source_vision=m['actual_vision_forwards'],
        actual_forwards=j.forward_calls-before,actual_vision=j.vision_calls-before_vision,diagnostic_forwards=diagnostic_calls,diagnostic_vision=diagnostic_vision,
        peak_GiB=max(torch.cuda.max_memory_allocated()/2**30,m['peak_GiB']),detection_calls=sum(w['detector'] is not None for w in m['windows']),
        search_calls=sum(len(w['steps']) for w in m['windows']),search_windows=sum(w['decision']['kind']=='SEARCH' for w in m['windows']),found_windows=found,used_crop_windows=used,unsupported_crop_windows=unsupported)
    assert checks['actual_forwards']==base['calls']+len(m['windows'])+diagnostic_calls
    assert checks['actual_vision']==1+used+diagnostic_vision
    result=dict(base=base,optimized=new,checks=checks,traces=traces,segments=[list(s) for s in segments],
        native_ctx={k:v for k,v in ctx.items() if k not in ('positions','rope','files')},native_rope=ctx['rope'].tolist())
    del cache;return result


def validate_bundle(row,b,segments,m,j,smoke):
    base,new=b['base'],b['optimized'];assert b['segments']==m['segments']==[list(s) for s in segments]
    assert b['checks']['GT_read'] is False and b['checks']['source_seconds']==m['standalone_seconds']
    wins=fixed_windows(float(row['duration']),SPEC['window_seconds']);assert len(b['traces'])==len(m['windows'])==len(wins)
    for key in ('z_video','stance','prefix_tokens','stance_cache_tokens','stance_cache_logical_start'):assert base['extra'][key]==new['extra'][key]
    assert b['checks']['actual_forwards']==base['calls']+len(wins)+b['checks']['diagnostic_forwards']
    assert b['checks']['diagnostic_forwards']==int(smoke)
    found=sum(w['found'] is not None for w in m['windows']);unsupported=sum(unavailable_crop(w['found']) is not None for w in m['windows']);used=found-unsupported
    assert b['checks']['found_windows']==found and b['checks'].get('used_crop_windows',found)==used and b['checks'].get('unsupported_crop_windows',0)==unsupported
    assert b['checks']['actual_vision']==1+used+b['checks']['diagnostic_vision']
    assert base['calls']==3+len(wins)+sum(bool(window_text(segments,a,z).strip()) for a,z in wins)
    assert new['calls']==base['calls']+m['actual_forwards']
    t=b['checks']['times'];assert abs(base['extra']['standalone_seconds']-sum(t[k] for k in ('prefix','native_visual','native_speech')))<1e-6
    assert abs(new['extra']['standalone_seconds']-m['standalone_seconds']-sum(t[k] for k in ('prefix','native_speech','new_visual')))<1e-6
    for trace,source,(a,z),bw,nw in zip(b['traces'],m['windows'],wins,base['extra']['windows'],new['extra']['windows']):
        i=trace['i'];body=window_text(segments,a,z);assert (trace['start'],trace['end'],trace['body'])==(a,z,body)
        assert trace['question']==yesno_question(i,len(wins),a,z,body,'visual')
        assert trace['native_visual']==bw['z_visual'] and trace['new_visual']==nw['z_visual'] and trace['cache_restored'] and trace['rope_restored']
        unavailable=unavailable_crop(source['found']);assert trace.get('source_unavailable')==unavailable
        if source['found'] is None or unavailable is not None:assert trace['source_branch'] is None and bw['z_visual']==nw['z_visual']
        else:
            content,paths=memory(source,source['found']);_,_,expected=encode_branch(j,b['native_ctx'],trace['question'],content,paths)
            assert expected==trace['source_branch']
        assert bw.get('z_speech')==nw.get('z_speech')==trace['native_speech']
    for record in (base,new):
        assert (record['dataset'],record['video_id'],record['duration'],record['native_rate'])==(row['dataset'],row['video_id'],float(row['duration']),4)
        assert record['error'] is None and np.isfinite(record['score_curve']).all()
        ww=record['extra']['windows'];assert all(w['z']==max(w['z_visual'],w.get('z_speech',-math.inf)) for w in ww)
        idx=np.clip(((np.arange(len(record['score_curve']))+.5)/4//SPEC['window_seconds']).astype(int),0,len(wins)-1)
        assert np.array_equal(record['score_curve'],np.asarray([w['z'] for w in ww])[idx])
    if smoke:assert sum(t.get('clone_exact',False) for t in b['traces'])==1


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261005_m1_spatial_search'/('r1_full_smoke' if a.smoke else 'r1_full_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,spec=SPEC,GT_read=False,smoke=a.smoke,
        torch=torch.__version__,transformers=transformers.__version__,code='experiments/20261005_m1_spatial_search/measure.py + src/source_image_branch.py;2026-10-05',command='python -u '+' '.join(__import__('sys').argv))
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=j.vision_calls=0
    hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    rows=selected_rows(a.smoke);asr={ds:load_asr(ds) for ds in DATASETS};results={name:[] for name in ('base','optimized')}
    for number,row in enumerate(rows,1):
        segments=asr[row['dataset']].get(row['video_id'],[]);m=json.loads((CACHE/row['dataset']/row['video_id']/'metadata.json').read_text());validate_acquisition(m,row,segments,j)
        p=out/'records'/row['dataset']/(row['video_id']+'.json');p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():b=json.loads(p.read_text())
        else:
            b=read_video(j,row,segments,m,a.smoke);validate_bundle(row,b,segments,m,j,a.smoke)
            temporary=p.with_suffix('.partial');temporary.write_text(json.dumps(b)+'\n');temporary.replace(p)
        validate_bundle(row,b,segments,m,j,a.smoke)
        for name in results:results[name].append(b[name])
        logging.info('%d/%d %s/%s %.2fs found=%d',number,len(rows),row['dataset'],row['video_id'],b['optimized']['extra']['standalone_seconds'],b['checks']['found_windows'])
    for name,rr in results.items():
        d=out/name;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**cfg,'measurement':name},indent=2)+'\n');(d/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
    for hook in hooks:hook.remove()
    logging.info('DONE coverage=%d',len(rows))


if __name__=='__main__':main()
