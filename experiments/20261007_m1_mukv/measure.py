"""Paired whole-video native and coherent-source measurements without labels."""
import argparse
import copy
import json
import logging
import math
import os
import socket
import sys
import time
import numpy as np
import torch
from inputs import ROOT,SPEC,DATASETS,selected_rows,frames_for,windows_for
from memory import Memory
from collect import collect,tick
from reader import probe,visual_margin
from src.mllm_judge import Judge,MODEL,yesno_question
from src.stance_cache import build,margin
from src.video_inputs import frame_paths,load_asr
from src.native_input_binding import save as save_native

IMPLEMENTATION='R1 full-history paper-anchored multigrain DCP/coherence;2026-10-07'


def serial(value):
    if torch.is_tensor(value):return value.tolist()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value


def prediction(row,ctx,visual,speech,seconds,method,source_calls=0,probes=0):
    ww=[]
    for w,v,s in zip(windows_for(row,[]),visual,speech):
        one=dict(i=w['i'],start=w['start'],end=w['end'],z_visual=v,z=max(v,s) if s is not None else v)
        if s is not None:one['z_speech']=s
        ww.append(one)
    idx=np.clip(((np.arange(math.ceil(float(row['duration'])*4))+.5)/4//8).astype(int),0,len(ww)-1)
    return dict(schema_version=1,method=method,dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),native_rate=4,
        score_curve=np.asarray([w['z'] for w in ww])[idx].tolist(),intervals=[],error=None,seed=0,calls=3+len(ww)+sum(s is not None for s in speech)+source_calls+probes,
        code_path='experiments/20261007_m1_mukv/measure.py',extra=dict(z_video=ctx['global_margin'],stance=ctx['stance'],prefix_tokens=ctx['prefix_tokens'],
            stance_cache_tokens=ctx['stance_cache_tokens'],stance_cache_logical_start=ctx['stance_cache_logical_start'],windows=ww,standalone_seconds=seconds))


@torch.no_grad()
def read_video(j,row,segments,out,smoke):
    start=tick(j);first=j.forward_calls;vision=j.vision_calls
    if j.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    stamp=tick(j);source,window_frames=frames_for(row);input_validation=tick(j)-stamp
    stamp=tick(j);cache,ctx=build(j,frame_paths(row['dataset'],row['video_id'],SPEC['native_frames']),segments)
    times=dict(prefix=tick(j)-stamp,native_visual=0.,native_speech=0.,source=0.,probe=0.,new_visual=0.,diagnostic=0.)
    windows=windows_for(row,segments);visual=[];speech=[]
    for w in windows:
        stamp=tick(j);visual.append(margin(j,cache,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')));times['native_visual']+=tick(j)-stamp
        if w['body'].strip():
            stamp=tick(j);s=margin(j,cache,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'speech'));times['native_speech']+=tick(j)-stamp
        else:s=None
        speech.append(s)
    proof=out/'proof'/row['dataset']/row['video_id'];proof.mkdir(parents=True,exist_ok=True)
    stamp=tick(j);binding=save_native(j,row,segments,ctx,SPEC,IMPLEMENTATION,proof/'native_pixels.npy');binding_seconds=tick(j)-stamp
    temporary=out/'temporary'/row['dataset']/row['video_id']
    if temporary.exists() and any(temporary.iterdir()):
        failed=out/'failed_temporary'/row['dataset'];failed.mkdir(parents=True,exist_ok=True);n=1
        while (failed/(row['video_id']+f'_{n:04d}')).exists():n+=1
        temporary.rename(failed/(row['video_id']+f'_{n:04d}'))
    memory=Memory(temporary)
    try:
        acquisition,cached=collect(j,cache,ctx,memory,window_frames,proof/'source');times['source']=acquisition['seconds']
        reps=torch.stack(memory.reps).numpy() if memory.reps else np.zeros((0,cache.layers[0].keys.shape[1]*cache.layers[0].keys.shape[-1]),np.float32)
        np.save(proof/'source_keys.npy',reps,allow_pickle=False);values=[];traces=[];clones=0;vclone=False;sclone=False;probes=0
        for w,media,v,s in zip(windows,window_frames,visual,speech):
            question=yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual');selection=None;branch=None
            if media:
                stamp=tick(j);selected,details=probe(j,cache,ctx,memory,w['i'],media,cached,w['body']);times['probe']+=tick(j)-stamp;probes+=1
                selection=serial(details);stamp=tick(j);z,branch=visual_margin(j,cache,ctx,memory,media,cached,selected,question);times['new_visual']+=tick(j)-stamp
            else:
                stamp=tick(j);z=margin(j,cache,ctx,question);times['new_visual']+=tick(j)-stamp;assert z==v;selected=[]
            trace=dict(i=w['i'],bounds=[w['start'],w['end']],body=w['body'],question=question,LOCAL=media,native_visual=v,native_speech=s,
                new_visual=z,probe=selection,branch=branch,selected_source_blocks=selected)
            if smoke and not vclone:
                stamp=tick(j);clone=copy.deepcopy(cache)
                other=visual_margin(j,clone,ctx,memory,media,cached,selected,question)[0] if media else margin(j,clone,ctx,question)
                assert other==z and all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,clone.layers))
                vclone=True;clones+=1;trace['visual_clone_exact']=True;times['diagnostic']+=tick(j)-stamp;del clone
            if smoke and s is not None and not sclone:
                stamp=tick(j);clone=copy.deepcopy(cache);assert margin(j,clone,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'speech'))==s
                assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,clone.layers))
                sclone=True;clones+=1;trace['speech_clone_exact']=True;times['diagnostic']+=tick(j)-stamp;del clone
            values.append(z);traces.append(trace)
        source_seconds=source['decode_seconds']+times['source']+binding_seconds
        base=prediction(row,ctx,visual,speech,sum(times[k] for k in ('prefix','native_visual','native_speech')),'m1_native')
        new=prediction(row,ctx,values,speech,source_seconds+sum(times[k] for k in ('prefix','native_speech','probe','new_visual')),'m1_mukv',acquisition['actual_LM'],probes)
        checks=dict(GT_read=False,host=socket.gethostname(),times=times,source_seconds=source_seconds,decode_seconds=source['decode_seconds'],native_binding_seconds=binding_seconds,
            input_validation_seconds=input_validation,actual_forwards=j.forward_calls-first,actual_vision=j.vision_calls-vision,source_LM=acquisition['actual_LM'],source_vision=acquisition['actual_vision'],
            source_blocks=len(memory.blocks),source_frames=sum(len(f) for f in window_frames),probe_calls=probes,diagnostic_forwards=clones,
            missing_LOCAL=sum(not f for f in window_frames),selected_context_blocks=sum(len(t['selected_source_blocks']) for t in traces),
            peak_GiB=torch.cuda.max_memory_allocated()/2**30 if j.device.type=='cuda' else 0.,temporary_KV_bytes=sum(b['storage']['bytes'] for b in memory.blocks),
            actual_reader_seconds=tick(j)-start,scope='all actualvision/sourcefullhistory/lastattention/FFT/probe/reader/storage charged, original source decode charged; native20/fullASR acquisition extra')
        assert checks['actual_forwards']==base['calls']+acquisition['actual_LM']+probes+len(windows)+clones
        assert checks['actual_vision']==1+acquisition['actual_vision']
        return dict(version=SPEC['version'],spec=SPEC,base=base,optimized=new,checks=checks,binding=binding,native_ctx={k:v for k,v in ctx.items() if k not in ('positions','rope','files')},
            native_rope=ctx['rope'].tolist(),segments=[list(s) for s in segments],source_input=source,window_frames=window_frames,source_blocks=memory.metadata(),
            source_acquisition=acquisition,traces=traces,representative_path=str((proof/'source_keys.npy').relative_to(ROOT))),memory
    except Exception:memory.close(release=False);raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261007_m1_mukv'/('r1_full_smoke' if a.smoke else 'r1_full_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()]);logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import av,transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,spec=SPEC,implementation=IMPLEMENTATION,GT_read=False,smoke=a.smoke,
        torch=torch.__version__,transformers=transformers.__version__,av=av.__version__,code='experiments/20261007_m1_mukv/{measure,collect,encoding,source_attention,compression,coherence,reader,memory}.py;2026-10-07',command='python -u '+' '.join(sys.argv))
    n=1
    while (out/f'pipeline_attempt_{n:04d}.json').exists():n+=1
    audit_path=out/f'pipeline_attempt_{n:04d}.json';start=time.perf_counter();audit=dict(host=cfg['host'],completed=False,GT_read=False,videos=[]);j=None;hooks=[];predictions={name:[] for name in ('base','optimized')}
    try:
        path=out/'config.json'
        if path.exists():
            old=json.loads(path.read_text());assert all(old[k]==cfg[k] for k in cfg if k not in ('date','command'))
        else:path.write_text(json.dumps(cfg,indent=2)+'\n')
        asr={ds:load_asr(ds) for ds in DATASETS};stamp=time.perf_counter();torch.manual_seed(0);torch.set_num_threads(SPEC['cpu_threads']);j=Judge(MODEL);audit['model_load_seconds']=time.perf_counter()-stamp;j.forward_calls=j.vision_calls=0
        hooks=[j.model.model.language_model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
        from validate import validate_bundle
        rows=selected_rows(a.smoke)
        for i,row in enumerate(rows,1):
            stamp=time.perf_counter();first=j.forward_calls;vision=j.vision_calls;segments=asr[row['dataset']].get(row['video_id'],[]);memory=None
            path=out/'records'/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True);reused=path.exists()
            try:
                if reused:bundle=json.loads(path.read_text())
                else:bundle,memory=read_video(j,row,segments,out,a.smoke)
                validate_bundle(j,row,segments,bundle,a.smoke)
                if not reused:
                    partial=path.with_suffix('.partial');partial.write_text(json.dumps(bundle,allow_nan=False)+'\n');partial.replace(path)
            finally:
                if memory is not None:memory.close(release=path.exists())
            for name in predictions:predictions[name].append(bundle[name])
            audit['videos'].append(dict(dataset=row['dataset'],video_id=row['video_id'],reused=reused,elapsed_seconds=time.perf_counter()-stamp,current_actual_LM=j.forward_calls-first,current_actual_vision=j.vision_calls-vision))
            audit['elapsed_seconds']=time.perf_counter()-start;audit_path.write_text(json.dumps(audit,indent=2)+'\n');logging.info('%d/%d %s/%s',i,len(rows),row['dataset'],row['video_id'])
        for name,rr in predictions.items():
            folder=out/name;folder.mkdir(exist_ok=True);(folder/'config.json').write_text(json.dumps({**cfg,'measurement':name},indent=2)+'\n');(folder/'predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in rr))
        audit['completed']=True;logging.info('DONE coverage=%d',len(rows))
    finally:
        for h in hooks:h.remove()
        audit['elapsed_seconds']=time.perf_counter()-start;audit['actual_LM']=j.forward_calls if j else 0;audit['actual_vision']=j.vision_calls if j else 0;audit_path.write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
