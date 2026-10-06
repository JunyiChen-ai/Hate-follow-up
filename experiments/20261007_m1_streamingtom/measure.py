"""Whole-video native/source/LOCAL-context measurements, without annotations."""
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
from inputs import ROOT,SPEC,DATASETS,selected_rows,frames_for,windows_for,local_ids
from memory import Memory
from collect import collect,tick
from reader import visual_margin
from src.stance_cache import build,margin
from src.mllm_judge import Judge,MODEL,yesno_question
from src.video_inputs import frame_paths,load_asr
from src.native_input_binding import save as save_native

IMPLEMENTATION='R1 actual dual-path/uint4/LOCAL-context source memory;2026-10-07'


def prediction(row,ctx,visual,speech,seconds,name,source_calls=0):
    ww=[]
    for w,v,s in zip(windows_for(row,[]),visual,speech):
        one=dict(i=w['i'],start=w['start'],end=w['end'],z_visual=v,z=max(v,s) if s is not None else v)
        if s is not None:one['z_speech']=s
        ww.append(one)
    idx=np.clip(((np.arange(math.ceil(float(row['duration'])*4))+.5)/4//SPEC['window_seconds']).astype(int),0,len(ww)-1)
    return dict(schema_version=1,method=name,dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),native_rate=4,
        score_curve=np.asarray([w['z'] for w in ww])[idx].tolist(),intervals=[],error=None,seed=0,
        calls=3+len(ww)+sum(s is not None for s in speech)+source_calls,code_path='experiments/20261007_m1_streamingtom/measure.py',
        extra=dict(z_video=ctx['global_margin'],stance=ctx['stance'],prefix_tokens=ctx['prefix_tokens'],stance_cache_tokens=ctx['stance_cache_tokens'],
            stance_cache_logical_start=ctx['stance_cache_logical_start'],windows=ww,standalone_seconds=seconds))


@torch.no_grad()
def read_video(j,row,segments,out,smoke):
    begin=tick(j);first=j.forward_calls;vision=j.vision_calls
    if j.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    stamp=tick(j);source,frames=frames_for(row);input_validation=tick(j)-stamp
    stamp=tick(j);cache,ctx=build(j,frame_paths(row['dataset'],row['video_id'],SPEC['native_frames']),segments)
    times=dict(prefix=tick(j)-stamp,native_visual=0.,native_speech=0.,source=0.,new_visual=0.,diagnostic=0.)
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
        failed=out/'failed_temporary'/row['dataset'];failed.mkdir(parents=True,exist_ok=True);number=1
        while (failed/(row['video_id']+f'_{number:04d}')).exists():number+=1
        temporary.rename(failed/(row['video_id']+f'_{number:04d}'))
    memory=Memory(temporary)
    try:
        acquisition=collect(j,cache,ctx,memory,frames,proof/'source');times['source']=acquisition['seconds']
        layers=len(cache.layers);dimension=cache.layers[0].keys.shape[1]*cache.layers[0].keys.shape[-1]
        representatives=memory.representatives().numpy() if frames else np.zeros((layers,0,dimension),np.float32)
        np.save(proof/'source_keys.npy',representatives,allow_pickle=False)
        queries=np.zeros((len(windows),layers,dimension),np.float32);traces=[];values=[];clones=0;vclone=False;sclone=False
        for w,v,s in zip(windows,visual,speech):
            local=local_ids(frames,w);question=yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual');branch=None
            stamp=tick(j)
            if local:
                z,details=visual_margin(j,cache,ctx,memory,frames,local,question)
                for l in range(layers):queries[w['i'],l]=details['layers'][l]['query_vector'].numpy()
                branch=dict(input=details['input'],layers=[{k:v for k,v in details['layers'][l].items() if k!='query_vector'} for l in range(layers)])
            else:z=margin(j,cache,ctx,question);assert z==v
            times['new_visual']+=tick(j)-stamp
            trace=dict(i=w['i'],bounds=[w['start'],w['end']],body=w['body'],question=question,LOCAL=local,native_visual=v,native_speech=s,new_visual=z,branch=branch)
            if smoke and not vclone:
                stamp=tick(j);clone=copy.deepcopy(cache)
                replay=visual_margin(j,clone,ctx,memory,frames,local,question)[0] if local else margin(j,clone,ctx,question)
                assert replay==z and all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,clone.layers))
                trace['visual_clone_exact']=True;vclone=True;clones+=1;times['diagnostic']+=tick(j)-stamp;del clone
            if smoke and s is not None and not sclone:
                stamp=tick(j);clone=copy.deepcopy(cache)
                assert margin(j,clone,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'speech'))==s
                assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,clone.layers))
                trace['speech_clone_exact']=True;sclone=True;clones+=1;times['diagnostic']+=tick(j)-stamp;del clone
            values.append(z);traces.append(trace)
        np.save(proof/'question_vectors.npy',queries,allow_pickle=False)
        source_seconds=source['decode_seconds']+times['source']+binding_seconds
        base=prediction(row,ctx,visual,speech,sum(times[k] for k in ('prefix','native_visual','native_speech')),'m1_native')
        new=prediction(row,ctx,values,speech,source_seconds+sum(times[k] for k in ('prefix','native_speech','new_visual')),'m1_streamingtom',len(frames))
        checks=dict(GT_read=False,host=socket.gethostname(),times=times,source_seconds=source_seconds,decode_seconds=source['decode_seconds'],
            input_validation_seconds=input_validation,native_binding_seconds=binding_seconds,actual_forwards=j.forward_calls-first,actual_vision=j.vision_calls-vision,
            diagnostic_forwards=clones,source_frames=len(frames),source_forwards=len(frames),source_vision=len(frames),
            missing_LOCAL=sum(not t['LOCAL'] for t in traces),remote_reads=sum(bool(l['remote_ids']) for t in traces if t['branch'] for l in t['branch']['layers']),
            peak_GiB=torch.cuda.max_memory_allocated()/2**30 if j.device.type=='cuda' else 0.,
            temporary_KV_bytes=sum(b['storage_bytes'] for b in memory.blocks),pipeline_seconds=tick(j)-begin,
            standalone_scope='includes source decoding/native binding/fresh vision/reduction/LM/storage-proof/LOCAL reader; original native20/fullASR acquisition additional')
        assert checks['actual_forwards']==base['calls']+len(frames)+len(windows)+clones and checks['actual_vision']==1+len(frames)
        native={k:v for k,v in ctx.items() if k not in ('positions','rope','files')}
        bundle=dict(version=SPEC['version'],spec=SPEC,base=base,optimized=new,checks=checks,binding=binding,native_ctx=native,native_rope=ctx['rope'].tolist(),
            segments=[list(s) for s in segments],frames=frames,source_input=source,source_blocks=memory.metadata(),source_acquisition=acquisition,traces=traces,
            representative_path=str((proof/'source_keys.npy').relative_to(ROOT)),query_path=str((proof/'question_vectors.npy').relative_to(ROOT)))
        return bundle,memory
    except Exception:memory.close(release=False);raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261007_m1_streamingtom'/('r1_full_smoke' if a.smoke else 'r1_full_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import av,transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),spec=SPEC,model=MODEL,implementation=IMPLEMENTATION,GT_read=False,smoke=a.smoke,
        torch=torch.__version__,transformers=transformers.__version__,av=av.__version__,source='experiments/20261007_m1_streamingtom/{measure,collect,reader,memory,ctr,quantization}.py;2026-10-07',command='python -u '+' '.join(sys.argv))
    number=1
    while (out/f'pipeline_attempt_{number:04d}.json').exists():number+=1
    audit_path=out/f'pipeline_attempt_{number:04d}.json';audit=dict(host=cfg['host'],completed=False,GT_read=False,videos=[]);start=time.perf_counter();j=None;hooks=[]
    results={name:[] for name in ('base','optimized')}
    try:
        path=out/'config.json'
        if path.exists():
            old=json.loads(path.read_text());assert all(old[k]==cfg[k] for k in cfg if k not in ('date','command'))
        else:path.write_text(json.dumps(cfg,indent=2)+'\n')
        asr={ds:load_asr(ds) for ds in DATASETS};stamp=time.perf_counter();torch.manual_seed(0);torch.set_num_threads(4);j=Judge(MODEL)
        audit['model_load_seconds']=time.perf_counter()-stamp;j.forward_calls=j.vision_calls=0
        hooks=[j.model.model.language_model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),
            j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
        from validate import validate_bundle
        rows=selected_rows(a.smoke)
        for ordinal,row in enumerate(rows,1):
            stamp=time.perf_counter();first=j.forward_calls;vision=j.vision_calls;memory=None
            path=out/'records'/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True);reused=path.exists()
            try:
                segments=asr[row['dataset']].get(row['video_id'],[])
                if reused:bundle=json.loads(path.read_text())
                else:bundle,memory=read_video(j,row,segments,out,a.smoke)
                validate_bundle(j,row,segments,bundle,a.smoke)
                if not reused:
                    partial=path.with_suffix('.partial');partial.write_text(json.dumps(bundle,allow_nan=False)+'\n');partial.replace(path)
            finally:
                if memory is not None:memory.close(release=path.exists())
            for name in results:results[name].append(bundle[name])
            audit['videos'].append(dict(dataset=row['dataset'],video_id=row['video_id'],reused=reused,elapsed_seconds=time.perf_counter()-stamp,
                current_actual_forwards=j.forward_calls-first,current_actual_vision=j.vision_calls-vision))
            audit['elapsed_seconds']=time.perf_counter()-start;audit_path.write_text(json.dumps(audit,indent=2)+'\n')
            logging.info('%d/%d %s/%s',ordinal,len(rows),row['dataset'],row['video_id'])
        for name,rr in results.items():
            folder=out/name;folder.mkdir(exist_ok=True);(folder/'config.json').write_text(json.dumps({**cfg,'measurement':name},indent=2)+'\n')
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
        audit['completed']=True;logging.info('DONE coverage=%d',len(rows))
    finally:
        for h in hooks:h.remove()
        audit['elapsed_seconds']=time.perf_counter()-start;audit['actual_forwards']=j.forward_calls if j else 0;audit['actual_vision']=j.vision_calls if j else 0
        audit_path.write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
