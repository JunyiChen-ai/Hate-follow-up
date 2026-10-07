"""Candidate39 R1 mechanism controls: one declared change per arm, no annotations.

Job dualpath rebuilds the R1 source memory and reads every covered window three
times: replay (R1 itself, must reproduce the R1 record), no_remote (no remote
blocks) and nearest (the four non-LOCAL frames closest in time, at every layer).
Job uniform rebuilds the memory with content-independent uniform token selection
and reads with R1's per-layer question matching. Everything else is R1.
"""
import argparse
import json
import logging
import os
import socket
import sys
import time
import numpy as np
import torch
from inputs import ROOT,SPEC,DATASETS,selected_rows,frames_for,windows_for,local_ids
from memory import Memory,read_tensor
from collect import collect,tick
from reader import visual_margin
from ctr import BUDGET,group
from measure import prediction
from src.stance_cache import build,margin
from src.mllm_judge import Judge,MODEL,yesno_question
from src.video_inputs import frame_paths,load_asr

IMPLEMENTATION='R1 controls no_remote/nearest/uniform with in-job R1 replay;2026-10-08'
EXP=ROOT/'runs/20261007_m1_streamingtom'
R1=EXP/'r1_full_main'
JOBS=dict(dualpath=('replay','no_remote','nearest'),uniform=('uniform',))


def uniform_group(features,saliency,grid,previous=None,previous_grid=None):
    """Same budget g=min(50,n); raster positions floor((k+.5)n/g); no static/dynamic split, merging or saliency."""
    n=len(features);g=min(BUDGET,n);roots=[(2*k+1)*n//(2*g) for k in range(g)]
    assert n>0 and len(set(roots))==g and roots==sorted(roots)
    return dict(original_tokens=n,retained_tokens=g,static_ids=[],dynamic_ids=list(range(n)),static_budget=0,dynamic_budget=g,
        reset_adjacency=True,similarity=None,groups=[dict(root=r,members=[r],weights=[1.],kind='uniform') for r in roots],dpc=None)


def nearest(frames,local,window):
    """Same count as R1; the non-LOCAL frames closest in time to the window, earlier frame first on ties."""
    def distance(i):
        t=frames[i]['time'];return window['start']-t if t<window['start'] else t-window['end']
    rest=[i for i in range(len(frames)) if i not in local]
    return sorted(rest,key=lambda i:(distance(i),frames[i]['index']))[:SPEC['remote_frames_per_layer']]


def picker(arm,frames,local,window):
    if arm in ('replay','uniform'):return None
    if arm=='no_remote':return lambda layer:[]
    assert arm=='nearest';ids=tuple(nearest(frames,local,window));return lambda layer:ids


def compare(memory,acquisition,reps,queries,reference,job):
    """Rebuilt inputs against the persisted R1 record of the same video (exact equality, no tolerance)."""
    old=reference['source_acquisition']['records'];new=acquisition['records'];assert len(old)==len(new)==len(memory.blocks)
    result=dict(frames=len(new),features_exact=True,deepstack_exact=True,saliency_exact=True,feature_max_abs=0.)
    for block,o,n in zip(memory.blocks,old,new):
        f=read_tensor(block['features']);g=read_tensor(o['full_projector'])
        same=f.shape==g.shape and torch.equal(f,g);result['features_exact']&=same
        if f.shape==g.shape:result['feature_max_abs']=max(result['feature_max_abs'],float((f.float()-g.float()).abs().max()))
        result['deepstack_exact']&=all(torch.equal(read_tensor(a),read_tensor(b)) for a,b in zip(block['deepstack'],o['full_deepstack']))
        result['saliency_exact']&=n['saliency']==o['saliency']
    if job=='dualpath':
        result['plan_exact']=all(n['plan']==o['plan'] for n,o in zip(new,old))
        r=np.load(ROOT/reference['representative_path'],allow_pickle=False);result['representatives_exact']=bool(r.shape==reps.shape and np.array_equal(r,reps))
        q=np.load(ROOT/reference['query_path'],allow_pickle=False);result['replay_queries_exact']=bool(q.shape==queries.shape and np.array_equal(q,queries))
    return result


@torch.no_grad()
def read(j,row,segments,out,job,reference):
    arms=JOBS[job];begin=tick(j);first=j.forward_calls;vision=j.vision_calls
    if j.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    source,frames=frames_for(row);assert frames==reference['frames']
    stamp=tick(j);cache,ctx=build(j,frame_paths(row['dataset'],row['video_id'],SPEC['native_frames']),segments)
    times=dict(prefix=tick(j)-stamp,native_visual=0.,native_speech=0.,source=0.,uncovered=0.,**{a:0. for a in arms})
    windows=windows_for(row,segments);visual=[];speech=[]
    for w in windows:
        stamp=tick(j);visual.append(margin(j,cache,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')));times['native_visual']+=tick(j)-stamp
        if w['body'].strip():
            stamp=tick(j);s=margin(j,cache,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'speech'));times['native_speech']+=tick(j)-stamp
        else:s=None
        speech.append(s)
    temporary=out/'temporary'/row['dataset']/row['video_id']
    if temporary.exists() and any(temporary.iterdir()):
        failed=out/'failed_temporary'/row['dataset'];failed.mkdir(parents=True,exist_ok=True);number=1
        while (failed/(row['video_id']+f'_{number:04d}')).exists():number+=1
        temporary.rename(failed/(row['video_id']+f'_{number:04d}'))
    memory=Memory(temporary)
    try:
        acquisition=collect(j,cache,ctx,memory,frames,None,uniform_group if job=='uniform' else group,persist=False);times['source']=acquisition['seconds']
        layers=len(cache.layers);dimension=cache.layers[0].keys.shape[1]*cache.layers[0].keys.shape[-1]
        reps=memory.representatives().numpy() if frames else np.zeros((layers,0,dimension),np.float32)
        retrieval='replay' if job=='dualpath' else 'uniform'
        queries=np.zeros((len(windows),layers,dimension),np.float32);traces=[];values={a:[] for a in arms}
        for w,v,s in zip(windows,visual,speech):
            local=local_ids(frames,w);question=yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')
            trace=dict(i=w['i'],bounds=[w['start'],w['end']],LOCAL=local,native_visual=v,native_speech=s,arms={})
            if local:
                if 'nearest' in arms:trace['nearest_ids']=nearest(frames,local,w)
                for a in arms:
                    stamp=tick(j);z,details=visual_margin(j,cache,ctx,memory,frames,local,question,picker(a,frames,local,w));times[a]+=tick(j)-stamp
                    if a==retrieval:
                        for l in range(layers):queries[w['i'],l]=details['layers'][l]['query_vector'].numpy()
                    trace['arms'][a]=dict(z=z,remote_ids=[details['layers'][l]['remote_ids'] for l in range(layers)],
                        source_tokens=[details['layers'][l]['source_tokens'] for l in range(layers)])
                    values[a].append(z)
            else:
                # No LOCAL frame: every arm keeps the native V, exactly as R1.
                stamp=tick(j);z=margin(j,cache,ctx,question);assert z==v;times['uncovered']+=tick(j)-stamp
                for a in arms:trace['arms'][a]=None;values[a].append(z)
            traces.append(trace)
        stamp=tick(j);comparison=compare(memory,acquisition,reps,queries,reference,job);comparison_seconds=tick(j)-stamp
        vectors=out/'vectors'/row['dataset'];vectors.mkdir(parents=True,exist_ok=True)
        if job=='uniform':
            np.save(vectors/(row['video_id']+'_source_keys.npy'),reps,allow_pickle=False)
            np.save(vectors/(row['video_id']+'_question_vectors.npy'),queries,allow_pickle=False)
        source_seconds=source['decode_seconds']+times['source']
        base=prediction(row,ctx,visual,speech,sum(times[k] for k in ('prefix','native_visual','native_speech')),'m1_native')
        predictions={}
        for a in arms:
            p=prediction(row,ctx,values[a],speech,source_seconds+sum(times[k] for k in ('prefix','native_speech','uncovered',a)),'m1_streamingtom_'+a,len(frames))
            p['code_path']='experiments/20261007_m1_streamingtom/controls.py';predictions[a]=p
        covered=sum(bool(t['LOCAL']) for t in traces)
        checks=dict(GT_read=False,host=socket.gethostname(),times=times,source_seconds=source_seconds,decode_seconds=source['decode_seconds'],
            comparison_seconds=comparison_seconds,actual_forwards=j.forward_calls-first,actual_vision=j.vision_calls-vision,source_frames=len(frames),
            covered_windows=covered,missing_LOCAL=len(windows)-covered,peak_GiB=torch.cuda.max_memory_allocated()/2**30 if j.device.type=='cuda' else 0.,
            pipeline_seconds=tick(j)-begin)
        assert checks['actual_forwards']==base['calls']+len(frames)+covered*len(arms)+(len(windows)-covered) and checks['actual_vision']==1+len(frames)
        plans=[dict(original_tokens=r['plan']['original_tokens'],roots=[g['root'] for g in r['plan']['groups']],
            static_budget=r['plan']['static_budget'],dynamic_budget=r['plan']['dynamic_budget']) for r in acquisition['records']]
        bundle=dict(version=SPEC['version'],implementation=IMPLEMENTATION,job=job,arms=list(arms),spec=SPEC,checks=checks,comparison=comparison,
            native=dict(global_margin=ctx['global_margin'],stance=ctx['stance'],stance_cache_tokens=ctx['stance_cache_tokens']),
            base=base,predictions=predictions,frames=frames,plans=plans,traces=traces,
            vectors=None if job!='uniform' else dict(source_keys=str((vectors/(row['video_id']+'_source_keys.npy')).relative_to(ROOT)),
                question_vectors=str((vectors/(row['video_id']+'_question_vectors.npy')).relative_to(ROOT))))
        return bundle,memory
    except Exception:memory.close(release=False);raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',choices=tuple(JOBS),required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=EXP/f"controls_{a.job}_{'smoke' if a.smoke else 'main'}";out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import av,transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),spec=SPEC,model=MODEL,implementation=IMPLEMENTATION,job=a.job,arms=list(JOBS[a.job]),
        GT_read=False,smoke=a.smoke,reference=str(R1.relative_to(ROOT)),torch=torch.__version__,transformers=transformers.__version__,av=av.__version__,
        source='experiments/20261007_m1_streamingtom/{controls,measure,collect,reader,memory,ctr,quantization,source_encoding}.py;2026-10-08',
        command='python -u '+' '.join(sys.argv))
    number=1
    while (out/f'pipeline_attempt_{number:04d}.json').exists():number+=1
    audit_path=out/f'pipeline_attempt_{number:04d}.json';audit=dict(host=cfg['host'],completed=False,GT_read=False,videos=[]);start=time.perf_counter();j=None;hooks=[]
    results={name:[] for name in ('base',*JOBS[a.job])}
    try:
        path=out/'config.json'
        if path.exists():
            old=json.loads(path.read_text());assert all(old[k]==cfg[k] for k in cfg if k not in ('date','command'))
        else:path.write_text(json.dumps(cfg,indent=2)+'\n')
        asr={ds:load_asr(ds) for ds in DATASETS};stamp=time.perf_counter();torch.manual_seed(0);torch.set_num_threads(4);j=Judge(MODEL)
        audit['model_load_seconds']=time.perf_counter()-stamp;j.forward_calls=j.vision_calls=0
        hooks=[j.model.model.language_model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),
            j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
        rows=selected_rows(a.smoke)
        for ordinal,row in enumerate(rows,1):
            stamp=time.perf_counter();first=j.forward_calls;vision=j.vision_calls;memory=None
            path=out/'records'/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True);reused=path.exists()
            try:
                segments=asr[row['dataset']].get(row['video_id'],[])
                if reused:bundle=json.loads(path.read_text())
                else:
                    reference=json.loads((R1/'records'/row['dataset']/(row['video_id']+'.json')).read_text())
                    bundle,memory=read(j,row,segments,out,a.job,reference);del reference
                    partial=path.with_suffix('.partial');partial.write_text(json.dumps(bundle,allow_nan=False)+'\n');partial.replace(path)
            finally:
                if memory is not None:memory.close(release=path.exists())
            results['base'].append(bundle['base'])
            for name in JOBS[a.job]:results[name].append(bundle['predictions'][name])
            audit['videos'].append(dict(dataset=row['dataset'],video_id=row['video_id'],reused=reused,elapsed_seconds=time.perf_counter()-stamp,
                current_actual_forwards=j.forward_calls-first,current_actual_vision=j.vision_calls-vision,comparison=bundle['comparison']))
            audit['elapsed_seconds']=time.perf_counter()-start;audit_path.write_text(json.dumps(audit,indent=2)+'\n')
            logging.info('%d/%d %s/%s %s',ordinal,len(rows),row['dataset'],row['video_id'],json.dumps(bundle['comparison']))
        for name,rr in results.items():
            folder=out/name;folder.mkdir(exist_ok=True);(folder/'config.json').write_text(json.dumps({**cfg,'measurement':name},indent=2)+'\n')
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
        audit['completed']=True;logging.info('DONE coverage=%d',len(rows))
    finally:
        for h in hooks:h.remove()
        audit['elapsed_seconds']=time.perf_counter()-start;audit['actual_forwards']=j.forward_calls if j else 0;audit['actual_vision']=j.vision_calls if j else 0
        audit_path.write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
