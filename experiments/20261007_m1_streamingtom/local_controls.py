"""Decompose control #1 (no_remote) against r6: attention code, LOCAL frames, role text; no annotations.

Arms per video, all on the same fresh native prefix (r6 G/stance/S unchanged):
- custom_native (B): r6's own visual question ids and positions, read through the R1 attention code with nothing
  inserted; every window.
- local_clean (E): LOCAL frames with their time labels, then the question; no role text; R1 code, no remote.
- no_remote_replay: control #1 exactly (role text kept); must reproduce controls_dualpath_main no_remote.
LOCAL features are the persisted R1 proofs, equal to the fresh features (controls alignment, all 333 videos).
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
from memory import read_tensor
from collect import tick
from reader import visual_margin,factory
from measure import prediction
from src.pre_rotary_memory import attention_scope
from src.stance_cache import build,margin,positions
from src.mllm_judge import Judge,MODEL,yesno_question
from src.video_inputs import frame_paths,load_asr

IMPLEMENTATION='LOCAL decomposition custom_native/local_clean/no_remote_replay;2026-10-08'
EXP=ROOT/'runs/20261007_m1_streamingtom'
R1=EXP/'r1_full_main'
CONTROLS=EXP/'controls_dualpath_main'
ARMS=('custom_native','local_clean','no_remote_replay')


def nothing(layer):
    return []


class ProofFrames:
    """Reader memory interface backed by the persisted R1 LOCAL features; never supplies remote blocks."""
    def __init__(self,frames,records,layers,dimension):
        assert len(frames)==len(records)
        self.blocks=[dict(source_index=f['index']) for f in frames];self.records=records;self.shape=(layers,len(frames),dimension)

    def representatives(self):
        return torch.zeros(self.shape)

    def local_features(self,index):
        r=self.records[index];return read_tensor(r['full_projector']),[read_tensor(m) for m in r['full_deepstack']]


@torch.no_grad()
def custom_margin(j,cache,ctx,memory,question):
    """r6's exact question ids and positions through the R1 attention code, nothing inserted."""
    assert cache.get_seq_length()==ctx['stance_cache_tokens']
    ids,_=j.branch_ids(ctx['msgs'],question,ctx['history'],head_text=ctx['head'])
    ids=torch.tensor([ids],device=j.device);relative,_=positions(j,ids,None)
    assert torch.equal(relative,torch.arange(ids.shape[1],device=j.device).expand(3,1,-1))
    trace={};old=j.model.model.rope_deltas.clone();first=j.forward_calls;vision=j.vision_calls
    try:
        with attention_scope(j,factory(j,cache,ctx,memory,[],relative,list(range(ids.shape[1])),trace,nothing)):
            out=j.model.model.language_model(inputs_embeds=j.model.model.get_input_embeddings()(ids),
                position_ids=relative+ctx['stance_cache_logical_start'],past_key_values=cache,use_cache=True)
            hidden=out.last_hidden_state[0,-1].clone();del out
        assert len(trace)==len(cache.layers) and all(t['remote_ids']==[] and t['source_tokens']==0 and t['suffix_logical_start']==ctx['stance_cache_logical_start'] for t in trace.values())
        assert cache.get_seq_length()==ctx['stance_cache_tokens'] and j.forward_calls-first==1 and j.vision_calls==vision
        return j.margins_fp32(hidden[None])[0]
    finally:j.model.model.rope_deltas=old


@torch.no_grad()
def read(j,row,segments,r1,reference):
    begin=tick(j);first=j.forward_calls;vision=j.vision_calls
    if j.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    _,frames=frames_for(row);assert frames==r1['frames']
    stamp=tick(j);cache,ctx=build(j,frame_paths(row['dataset'],row['video_id'],SPEC['native_frames']),segments)
    times=dict(prefix=tick(j)-stamp,native_visual=0.,native_speech=0.,**{a:0. for a in ARMS})
    windows=windows_for(row,segments);visual=[];speech=[]
    for w in windows:
        stamp=tick(j);visual.append(margin(j,cache,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')));times['native_visual']+=tick(j)-stamp
        if w['body'].strip():
            stamp=tick(j);s=margin(j,cache,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'speech'));times['native_speech']+=tick(j)-stamp
        else:s=None
        speech.append(s)
    layers=len(cache.layers);dimension=cache.layers[0].keys.shape[1]*cache.layers[0].keys.shape[-1]
    memory=ProofFrames(frames,r1['source_acquisition']['records'],layers,dimension)
    values={a:[] for a in ARMS};traces=[];expected=reference['traces']
    for w,v,s,old in zip(windows,visual,speech,expected):
        local=local_ids(frames,w);question=yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')
        assert old['i']==w['i'] and old['LOCAL']==local and old['native_visual']==v
        stamp=tick(j);zb=custom_margin(j,cache,ctx,memory,question);times['custom_native']+=tick(j)-stamp;values['custom_native'].append(zb)
        trace=dict(i=w['i'],bounds=[w['start'],w['end']],LOCAL=local,native_visual=v,native_speech=s,custom_native=zb)
        for arm,role in (('local_clean',None),('no_remote_replay',SPEC['reader_role_text'])):
            if local:
                stamp=tick(j);z,details=visual_margin(j,cache,ctx,memory,frames,local,question,nothing,role);times[arm]+=tick(j)-stamp
                assert all(l['remote_ids']==[] and l['source_tokens']==0 for l in details['layers'].values())
                text=details['input']['suffix_text']
                trace[arm]=dict(z=z,role_in_suffix=SPEC['reader_role_text'] in text,image_counts=details['input']['image_counts'],
                    time_labels=sum(f"Actual LOCAL source at {frames[i]['time']:.3f} seconds." in text for i in local))
            else:
                # No LOCAL frame: native V, exactly as control #1.
                z=v;trace[arm]=None
            values[arm].append(z)
        trace['replay_exact']=trace['no_remote_replay'] is None or trace['no_remote_replay']['z']==old['arms']['no_remote']['z']
        traces.append(trace)
    base=prediction(row,ctx,visual,speech,sum(times[k] for k in ('prefix','native_visual','native_speech')),'m1_native')
    predictions={}
    for a in ARMS:
        p=prediction(row,ctx,values[a],speech,sum(times[k] for k in ('prefix','native_speech',a)),'m1_local_'+a)
        p['code_path']='experiments/20261007_m1_streamingtom/local_controls.py';predictions[a]=p
    covered=sum(bool(t['LOCAL']) for t in traces)
    checks=dict(GT_read=False,host=socket.gethostname(),times=times,actual_forwards=j.forward_calls-first,actual_vision=j.vision_calls-vision,
        covered_windows=covered,missing_LOCAL=len(windows)-covered,peak_GiB=torch.cuda.max_memory_allocated()/2**30 if j.device.type=='cuda' else 0.,
        pipeline_seconds=tick(j)-begin,vision_scope='LOCAL features read from persisted R1 proofs; their single-frame vision cost is not in these times')
    assert checks['actual_forwards']==base['calls']+len(windows)+2*covered and checks['actual_vision']==1
    return dict(version=SPEC['version'],implementation=IMPLEMENTATION,arms=list(ARMS),spec=SPEC,checks=checks,
        native=dict(global_margin=ctx['global_margin'],stance=ctx['stance'],stance_cache_tokens=ctx['stance_cache_tokens']),
        base=base,predictions=predictions,frames=frames,traces=traces)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=EXP/f"local_controls_{'smoke' if a.smoke else 'main'}";out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import av,transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),spec=SPEC,model=MODEL,implementation=IMPLEMENTATION,arms=list(ARMS),
        GT_read=False,smoke=a.smoke,reference=[str(R1.relative_to(ROOT)),str(CONTROLS.relative_to(ROOT))],torch=torch.__version__,
        transformers=transformers.__version__,av=av.__version__,source='experiments/20261007_m1_streamingtom/{local_controls,reader,measure}.py;2026-10-08',
        command='python -u '+' '.join(sys.argv))
    number=1
    while (out/f'pipeline_attempt_{number:04d}.json').exists():number+=1
    audit_path=out/f'pipeline_attempt_{number:04d}.json';audit=dict(host=cfg['host'],completed=False,GT_read=False,videos=[]);start=time.perf_counter();j=None;hooks=[]
    results={name:[] for name in ('base',*ARMS)}
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
            stamp=time.perf_counter();first=j.forward_calls
            path=out/'records'/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True);reused=path.exists()
            segments=asr[row['dataset']].get(row['video_id'],[])
            if reused:bundle=json.loads(path.read_text())
            else:
                r1=json.loads((R1/'records'/row['dataset']/(row['video_id']+'.json')).read_text())
                reference=json.loads((CONTROLS/'records'/row['dataset']/(row['video_id']+'.json')).read_text())
                bundle=read(j,row,segments,r1,reference);del r1,reference
                partial=path.with_suffix('.partial');partial.write_text(json.dumps(bundle,allow_nan=False)+'\n');partial.replace(path)
            results['base'].append(bundle['base'])
            for name in ARMS:results[name].append(bundle['predictions'][name])
            exact=all(t['replay_exact'] for t in bundle['traces'])
            diff=max((abs(t['custom_native']-t['native_visual']) for t in bundle['traces']),default=0.)
            audit['videos'].append(dict(dataset=row['dataset'],video_id=row['video_id'],reused=reused,elapsed_seconds=time.perf_counter()-stamp,
                current_actual_forwards=j.forward_calls-first,replay_exact=exact,custom_native_max_abs=diff))
            audit['elapsed_seconds']=time.perf_counter()-start;audit_path.write_text(json.dumps(audit,indent=2)+'\n')
            logging.info('%d/%d %s/%s replay_exact=%s custom_native_max_abs=%.6f',ordinal,len(rows),row['dataset'],row['video_id'],exact,diff)
        for name,rr in results.items():
            folder=out/name;folder.mkdir(exist_ok=True);(folder/'config.json').write_text(json.dumps({**cfg,'measurement':name},indent=2)+'\n')
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
        audit['completed']=True;logging.info('DONE coverage=%d',len(rows))
    finally:
        for h in hooks:h.remove()
        audit['elapsed_seconds']=time.perf_counter()-start;audit['actual_forwards']=j.forward_calls if j else 0;audit['actual_vision']=j.vision_calls if j else 0
        audit_path.write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
