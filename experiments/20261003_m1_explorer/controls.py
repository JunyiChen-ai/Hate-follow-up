#!/usr/bin/env python3
"""Declared full-video acquisition controls. No labels or performance input."""
import argparse
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
from measure import ROOT,tick,rope_positions,read_suffix,read_images
from explorer import FrameAttention,FrameSource,choose_frames
from src.mllm_judge import Judge,MODEL,VIDEO_QUESTION,yesno_question
from src.video_inputs import FPS,frame_paths,load_asr,load_manifest,fixed_windows,window_text


def uniform(candidates,start,end,count):
    """Window quantiles, nearest still-available actual PTS; chronological ties."""
    available=list(candidates);chosen=[]
    for k in range(count):
        target=start+(k+.5)*(end-start)/count
        i=min(range(len(available)),key=lambda j:(abs(available[j]['time']-target),available[j]['time']))
        chosen.append(available.pop(i))
    return chosen


def mismatch_map(traces):
    groups={};mapping={};matched=set()
    for t in traces:
        frames=t['rounds'][-1]['acquired'] if t['rounds'] else []
        if frames:groups.setdefault(len(frames),[]).append((t['window'],frames))
    for group in groups.values():
        group.sort();k=len(group)
        for i,(receiver,frames) in enumerate(group):
            donor,other=group[(i+math.ceil(k/2))%k]
            if k>=2:matched.add(receiver)
            for e,d in zip(frames,other):mapping[e['index']]=d
    return mapping,matched


class MismatchSource:
    def __init__(self,source,mapping):self.source=source;self.mapping=mapping
    def image(self,entry):return self.source.image(self.mapping[entry['index']])


@torch.no_grad()
def read_control(j,engine,row,segments,out,arm,main,smoke=False):
    ds,vid,duration=row['dataset'],row['video_id'],float(row['duration'])
    frames=frame_paths(ds,vid,20,'k20');wins=fixed_windows(duration,8)
    texts=[window_text(segments,a,b) for a,b in wins];V=len(wins)
    B=V+sum(bool(t.strip()) for t in texts)
    assert len(main['traces'])==V
    mapping,matched=mismatch_map(main['traces']) if arm=='mismatch' else ({},set())
    j.model.model.rope_deltas=None;torch.cuda.reset_peak_memory_stats();initial=j.forward_calls
    started=tick();msgs,files=j.prefix_messages(frames,segments);text,enc=j.encode_prefix(msgs,files)
    counts=list(j.img_tokens);engine.start('prefix',enc['input_ids'][0]==j.image_token_id)
    try:cache=j.prefix_cache(enc)
    finally:engine.stop()
    P=cache.get_seq_length();qid,qtext=j.branch_ids(msgs,VIDEO_QUESTION)
    zv=j.cached_margin(cache,qid,in_place=True);stance='Yes' if zv>0 else 'No'
    aid,atext=j.answer_ids(msgs,VIDEO_QUESTION,stance);j.extend_cache(cache,aid)
    history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',stance)]
    head=text+qtext+atext;n=cache.get_seq_length();rope=j.model.model.rope_deltas.clone()
    ids=torch.cat((enc['input_ids'].to(j.device),torch.tensor([qid+aid],device=j.device)),1)
    grids=enc['image_grid_thw'].to(j.device);positions,_=rope_positions(j,ids,grids)
    native={'msgs':msgs,'history':history,'head':head,'ids':ids,'grids':grids,'positions':positions,'counts':counts,'rope':rope}
    source=None;windows=[];traces=[];acquisitions=0;diagnostic_seconds=0.
    for i,((a,b),transcript) in enumerate(zip(wins,texts)):
        q=yesno_question(i,V,a,b,transcript,'visual')
        bids,_=j.branch_ids(msgs,q,history,head_text=head)
        base,prior=read_suffix(j,engine,cache,bids,rope,counts);z=base
        if transcript.strip():
            speech_ids,_=j.branch_ids(msgs,yesno_question(i,V,a,b,transcript,'speech'),history,head_text=head)
            j.model.model.rope_deltas=rope.clone();speech=j.cached_margin(cache,speech_ids,in_place=True);cache.crop(n)
        else:speech=None
        main_trace=main['traces'][i];sizes=[len(r['added']) for r in main_trace['rounds']]
        assert main_trace['initial_z']==base
        trace={'window':i,'base_visual':base,'matched':i in matched if arm=='mismatch' else None,'rounds':[]}
        if sizes or arm=='fixed4':
            if source is None:source=FrameSource(row,frames,out/'acquired_frames'/ds/vid)
            all_candidates=source.candidates(a,b)
            if arm=='fixed4':
                count=min(4,len(all_candidates));sizes=[min(2,count)] if count else []
                if count>2:sizes.append(count-2)
            count=sum(sizes)
            if arm in ('uniform','fixed4'):chosen=uniform(all_candidates,a,b,count)
            elif arm=='mismatch':chosen=[e for r in main_trace['rounds'] for e in r['added']]
            else:chosen=[]
            observed=[f[0] for f in frames];acquired=[];offset=0
            for size in sizes:
                if arm=='distance':
                    candidates=source.candidates(a,b,[e['index'] for e in acquired])
                    selected=choose_frames(candidates,observed,prior,size,attention=False)
                    added=[candidates[k] for k in selected]
                else:added=chosen[offset:offset+size]
                assert len(added)==size and size>0
                offset+=size;acquired.extend(added);acquired.sort(key=lambda e:e['time'])
                picture_source=MismatchSource(source,mapping) if arm=='mismatch' else source
                z,prior,meta=read_images(j,engine,cache,native,q,acquired,picture_source)
                observed=[f[0] for f in frames]+[e['time'] for e in acquired];acquisitions+=1
                trace['rounds'].append({'added':added,'acquired':list(acquired),'z':z,
                    'donor_entries':[mapping[e['index']] for e in acquired] if arm=='mismatch' else None,**meta})
        w={'i':i,'start':a,'end':b,'z_visual':z,'z':max(z,speech) if speech is not None else z}
        if speech is not None:w['z_speech']=speech
        windows.append(w);traces.append(trace)
        if smoke:
            t0=tick();j.model.model.rope_deltas=rope.clone()
            replay=j.cached_margin(cache,bids,in_place=True);cache.crop(n);assert replay==base
            diagnostic_seconds+=tick()-t0
        assert cache.get_seq_length()==n and engine.mode is None
    if source is not None:source.close()
    assert j.forward_calls-initial==3+B+acquisitions+(V if smoke else 0)
    elapsed=tick()-started;index=np.clip(((np.arange(math.ceil(duration*FPS))+.5)/FPS//8).astype(int),0,V-1)
    scores=np.asarray([w['z'] for w in windows]);assert np.isfinite(scores).all()
    record={'schema_version':1,'method':'m1_explorer_'+arm,'dataset':ds,'video_id':vid,'duration':duration,
        'native_rate':FPS,'score_curve':scores[index].tolist(),'intervals':[],'error':None,'seed':0,
        'code_path':str(Path(__file__).relative_to(ROOT)),'calls':3+B+acquisitions,
        'extra':{'z_video':zv,'stance':stance,'prefix_tokens':P,'standalone_seconds':elapsed-diagnostic_seconds,
            'n_branches':B,'windows':windows}}
    details={'dataset':ds,'video_id':vid,'traces':traces,'actual_forwards':j.forward_calls-initial,
        'acquisition_reads':acquisitions,'native_replay_exact':True if smoke else None,
        'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'paired_seconds':elapsed,
        'diagnostic_seconds':diagnostic_seconds,'main_trace':f'runs/20261003_m1_explorer/r1_main/details/{ds}/{vid}.json'}
    return record,details


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=('uniform','distance','fixed4','mismatch'),required=True)
    ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    main_root=ROOT/'runs/20261003_m1_explorer'/('r1_smoke' if a.smoke else 'r1_main')
    out=ROOT/'runs/20261003_m1_explorer'/('controls_smoke' if a.smoke else 'controls')/a.arm
    out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    torch.manual_seed(0);rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',['HateMM','HateClipSeg'])
    if a.smoke:rows=[r for ds in ('HateMM','HateClipSeg') for r in [x for x in rows if x['dataset']==ds][:2]]+[
        r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    import transformers
    config={'host':socket.gethostname(),'date':time.strftime('%Y-%m-%d'),'arm':a.arm,'smoke':a.smoke,
        'code':'experiments/20261003_m1_explorer/controls.py + reviewed measure.py/explorer.py; sources2026-10-03',
        'model':MODEL,'torch':torch.__version__,'transformers':transformers.__version__,'seed':0,'GT_in_reader':False,
        'counts':'always up to4' if a.arm=='fixed4' else 'replay actual main per-window round counts',
        'main_trace_root':str(main_root.relative_to(ROOT)),'entropy_regating':False,'new_frames_per_round':2,
        'fps':4,'window_seconds':8,'original_frames':20,'baseline_instrumentation':'same all-layer read-only capture'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    assert not (out/'predictions.jsonl').exists(),'Control output already exists; inspect before resuming'
    asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};j=Judge(MODEL);engine=FrameAttention(j);j.forward_calls=0
    def count(*_):j.forward_calls+=1
    counter=j.model.model.register_forward_pre_hook(count);started=time.time()
    with (out/'predictions.jsonl').open('w') as handle:
        for i,row in enumerate(rows):
            ds,vid=row['dataset'],row['video_id'];source=main_root/'details'/ds/(vid+'.json')
            record,details=read_control(j,engine,row,asr[ds].get(vid,[]),out,a.arm,json.loads(source.read_text()),a.smoke)
            details['main_trace']=str(source.relative_to(ROOT));dest=out/'details'/ds;dest.mkdir(parents=True,exist_ok=True)
            (dest/(vid+'.json')).write_text(json.dumps(details)+'\n');handle.write(json.dumps(record)+'\n');handle.flush()
            logging.info('progress %d/%d %s %s elapsed=%.1f peak_GiB=%.2f acquisitions=%d',i+1,len(rows),ds,vid,time.time()-started,details['peak_GiB'],details['acquisition_reads'])
    counter.remove();engine.close();logging.info('RUN_DONE videos=%d elapsed=%.1f',len(rows),time.time()-started)


if __name__=='__main__':main()
