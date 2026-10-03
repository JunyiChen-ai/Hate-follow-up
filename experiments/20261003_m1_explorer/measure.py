#!/usr/bin/env python3
"""Native paired reads and bounded raw-frame acquisition; no ground-truth inputs."""
import argparse
import inspect
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
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.mllm_judge import Judge,MODEL,VIDEO_QUESTION,yesno_question
from src.video_inputs import FPS,frame_paths,load_asr,load_manifest,fixed_windows,window_text
from explorer import FrameAttention,FrameSource,binary_entropy,choose_frames


def tick():
    torch.cuda.synchronize();return time.perf_counter()


def rope_positions(j,ids,grids):
    model=j.model.model
    kwargs={'input_ids':ids,'image_grid_thw':grids,'attention_mask':torch.ones_like(ids)}
    if 'mm_token_type_ids' in inspect.signature(model.get_rope_index).parameters:
        kwargs['mm_token_type_ids']=(ids==j.image_token_id).to(torch.int32)
    return model.get_rope_index(**kwargs)


@torch.no_grad()
def read_suffix(j,engine,cache,ids,rope,counts):
    n=cache.get_seq_length();j.model.model.rope_deltas=rope.clone()
    engine.start('query',torch.zeros(len(ids),dtype=torch.bool))
    try:h=j._step(cache,ids)
    finally:engine.stop();cache.crop(n)
    z=j.margins_fp32(h[None])[0]
    return z,engine.prior(counts)


@torch.no_grad()
def read_images(j,engine,cache,native,question,entries,source):
    n=cache.get_seq_length();imgs=[];paths=[];content=[]
    for e in sorted(entries,key=lambda e:e['time']):
        image,path=source.image(e);imgs.append(image);paths.append(str(path.relative_to(ROOT)))
        content.extend([{'type':'text','text':f'[t={e["time"]:.3f}s]\n'},{'type':'image'}])
    content.append({'type':'text','text':question})
    full=j.render(native['msgs']+native['history']+[{'role':'user','content':content}],True)
    assert full.startswith(native['head'])
    suffix=full[len(native['head']):]
    try:enc=j.encode(suffix,imgs)
    finally:
        for im in imgs:im.close()
    newids=enc['input_ids'].to(j.device)
    fullids=torch.cat((native['ids'],newids),1)
    grids=torch.cat((native['grids'],enc['image_grid_thw'].to(j.device)),0)
    pos,delta=rope_positions(j,fullids,grids)
    assert torch.equal(pos[:,:,:n],native['positions']), 'Cached-prefix positions changed'
    counts=(enc['image_grid_thw'].prod(1)//j.processor.image_processor.merge_size**2).tolist()
    assert len(counts)==len(entries) and int((newids==j.image_token_id).sum())==sum(counts)
    kw=j.model_inputs(enc)
    kw.update(position_ids=pos[:,:,n:],attention_mask=torch.ones_like(fullids),past_key_values=cache,use_cache=True)
    if 'cache_position' in j.forward_params:kw['cache_position']=torch.arange(n,len(fullids[0]),device=j.device)
    j.model.model.rope_deltas=delta
    engine.start('query',newids[0]==j.image_token_id)
    try:
        out=j.model.model(**kw);h=out.last_hidden_state[0,-1];del out
    finally:
        engine.stop();cache.crop(n);j.model.model.rope_deltas=native['rope'].clone()
    z=j.margins_fp32(h[None])[0]
    prior=engine.prior(native['counts']+counts)
    return z,prior,{'paths':paths,'image_counts':counts,'image_grid_thw':enc['image_grid_thw'].tolist(),
        'suffix_ids':newids[0].cpu().tolist(),'suffix_positions':pos[:,:,n:].cpu().tolist()}


@torch.no_grad()
def read_video(j,engine,row,segments,out,smoke=False,support_override=True):
    ds,vid,duration=row['dataset'],row['video_id'],float(row['duration'])
    frames=frame_paths(ds,vid,20,'k20');assert 0<len(frames)<=20
    wins=fixed_windows(duration,8);texts=[window_text(segments,a,b) for a,b in wins]
    available=[bool(t.strip()) for t in texts];V=len(wins);B=V+sum(available)
    j.model.model.rope_deltas=None;torch.cuda.reset_peak_memory_stats();initial=j.forward_calls
    started=tick();msgs,files=j.prefix_messages(frames,segments);text,enc=j.encode_prefix(msgs,files)
    counts=list(j.img_tokens)
    engine.start('prefix',enc['input_ids'][0]==j.image_token_id)
    try:cache=j.prefix_cache(enc)
    finally:engine.stop()
    P=cache.get_seq_length();qid,qtext=j.branch_ids(msgs,VIDEO_QUESTION)
    zv=j.cached_margin(cache,qid,in_place=True);stance='Yes' if zv>0 else 'No'
    aid,atext=j.answer_ids(msgs,VIDEO_QUESTION,stance);j.extend_cache(cache,aid)
    history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',stance)]
    head=text+qtext+atext;n=cache.get_seq_length();rope=j.model.model.rope_deltas.clone()
    ids=torch.cat((enc['input_ids'].to(j.device),torch.tensor([qid+aid],device=j.device)),1)
    grids=enc['image_grid_thw'].to(j.device);positions,_=rope_positions(j,ids,grids)
    assert ids.shape[1]==n
    native={'msgs':msgs,'history':history,'head':head,'ids':ids,'grids':grids,
        'positions':positions,'counts':counts,'rope':rope}
    prefix_time=tick()-started
    vs=[];ss=[];ev=[];traces=[];base_visual_seconds=speech_seconds=extra_seconds=diagnostic_seconds=0.
    source=None;source_seconds=0.
    for i,((a,b),t) in enumerate(zip(wins,texts)):
        q= yesno_question(i,V,a,b,t,'visual')
        bids,_=j.branch_ids(msgs,q,history,head_text=head)
        t0=tick();z,prior=read_suffix(j,engine,cache,bids,rope,counts);base_visual_seconds+=tick()-t0
        vs.append(z)
        if available[i]:
            bids_s,_=j.branch_ids(msgs,yesno_question(i,V,a,b,t,'speech'),history,head_text=head)
            t0=tick();j.model.model.rope_deltas=rope.clone()
            s=j.cached_margin(cache,bids_s,in_place=True);cache.crop(n);speech_seconds+=tick()-t0
        else:s=None
        ss.append(s)
        support=sum(a<=ft<b for ft,_ in frames)
        trace={'window':i,'nominal_support':support,'initial_z':z,'initial_entropy':binary_entropy(z),
            'rounds':[],'initial_prior':prior.tolist()}
        acquired=[];observed=[ft for ft,_ in frames]
        for round_index in range(2):
            threshold=(.1,.3)[round_index]
            if (round_index>0 or support>0 or not support_override) and binary_entropy(z)<threshold:break
            t0=tick()
            if source is None:
                ts=tick();source=FrameSource(row,frames,out/'acquired_frames'/ds/vid);source_seconds+=tick()-ts
            candidates=source.candidates(a,b,[e['index'] for e in acquired])
            if not candidates:trace['candidate_exhausted']=True;extra_seconds+=tick()-t0;break
            selected=choose_frames(candidates,observed,prior,2)
            added=[candidates[k] for k in selected];acquired.extend(added)
            acquired.sort(key=lambda e:e['time'])
            z,prior,meta=read_images(j,engine,cache,native,q,acquired,source)
            observed=[ft for ft,_ in frames]+[e['time'] for e in acquired]
            trace['rounds'].append({'added':added,'acquired':list(acquired),'z':z,'entropy':binary_entropy(z),
                'prior':prior.tolist(),**meta})
            extra_seconds+=tick()-t0
        ev.append(z);trace['final_z']=z;traces.append(trace)
        if smoke:
            t0=tick();j.model.model.rope_deltas=rope.clone()
            replay=j.cached_margin(cache,bids,in_place=True);cache.crop(n)
            assert replay==vs[-1], 'Acquisition contaminated native replay'
            diagnostic_seconds+=tick()-t0
        assert cache.get_seq_length()==n and engine.mode is None
    if source is not None:
        source_metadata={'path':str(source.path),'presentation_origin':source.origin,'frames':source.entries,
            'legacy_excluded_indices':sorted(source.legacy_excluded),'legacy_exclusion_is_conservative':True}
        source.close()
    else:source_metadata=None
    acquisitions=sum(len(t['rounds']) for t in traces)
    assert j.forward_calls-initial==3+B+acquisitions+(V if smoke else 0)
    standalone={'base':prefix_time+base_visual_seconds+speech_seconds,
        'explore':prefix_time+base_visual_seconds+speech_seconds+extra_seconds}
    index=np.clip(((np.arange(math.ceil(duration*FPS))+.5)/FPS//8).astype(int),0,V-1)
    records={}
    for arm,values in (('base',vs),('explore',ev)):
        windows=[]
        for i,((a,b),v,s) in enumerate(zip(wins,values,ss)):
            w={'i':i,'start':a,'end':b,'z_visual':v,'z':max(v,s) if s is not None else v}
            if s is not None:w['z_speech']=s
            windows.append(w)
        scores=np.asarray([w['z'] for w in windows]);assert np.isfinite(scores).all()
        records[arm]={'schema_version':1,'method':'m1_explorer_'+arm,'dataset':ds,'video_id':vid,
            'duration':duration,'native_rate':FPS,'score_curve':scores[index].tolist(),'intervals':[],
            'error':None,'seed':0,'code_path':str(Path(__file__).relative_to(ROOT)),
            'calls':3+B+(acquisitions if arm=='explore' else 0),
            'extra':{'z_video':zv,'stance':stance,'prefix_tokens':P,'standalone_seconds':standalone[arm],
                'n_branches':B,'windows':windows}}
    check={'dataset':ds,'video_id':vid,'native_branches':B,'visual_queries':V,'acquisition_reads':acquisitions,
        'actual_forwards':j.forward_calls-initial,'prefix_tokens':P,'actual_original_frames':len(frames),
        'native_replay_exact':True if smoke else None,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,
        'standalone_seconds':standalone,'source_index_seconds':source_seconds,'diagnostic_seconds':diagnostic_seconds,
        'baseline_instrumented':True,'paired_seconds':tick()-started}
    details={'traces':traces,'source':source_metadata,'image_counts':counts,
        'original_frame_times':[ft for ft,_ in frames],'input_ids':enc['input_ids'][0].tolist()}
    del cache
    return records,check,details


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--version',choices=('r1','r2'),default='r1');a=ap.parse_args()
    out=ROOT/'runs/20261003_m1_explorer'/(a.version+('_smoke' if a.smoke else '_main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    torch.manual_seed(0)
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',['HateMM','HateClipSeg'])
    if a.smoke:
        rows=[r for ds in ('HateMM','HateClipSeg') for r in [x for x in rows if x['dataset']==ds][:2]]+[
            r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};j=Judge(MODEL);engine=FrameAttention(j)
    j.forward_calls=0
    def count(*_):j.forward_calls+=1
    counter=j.model.model.register_forward_pre_hook(count)
    import transformers
    config={'date':time.strftime('%Y-%m-%d'),'host':socket.gethostname(),'smoke':a.smoke,'seed':0,'model':MODEL,
        'torch':torch.__version__,'transformers':transformers.__version__,'GT_in_reader':False,
        'code':'experiments/20261003_m1_explorer/{measure,explorer}.py + src/{video_inputs,mllm_judge}.py; sources2026-10-03',
        'entropy_thresholds':[.1,.3],'frames_per_round':2,'round_limit':2,'attention_exponent':.5,
        'revision':a.version,'support_override':a.version=='r1',
        'layers':list(range(36)),'scope':'pre-RoPE normalized Q/K, image-only softmax',
        'frames':20,'window_seconds':8,'fps':FPS,'video_question':VIDEO_QUESTION,
        'global_and_answer':'original native','speech':'original native',
        'timing_note':'baseline includes read-only captures/priors; all acquisition/decoding costs charged to explore'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');handles={};done={}
    for arm in ('base','explore'):
        d=out/arm;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**config,'arm':arm},indent=2)+'\n')
        p=d/'predictions.jsonl';done[arm]={(r['dataset'],r['video_id']) for r in map(json.loads,p.open())} if p.exists() else set()
        handles[arm]=p.open('a')
    cp=out/'checks.jsonl';seen={(r['dataset'],r['video_id']) for r in map(json.loads,cp.open())} if cp.exists() else set()
    assert done['base']==done['explore']==seen,'incomplete paired record'
    checked=cp.open('a');started=time.time()
    for k,row in enumerate(rows):
        key=row['dataset'],row['video_id']
        if key in seen:continue
        recs,check,details=read_video(j,engine,row,asr[key[0]].get(key[1],[]),out,a.smoke,support_override=a.version=='r1')
        dest=out/'details'/key[0];dest.mkdir(parents=True,exist_ok=True)
        (dest/(key[1]+'.json')).write_text(json.dumps(details)+'\n')
        for arm,r in recs.items():handles[arm].write(json.dumps(r)+'\n');handles[arm].flush()
        checked.write(json.dumps(check)+'\n');checked.flush()
        logging.info('progress %d/%d %s %s elapsed=%.1f peak_GiB=%.2f acquisitions=%d',k+1,len(rows),*key,time.time()-started,check['peak_GiB'],check['acquisition_reads'])
    for f in handles.values():f.close()
    checked.close();counter.remove();engine.close();logging.info('RUN_DONE videos=%d elapsed=%.1f',len(rows),time.time()-started)


if __name__=='__main__':main()
