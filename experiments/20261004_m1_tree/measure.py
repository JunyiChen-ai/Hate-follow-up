#!/usr/bin/env python3
"""Paired native and independently reread tree-conditioned moderation values."""
import argparse
import copy
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
from src.video_inputs import FPS,frame_paths,load_asr,fixed_windows,window_text
from extract import CACHE,DATASETS,selected_rows,validate_cached
from tree import CACHE_VERSION,CONSTANTS,LOCAL_HEADER,tree_text,pixel_check


def tick():torch.cuda.synchronize();return time.perf_counter()


def rope_positions(j,ids,grids):
    kwargs=dict(input_ids=ids,image_grid_thw=grids,attention_mask=torch.ones_like(ids))
    if 'mm_token_type_ids' in inspect.signature(j.model.model.get_rope_index).parameters:
        kwargs['mm_token_type_ids']=(ids==j.image_token_id).to(torch.int32)
    return j.model.model.get_rope_index(**kwargs)


@torch.no_grad()
def context(j,frames,segments,tree=None):
    j.model.model.rope_deltas=None
    msgs,files=j.prefix_messages(frames,segments)
    if tree is not None:
        msgs=copy.deepcopy(msgs);tail=msgs[-1]['content'][-1]['text'];seam='\nBased on this platform'
        assert tail.count(seam)==1
        msgs[-1]['content'][-1]['text']=tail.replace(seam,'\n'+tree_text(tree)+seam,1)
    text,enc=j.encode_prefix(msgs,files);cache=j.prefix_cache(enc)
    qid,qtext=j.branch_ids(msgs,VIDEO_QUESTION);z=j.cached_margin(cache,qid,in_place=True)
    stance='Yes' if z>0 else 'No';aid,atext=j.answer_ids(msgs,VIDEO_QUESTION,stance);j.extend_cache(cache,aid)
    history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',stance)]
    ids=torch.cat((enc['input_ids'].to(j.device),torch.tensor([qid+aid],device=j.device)),1)
    grids=enc['image_grid_thw'].to(j.device);positions,delta=rope_positions(j,ids,grids)
    assert ids.shape[1]==cache.get_seq_length()
    assert torch.equal(delta,j.model.model.rope_deltas)
    return cache,dict(msgs=msgs,files=files,head=text+qtext+atext,history=history,
        ids=ids,grids=grids,positions=positions,rope=delta.clone(),global_margin=z,
        stance=stance,prefix_tokens=enc['input_ids'].shape[1],image_counts=list(j.img_tokens),
        prefix_text=text)


@torch.no_grad()
def standard(j,cache,ctx,question):
    n=cache.get_seq_length();j.model.model.rope_deltas=ctx['rope'].clone()
    ids,_=j.branch_ids(ctx['msgs'],question,ctx['history'],head_text=ctx['head'])
    try:return j.cached_margin(cache,ids,in_place=True)
    finally:cache.crop(n);j.model.model.rope_deltas=ctx['rope'].clone()


@torch.no_grad()
def local_visual(j,cache,ctx,question,packet,metadata,folder,verify=False):
    from PIL import Image
    content=[{'type':'text','text':packet['context']+LOCAL_HEADER}];images=[];paths=[]
    n=cache.get_seq_length()
    for i in packet['pool_members']:
        entry=metadata['entries'][i];path=folder/'witnesses'/f'frame_{entry["index"]:08d}.png'
        with Image.open(path) as image:
            assert image.size==(entry['width'],entry['height']);images.append(image.convert('RGB'))
        paths.append(str(path.relative_to(ROOT)))
        content.extend([{'type':'text','text':f'[t={entry["time"]:.3f}s]\n'},{'type':'image'}])
    if not images:content.append({'type':'text','text':'(none)\n'})
    content.append({'type':'text','text':question})
    full=j.render(ctx['msgs']+ctx['history']+[{'role':'user','content':content}],True)
    assert full.startswith(ctx['head']);suffix=full[len(ctx['head']):]
    try:
        enc=j.encode(suffix,images)
        if images:pixel_check(enc)
        ids=enc['input_ids'].to(j.device);fullids=torch.cat((ctx['ids'],ids),1)
        grids=torch.cat((ctx['grids'],enc['image_grid_thw'].to(j.device)),0) if images else ctx['grids']
        if verify:
            originals=[Image.open(p).convert('RGB') for p in ctx['files']]
            try:fresh=j.encode(full,originals+images)
            finally:
                for im in originals:im.close()
            assert torch.equal(fresh['input_ids'].to(j.device),fullids)
            assert torch.equal(fresh['image_grid_thw'].to(j.device),grids)
        positions,delta=rope_positions(j,fullids,grids)
        assert torch.equal(positions[:,:,:n],ctx['positions']), 'prefix multimodal positions changed'
        kw=j.model_inputs(enc);kw.update(position_ids=positions[:,:,n:],
            attention_mask=torch.ones_like(fullids),past_key_values=cache,use_cache=True)
        if 'cache_position' in j.forward_params:kw['cache_position']=torch.arange(n,fullids.shape[1],device=j.device)
        j.model.model.rope_deltas=delta
        try:
            out=j.model.model(**kw);hidden=out.last_hidden_state[0,-1];del out
        finally:cache.crop(n);j.model.model.rope_deltas=ctx['rope'].clone()
        z=j.margins_fp32(hidden[None])[0]
    finally:
        for im in images:im.close()
    return z,dict(packet=packet,paths=paths,suffix_tokens=len(ids[0]),
        new_image_counts=(enc['image_grid_thw'].prod(1)//j.processor.image_processor.merge_size**2).tolist() if images else [],
        fresh_render_verified=bool(verify),prefix_positions_exact=True,margin=z)


def record(row,ctx,visual,speech,seconds):
    wins=fixed_windows(float(row['duration']),8);windows=[]
    for i,((a,b),v,s) in enumerate(zip(wins,visual,speech)):
        w=dict(i=i,start=a,end=b,z_visual=v,z=max(v,s) if s is not None else v)
        if s is not None:w['z_speech']=s
        windows.append(w)
    duration=float(row['duration']);idx=np.clip(((np.arange(math.ceil(duration*FPS))+.5)/FPS//8).astype(int),0,len(wins)-1)
    return dict(schema_version=1,method='m1_tree',dataset=row['dataset'],video_id=row['video_id'],duration=duration,
        native_rate=FPS,score_curve=np.asarray([w['z'] for w in windows])[idx].tolist(),intervals=[],error=None,seed=0,
        calls=3+len(wins)+sum(s is not None for s in speech),code_path=str(Path(__file__).relative_to(ROOT)),
        extra=dict(z_video=ctx['global_margin'],stance=ctx['stance'],prefix_tokens=ctx['prefix_tokens'],
            standalone_seconds=seconds,n_branches=len(wins)+sum(s is not None for s in speech),windows=windows))


@torch.no_grad()
def read_video(j,row,segments,metadata,folder,smoke):
    frames=frame_paths(row['dataset'],row['video_id'],20);assert 0<len(frames)<=20
    wins=fixed_windows(float(row['duration']),8);texts=[window_text(segments,a,b) for a,b in wins]
    first=j.forward_calls;torch.cuda.reset_peak_memory_stats();t0=tick()
    cache,base_ctx=context(j,frames,segments);vs=[];ss=[]
    for i,((a,b),body) in enumerate(zip(wins,texts)):
        vs.append(standard(j,cache,base_ctx,yesno_question(i,len(wins),a,b,body,'visual')))
        ss.append(standard(j,cache,base_ctx,yesno_question(i,len(wins),a,b,body,'speech')) if body.strip() else None)
    native_seconds=tick()-t0;del cache
    t1=tick();cache,new_ctx=context(j,frames,segments,metadata['tree']);nv=[];ns=[];traces=[];diagnostic_seconds=0.
    for i,((a,b),body,packet) in enumerate(zip(wins,texts,metadata['packets'])):
        question=yesno_question(i,len(wins),a,b,body,'visual');verify=smoke and i==0
        z,trace=local_visual(j,cache,new_ctx,question,packet,metadata,folder,verify)
        nv.append(z);traces.append(trace)
        ns.append(standard(j,cache,new_ctx,yesno_question(i,len(wins),a,b,body,'speech')) if body.strip() else None)
    read_seconds=tick()-t1;del cache
    base=record(row,base_ctx,vs,ss,native_seconds)
    optimized=record(row,new_ctx,nv,ns,metadata['standalone_seconds']+read_seconds)
    assert j.forward_calls-first==base['calls']+optimized['calls']
    check=dict(dataset=row['dataset'],video_id=row['video_id'],GT_read=False,actual_forwards=j.forward_calls-first,
        standalone_seconds=dict(base=native_seconds,optimized=metadata['standalone_seconds']+read_seconds),
        read_seconds=read_seconds,preprocessing_seconds=metadata['standalone_seconds'],
        peak_GiB=max(torch.cuda.max_memory_allocated()/2**30,metadata['peak_GiB']),
        input_actual_forwards=metadata['actual_forwards'],caption_count=metadata['caption_count'],caption_tokens=metadata['caption_tokens'],
        added_local_images=sum(len(p['pool_members']) for p in metadata['packets']),
        feature_images=len(metadata['entries']),new_prefix_tokens=new_ctx['prefix_tokens'])
    detail=dict(dataset=row['dataset'],video_id=row['video_id'],traces=traces,
        native_frame_times=[t for t,p in frames],tree_global_text=tree_text(metadata['tree']),
        segments=[list(s) for s in segments],base_prefix_tokens=base_ctx['prefix_tokens'],
        new_prefix_tokens=new_ctx['prefix_tokens'],cache_version=CACHE_VERSION)
    return base,optimized,check,detail


@torch.no_grad()
def read_visual_only(j,row,segments,metadata,folder,smoke):
    """R2: one native stance cache; new local pixels change visual reads only."""
    frames=frame_paths(row['dataset'],row['video_id'],20);assert 0<len(frames)<=20
    wins=fixed_windows(float(row['duration']),8);texts=[window_text(segments,a,b) for a,b in wins]
    first=j.forward_calls;torch.cuda.reset_peak_memory_stats();t=tick()
    cache,ctx=context(j,frames,segments);prefix_seconds=tick()-t
    vs=[];ss=[];nv=[];traces=[];visual_seconds=speech_seconds=new_visual_seconds=diagnostic_seconds=0.
    for i,((a,b),body,packet) in enumerate(zip(wins,texts,metadata['packets'])):
        visual_question=yesno_question(i,len(wins),a,b,body,'visual')
        t=tick();vs.append(standard(j,cache,ctx,visual_question));visual_seconds+=tick()-t
        t=tick();ss.append(standard(j,cache,ctx,yesno_question(i,len(wins),a,b,body,'speech')) if body.strip() else None)
        speech_seconds+=tick()-t
        t=tick();z,trace=local_visual(j,cache,ctx,visual_question,packet,metadata,folder,smoke and i==0)
        new_visual_seconds+=tick()-t;nv.append(z)
        if smoke and i==0:
            t=tick();cloned=copy.deepcopy(cache)
            replay,_=local_visual(j,cloned,ctx,visual_question,packet,metadata,folder)
            assert replay==z,'cloned visual branch differs from cropped native cache'
            del cloned;diagnostic_seconds+=tick()-t;trace['cloned_margin']=replay
        traces.append(trace)
    del cache
    native_seconds=prefix_seconds+visual_seconds+speech_seconds
    read_seconds=prefix_seconds+speech_seconds+new_visual_seconds
    base=record(row,ctx,vs,ss,native_seconds)
    optimized=record(row,ctx,nv,ss,metadata['standalone_seconds']+read_seconds)
    diagnostics=int(smoke)
    assert j.forward_calls-first==base['calls']+len(wins)+diagnostics
    check=dict(dataset=row['dataset'],video_id=row['video_id'],GT_read=False,reader_revision='r2',
        actual_forwards=j.forward_calls-first,diagnostic_forwards=diagnostics,diagnostic_seconds=diagnostic_seconds,
        standalone_seconds=dict(base=native_seconds,optimized=metadata['standalone_seconds']+read_seconds),
        prefix_seconds=prefix_seconds,native_visual_seconds=visual_seconds,shared_speech_seconds=speech_seconds,
        new_visual_seconds=new_visual_seconds,read_seconds=read_seconds,preprocessing_seconds=metadata['standalone_seconds'],
        peak_GiB=max(torch.cuda.max_memory_allocated()/2**30,metadata['peak_GiB']),
        input_actual_forwards=metadata['actual_forwards'],caption_count=metadata['caption_count'],caption_tokens=metadata['caption_tokens'],
        added_local_images=sum(len(p['pool_members']) for p in metadata['packets']),feature_images=len(metadata['entries']),
        new_prefix_tokens=ctx['prefix_tokens'])
    detail=dict(dataset=row['dataset'],video_id=row['video_id'],reader_revision='r2',traces=traces,
        global_context='native',speech_context='native',native_frame_times=[t for t,p in frames],
        tree_global_text=tree_text(metadata['tree']),tree_in_global_prefix=False,
        segments=[list(s) for s in segments],base_prefix_tokens=ctx['prefix_tokens'],
        new_prefix_tokens=ctx['prefix_tokens'],cache_version=CACHE_VERSION)
    return base,optimized,check,detail


def validate_records(row,base,new,check,detail,metadata):
    wins=fixed_windows(float(row['duration']),8)
    for r in (base,new):
        assert (r['dataset'],r['video_id'])==(row['dataset'],row['video_id'])
        assert r['duration']==float(row['duration']) and r['native_rate']==FPS and r['error'] is None
        assert r['extra']['stance']==('Yes' if r['extra']['z_video']>0 else 'No')
        assert np.isfinite(r['extra']['z_video']) and len(r['extra']['windows'])==len(wins)
        for i,(w,(a,b)) in enumerate(zip(r['extra']['windows'],wins)):
            assert w['i']==i and w['start']==a and w['end']==b
            assert w['z']==max(w['z_visual'],w.get('z_speech',float('-inf')))
        idx=np.clip(((np.arange(math.ceil(r['duration']*FPS))+.5)/FPS//8).astype(int),0,len(wins)-1)
        assert np.array_equal(r['score_curve'],np.asarray([w['z'] for w in r['extra']['windows']])[idx])
        assert np.isfinite(r['score_curve']).all()
        assert r['calls']==3+len(wins)+sum('z_speech' in w for w in r['extra']['windows'])
    if detail.get('reader_revision')=='r2':
        assert check['reader_revision']=='r2'
        assert check['actual_forwards']==base['calls']+len(wins)+check['diagnostic_forwards']
        assert check['diagnostic_forwards']==sum('cloned_margin' in t for t in detail['traces'])
        assert base['extra']['z_video']==new['extra']['z_video']
        assert base['extra']['prefix_tokens']==new['extra']['prefix_tokens']==detail['base_prefix_tokens']==detail['new_prefix_tokens']
        assert detail['global_context']==detail['speech_context']=='native' and detail['tree_in_global_prefix'] is False
        assert all(a.get('z_speech')==b.get('z_speech') for a,b in zip(base['extra']['windows'],new['extra']['windows']))
        assert check['read_seconds']==check['prefix_seconds']+check['shared_speech_seconds']+check['new_visual_seconds']
        assert check['standalone_seconds']['base']==check['prefix_seconds']+check['native_visual_seconds']+check['shared_speech_seconds']
        assert check['standalone_seconds']['optimized']==check['preprocessing_seconds']+check['read_seconds']
    else:assert check['actual_forwards']==base['calls']+new['calls']
    assert detail['cache_version']==CACHE_VERSION and detail['tree_global_text']==tree_text(metadata['tree'])
    assert len(detail['traces'])==len(wins)
    for trace,packet,w in zip(detail['traces'],metadata['packets'],new['extra']['windows']):
        assert trace['packet']==packet and trace['margin']==w['z_visual']
        assert trace['prefix_positions_exact'] is True
        assert len(trace['paths'])==len(trace['new_image_counts'])==len(packet['pool_members'])
        if 'cloned_margin' in trace:assert trace['cloned_margin']==trace['margin']
    assert check['added_local_images']==sum(len(p['pool_members']) for p in metadata['packets'])


def existing(path,expected):
    rows={}
    if path.exists():
        for line in path.open():
            r=json.loads(line);key=r['dataset'],r['video_id'];assert key in expected and key not in rows
            rows[key]=r
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--revision',choices=('r1','r2'),default='r1');a=ap.parse_args()
    out=ROOT/'runs/20261004_m1_tree'/(a.revision+'_full_'+('smoke' if a.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    config=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,cache_version=CACHE_VERSION,
        constants=CONSTANTS,GT_in_reader=False,smoke=a.smoke,seed=0,frames=20,window_seconds=8,fps=FPS,
        torch=torch.__version__,transformers=transformers.__version__,
        code='experiments/20261004_m1_tree/{tree,extract,measure}.py + src/{video_inputs,mllm_judge}.py; sources2026-10-04')
    if a.revision=='r2':config.update(reader_revision='r2',global_context='native',speech_context='native',
        code='experiments/20261004_m1_tree/{tree,extract,measure}.py + src/{video_inputs,mllm_judge}.py; native visual-only revision2026-10-05')
    cp=out/'config.json'
    if cp.exists():
        old=json.loads(cp.read_text());assert {k:v for k,v in old.items() if k!='date'}=={k:v for k,v in config.items() if k!='date'}
    else:cp.write_text(json.dumps(config,indent=2)+'\n')
    rows=selected_rows(a.smoke);expected={(r['dataset'],r['video_id']):r for r in rows}
    done={name:existing(out/name/'predictions.jsonl',expected) for name in ('base','optimized')}
    checks=existing(out/'checks.jsonl',expected);assert done['base'].keys()==done['optimized'].keys()==checks.keys()
    asr={ds:load_asr(ds) for ds in DATASETS}
    for key in checks:
        meta,features=validate_cached(CACHE/key[0]/key[1],expected[key])
        detail=json.loads((out/'details'/key[0]/(key[1]+'.json')).read_text())
        assert detail.get('reader_revision','r1')==a.revision
        assert detail['segments']==[list(s) for s in asr[key[0]].get(key[1],[])]
        validate_records(expected[key],done['base'][key],done['optimized'][key],checks[key],detail,meta)
    torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=0
    hook=j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1))
    handles={}
    for name in ('base','optimized'):
        d=out/name;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**config,'measurement':name},indent=2)+'\n')
        handles[name]=(d/'predictions.jsonl').open('a',buffering=1)
    ch=(out/'checks.jsonl').open('a',buffering=1)
    for i,row in enumerate(rows,1):
        key=row['dataset'],row['video_id']
        if key in checks:logging.info('%d/%d reuse %s/%s',i,len(rows),*key);continue
        folder=CACHE/key[0]/key[1];meta,features=validate_cached(folder,row);begin=time.perf_counter()
        reader=read_visual_only if a.revision=='r2' else read_video
        base,new,check,detail=reader(j,row,asr[key[0]].get(key[1],[]),meta,folder,a.smoke)
        validate_records(row,base,new,check,detail,meta)
        detaildir=out/'details'/key[0];detaildir.mkdir(parents=True,exist_ok=True)
        (detaildir/(key[1]+'.json')).write_text(json.dumps(detail)+'\n')
        for name,r in [('base',base),('optimized',new)]:handles[name].write(json.dumps(r)+'\n')
        ch.write(json.dumps(check)+'\n');logging.info('%d/%d %s/%s %.2fs',i,len(rows),*key,time.perf_counter()-begin)
    for h in handles.values():h.close()
    ch.close();hook.remove();logging.info('DONE coverage=%d',len(rows))


if __name__=='__main__':main()
