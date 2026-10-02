#!/usr/bin/env python3
"""Paired paper-defined ACG transfer collection, no GT or annotations in scoring."""
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
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.mllm_judge import Judge,MODEL,VIDEO_QUESTION,yesno_question
from src.video_inputs import FPS,frame_paths,load_asr,load_manifest,fixed_windows,window_text
from projector import AttentionProjector


def synced_time():
    torch.cuda.synchronize();return time.perf_counter()


def prefix(j,frames,segments,forced=None,with_frames=True):
    msgs,files=j.prefix_messages(frames,segments,with_frames=with_frames)
    text,enc=j.encode_prefix(msgs,files)
    if hasattr(j.model.model,'rope_deltas'):j.model.model.rope_deltas=None
    cache=j.prefix_cache(enc);P=cache.get_seq_length()
    ids,btext=j.branch_ids(msgs,VIDEO_QUESTION)
    global_z=j.cached_margin(cache,ids,in_place=True)
    stance=forced if forced is not None else ('Yes' if global_z>0 else 'No')
    ids,atext=j.answer_ids(msgs,VIDEO_QUESTION,stance);j.extend_cache(cache,ids)
    history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',stance)]
    head=text+btext+atext
    return cache,msgs,head,history,global_z,stance,enc,P


def queries(j,msgs,head,history,wins,texts,kind):
    return [j.branch_ids(msgs,yesno_question(i,len(wins),a,b,txt,kind),history,head_text=head)[0]
        for i,((a,b),txt) in enumerate(zip(wins,texts))]


def native_reads(j,cache,ids,available):
    n=cache.get_seq_length();values=[]
    for tokens,use in zip(ids,available):
        values.append(j.cached_margin(cache,tokens,in_place=True) if use else None)
        cache.crop(n)
    return values


def read_video(j,engine,row,segments,smoke=False):
    ds,vid,duration=row['dataset'],row['video_id'],float(row['duration'])
    frames=frame_paths(ds,vid,20,'k20');assert frames
    wins=fixed_windows(duration,8);texts=[window_text(segments,a,b) for a,b in wins]
    V=len(wins);available=[bool(t.strip()) for t in texts];B=V+sum(available)
    initial=j.forward_calls;torch.cuda.reset_peak_memory_stats();tick=synced_time()
    cache,msgs,head,history,zv,stance,enc,P=prefix(j,frames,segments)
    visual=(enc['input_ids'][0]==j.image_token_id).numpy()
    assert visual.sum()==sum(j.img_tokens) and len(j.img_tokens)==len(frames)
    assert torch.all(enc['attention_mask']==1)
    prefix_seconds=synced_time()-tick;n=cache.get_seq_length();rope=j.model.model.rope_deltas.clone()
    vis_ids=queries(j,msgs,head,history,wins,texts,'visual')
    speech_ids=queries(j,msgs,head,history,wins,texts,'speech')
    tick=synced_time();bv=native_reads(j,cache,vis_ids,[True]*V);native_visual_seconds=synced_time()-tick
    tick=synced_time();speech=native_reads(j,cache,speech_ids,available);speech_seconds=synced_time()-tick
    vectors={};vs={'base':bv};seconds={};geometry={}
    for arm,gamma in [('eager',0.),('project',1.4)]:
        tick=synced_time();lv=[];gv=[];values=[]
        for ids in vis_ids:
            lg,diag=engine.logits(cache,ids,visual,gamma)
            assert torch.isfinite(lg).all();lv.append(lg);gv.append(diag['geometry'])
            values.append(float(lg[:len(j.yes_ids)].logsumexp(0)-lg[len(j.yes_ids):].logsumexp(0)))
        seconds[arm]=synced_time()-tick;vectors[arm]=torch.stack(lv).numpy();geometry[arm]=gv;vs[arm]=values
    assert cache.get_seq_length()==n and engine.active is None and torch.equal(j.model.model.rope_deltas,rope)
    del cache
    verify={};diagnostic_seconds=0.
    if smoke:
        tick=synced_time();rc,rm,rh,rhi,rz,rs,re,rP=prefix(j,frames,segments)
        rv=native_reads(j,rc,queries(j,rm,rh,rhi,wins,texts,'visual'),[True]*V)
        rsp=native_reads(j,rc,queries(j,rm,rh,rhi,wins,texts,'speech'),available)
        assert rz==zv and rs==stance and rv==bv and rsp==speech
        assert all(torch.equal(enc[k],re[k]) for k in enc if torch.is_tensor(enc[k]))
        assert rP==P;del rc;diagnostic_seconds=synced_time()-tick
        verify={'native_global_exact':True,'native_windows_exact':True,'inputs_exact':True,'extra_forwards':3+B}
    actual=j.forward_calls-initial;assert actual==3+B+2*V+verify.get('extra_forwards',0)
    length=int(math.ceil(duration*FPS));index=np.clip(((np.arange(length)+.5)/FPS//8).astype(int),0,V-1)
    records={};standalone={a:prefix_seconds+speech_seconds+(native_visual_seconds if a=='base' else seconds[a]) for a in vs}
    for arm,values in vs.items():
        windows=[]
        for i,((a,b),v,s) in enumerate(zip(wins,values,speech)):
            w={'i':i,'start':a,'end':b,'z_visual':v,'z':max(v,s) if s is not None else v}
            if s is not None:w['z_speech']=s
            windows.append(w)
        scores=np.asarray([w['z'] for w in windows]);assert np.isfinite(scores).all()
        records[arm]={'schema_version':1,'method':'m1_projector_'+arm,'dataset':ds,'video_id':vid,
            'duration':duration,'native_rate':FPS,'score_curve':scores[index].tolist(),'intervals':[],'error':None,
            'seed':0,'code_path':str(Path(__file__).relative_to(ROOT)),'calls':3+B,
            'extra':{'z_video':zv,'stance':stance,'prefix_tokens':P,'standalone_seconds':standalone[arm],
                'n_branches':B,'windows':windows}}
    check={'dataset':ds,'video_id':vid,'native_global':zv,'forced_answer':stance,
        'prefix_tokens':P,'visual_tokens':int(visual.sum()),'visual_queries':V,'native_branches':B,
        'actual_forwards':actual,'verify':verify,'diagnostic_seconds':diagnostic_seconds,
        'peak_GiB':torch.cuda.max_memory_allocated()/2**30,'layer_indices':engine.layer_indices,
        'geometry_fields':['ordinary_norm','reference_norm','delta_norm','perpendicular_norm','abs_projection_max'],
        'geometry':geometry,'seconds':{'prefix':prefix_seconds,'native_visual':native_visual_seconds,
            'speech':speech_seconds,**seconds},'paired_seconds':prefix_seconds+speech_seconds+native_visual_seconds+sum(seconds.values())}
    tokens={'yes_ids':np.asarray(j.yes_ids),'no_ids':np.asarray(j.no_ids),**vectors,
        'input_ids':enc['input_ids'][0].numpy(),'visual':visual}
    return records,check,tokens


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261003_m1_projector'/('r1_smoke' if a.smoke else 'r1_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    torch.manual_seed(0)
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',['HateMM','HateClipSeg'])
    if a.smoke:
        rows=[r for ds in ('HateMM','HateClipSeg') for r in [v for v in rows if v['dataset']==ds][:2]]+[
            r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
        assert len(rows)==5
    asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};j=Judge(MODEL);engine=AttentionProjector(j)
    j.forward_calls=0
    def count(*_):j.forward_calls+=1
    counter=j.model.model.register_forward_pre_hook(count)
    import transformers
    config={'date':time.strftime('%Y-%m-%d'),'host':socket.gethostname(),'smoke':a.smoke,'seed':0,
        'model':MODEL,'torch':torch.__version__,'transformers':transformers.__version__,'GT_in_reader':False,
        'code':'experiments/20261003_m1_projector/{measure,projector}.py + src/{video_inputs,mllm_judge}.py; sources2026-10-03',
        'gamma':1.4,'epsilon':1e-8,'layers':engine.layer_indices,
        'intervention':'per-head orthogonal attention-output guidance, last visual-query row only',
        'frames':20,'window_seconds':8,'fps':FPS,'video_question':VIDEO_QUESTION,
        'global_and_answer':'original native','speech':'original native',
        'timing_note':'all attention reductions and diagnostic arithmetic included' }
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');handles={};done={}
    for arm in ('base','eager','project'):
        d=out/arm;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**config,'arm':arm},indent=2)+'\n')
        p=d/'predictions.jsonl';done[arm]={(r['dataset'],r['video_id']) for r in map(json.loads,p.open())} if p.exists() else set()
        handles[arm]=p.open('a')
    cp=out/'checks.jsonl';seen={(r['dataset'],r['video_id']) for r in map(json.loads,cp.open())} if cp.exists() else set()
    assert done['base']==done['eager']==done['project']==seen,'incomplete paired record'
    checked=cp.open('a');started=time.time()
    for k,row in enumerate(rows):
        key=row['dataset'],row['video_id']
        if key in seen:continue
        recs,check,tokens=read_video(j,engine,row,asr[key[0]].get(key[1],[]),a.smoke)
        dest=out/'tokens'/key[0];dest.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest/(key[1]+'.npz'),**tokens)
        for arm,r in recs.items():handles[arm].write(json.dumps(r)+'\n');handles[arm].flush()
        checked.write(json.dumps(check)+'\n');checked.flush()
        logging.info('progress %d/%d %s %s elapsed=%.1f peak_GiB=%.2f',k+1,len(rows),*key,time.time()-started,check['peak_GiB'])
    for f in handles.values():f.close()
    checked.close();counter.remove();engine.close();logging.info('RUN_DONE videos=%d elapsed=%.1f',len(rows),time.time()-started)


if __name__=='__main__':main()
