#!/usr/bin/env python3
"""Paired native reader and actual media erasure with a fresh prefix each time."""
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
from erasure import erase_frames,erase_speech


def global_read(j,frames,segments,keep=False):
    # Every intervention starts from raw remaining media and an empty LM cache.
    if hasattr(j.model.model,'rope_deltas'):j.model.model.rope_deltas=None
    msgs,files=j.prefix_messages(frames,segments);prefix,enc=j.encode_prefix(msgs,files)
    cache=j.prefix_cache(enc);P=cache.get_seq_length()
    bids,btext=j.branch_ids(msgs,VIDEO_QUESTION)
    z=j.cached_margin(cache,bids,in_place=True)
    if keep:return z,(msgs,prefix,cache,btext),P
    del cache
    return z,None,P


def read_video(j,row,segments,smoke=False):
    ds,vid,dur=row['dataset'],row['video_id'],float(row['duration'])
    frames=frame_paths(ds,vid,20,'k20');assert frames,(ds,vid)
    wins=fixed_windows(dur,8);texts=[window_text(segments,a,b) for a,b in wins]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
    zv,state,P=global_read(j,frames,segments,keep=True)
    torch.cuda.synchronize();global_seconds=time.perf_counter()-tick
    msgs,prefix,cache,btext=state
    stance='Yes' if zv>0 else 'No';tick=time.perf_counter()
    answer,atext=j.answer_ids(msgs,VIDEO_QUESTION,stance);j.extend_cache(cache,answer)
    history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',stance)]
    head=prefix+btext+atext;B=cache.get_seq_length();base=[{} for _ in wins];branches=0
    for kind in ('visual','speech'):
        for i,((a,b),txt) in enumerate(zip(wins,texts)):
            if kind=='speech' and not txt.strip():continue
            q=yesno_question(i,len(wins),a,b,txt,kind)
            ids,_=j.branch_ids(msgs,q,history,head_text=head)
            base[i]['z_'+kind]=j.cached_margin(cache,ids,in_place=True);cache.crop(B);branches+=1
    torch.cuda.synchronize();base_seconds=time.perf_counter()-tick
    del cache,state
    values=[{} for _ in wins];checks=[];erasures=0;tick=time.perf_counter()
    for i,((a,b),txt) in enumerate(zip(wins,texts)):
        kept_frames,removed_frames=erase_frames(frames,a,b,last=i==len(wins)-1)
        kept_speech,removed_speech,nwords=erase_speech(segments,a,b)
        removed_words=[w for part in removed_speech for w in part['words']]
        assert removed_words==txt.split(),(ds,vid,i,'speech-window membership mismatch')
        detail={'i':i,'start':a,'end':b,'removed_frames':removed_frames,'removed_speech':removed_speech,
                'n_frames':len(removed_frames),'n_words':nwords,'remaining_prefix_tokens':{}}
        for kind,changed,xframes,xspeech in (('visual',bool(removed_frames),kept_frames,segments),
                                             ('speech',bool(nwords),frames,kept_speech)):
            if kind=='speech' and not txt.strip():continue
            if changed:
                minus,_,plen=global_read(j,xframes,xspeech);erasures+=1
                detail['remaining_prefix_tokens'][kind]=plen
            else:minus=zv
            values[i]['z_'+kind]=float(zv-minus);detail['minus_'+kind]=minus
        checks.append(detail)
    torch.cuda.synchronize();erasure_seconds=time.perf_counter()-tick
    verify={};diagnostic_seconds=0.
    if smoke:
        tick=time.perf_counter()
        restored,_,restored_P=global_read(j,frames,segments)
        assert restored==zv and restored_P==P,('fresh-prefill changed native score',zv,restored)
        verify={'restored_global_exact':True,'original_prefix_tokens':P,'extra_forwards':2}
        torch.cuda.synchronize();diagnostic_seconds=time.perf_counter()-tick
    L=int(math.ceil(dur*FPS));index=np.clip(((np.arange(L)+.5)/FPS//8).astype(int),0,len(wins)-1)
    records={}
    for arm,ww,calls,seconds in (('base',base,3+branches,base_seconds),('erase',values,2+2*erasures,erasure_seconds)):
        z=np.array([max(w.values()) for w in ww]);assert np.isfinite(z).all()
        records[arm]={'schema_version':1,'method':'m1_eraser_'+arm,'dataset':ds,'video_id':vid,
            'duration':dur,'native_rate':FPS,'score_curve':z[index].tolist(),'intervals':[],'error':None,
            'seed':0,'code_path':str(Path(__file__).relative_to(ROOT)),'calls':calls,
            'extra':{'z_video':zv,'stance':stance,'prefix_tokens':P,'prefix_seconds':global_seconds,
                'prefix_seconds_note':'includes original whole-video question',
                'branch_seconds':seconds,'nonempty_erasures':erasures,'n_branches':branches,
                'windows':[{'i':i,'start':a,'end':b,'z':float(z[i]),**ww[i]} for i,(a,b) in enumerate(wins)]}}
    return records,{'dataset':ds,'video_id':vid,'global_seconds':global_seconds,'base_seconds':base_seconds,
        'erasure_seconds':erasure_seconds,'nonempty_erasures':erasures,'branches':branches,
        'actual_forwards':3+branches+2*erasures+verify.get('extra_forwards',0),
        'diagnostic_seconds':diagnostic_seconds,'verify':verify,'windows':checks,
        'peak_GiB':torch.cuda.max_memory_allocated()/2**30}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-name',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    torch.manual_seed(0);out=ROOT/'runs/20261003_m1_eraser'/a.run_name;out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',['HateMM','HateClipSeg'])
    if a.smoke:rows=[r for ds in ('HateMM','HateClipSeg') for r in [v for v in rows if v['dataset']==ds][:2]]
    asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};j=Judge(MODEL)
    import transformers
    config={**vars(a),'date':time.strftime('%Y-%m-%d'),'host':socket.gethostname(),'seed':0,'model':MODEL,
        'torch':torch.__version__,'transformers':transformers.__version__,'GT_in_reader':False,
        'code':'experiments/20261003_m1_eraser/{measure,erasure}.py + src/{video_inputs,mllm_judge}.py, sources2026-10-03',
        'video_question':VIDEO_QUESTION,'frames':20,'window_seconds':8,'fps':FPS,
        'speech_removal':'proportional rounded word indices, original segment timestamps retained',
        'frame_removal':'timestamp membership, retain other modality; every changed input freshly encoded'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    handles={};done={}
    for arm in ('base','erase'):
        d=out/arm;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**config,'arm':arm},indent=2)+'\n')
        p=d/'predictions.jsonl';done[arm]={(r['dataset'],r['video_id']) for r in map(json.loads,p.open())} if p.exists() else set()
        handles[arm]=p.open('a')
    assert done['base']==done['erase'],'partial arm records need repair before resume'
    checked=(out/'checks.jsonl').open('a');started=time.time()
    for k,row in enumerate(rows):
        key=row['dataset'],row['video_id']
        if key in done['base']:continue
        recs,check=read_video(j,row,asr[key[0]].get(key[1],[]),smoke=a.smoke)
        for arm,r in recs.items():handles[arm].write(json.dumps(r)+'\n');handles[arm].flush()
        checked.write(json.dumps(check)+'\n');checked.flush()
        logging.info('progress %d/%d %s %s elapsed=%.1f erasures=%d peak_GiB=%.2f',k+1,len(rows),*key,time.time()-started,check['nonempty_erasures'],check['peak_GiB'])
    for f in handles.values():f.close()
    checked.close();logging.info('RUN_DONE videos=%d elapsed=%.1f',len(rows),time.time()-started)

if __name__=='__main__':main()
