#!/usr/bin/env python3
"""Paired original/corrupted-image visual queries; no annotations in scoring."""
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
from visual_contrast import corrupt_pixels,class_contrast,mix_window_pixels


def tick():torch.cuda.synchronize();return time.perf_counter()


def fresh_prefix(j,enc):
    if hasattr(j.model.model,'rope_deltas'):j.model.model.rope_deltas=None
    return j.prefix_cache(enc)


def read_windows(j,cache,msgs,prefix,btext,stance,wins,texts,visual_only=False,only_indices=None):
    answer,atext=j.answer_ids(msgs,VIDEO_QUESTION,stance);j.extend_cache(cache,answer)
    history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',stance)]
    head=prefix+btext+atext;length=cache.get_seq_length();values=[{} for _ in wins];B=0
    for kind in (('visual',) if visual_only else ('visual','speech')):
        for i,((a,b),txt) in enumerate(zip(wins,texts)):
            if only_indices is not None and i not in only_indices:continue
            if kind=='speech' and not txt.strip():continue
            ids,_=j.branch_ids(msgs,yesno_question(i,len(wins),a,b,txt,kind),history,head_text=head)
            values[i]['z_'+kind]=j.cached_margin(cache,ids,in_place=True);cache.crop(length);B+=1
    return values,B


def read_video(j,row,segments,smoke=False,noise_scope='full'):
    ds,vid,dur=row['dataset'],row['video_id'],float(row['duration'])
    frames=frame_paths(ds,vid,20,'k20');assert frames,(ds,vid)
    wins=fixed_windows(dur,8);texts=[window_text(segments,a,b) for a,b in wins]
    members=[[k for k,(t,_) in enumerate(frames) if a<=t and (t<b or (i==len(wins)-1 and t<=b))]
             for i,(a,b) in enumerate(wins)]
    assert sorted(k for mm in members for k in mm)==list(range(len(frames)))
    torch.cuda.reset_peak_memory_stats();start=tick()
    msgs,files=j.prefix_messages(frames,segments);prefix,enc=j.encode_prefix(msgs,files)
    cache=fresh_prefix(j,enc);P=cache.get_seq_length()
    bids,btext=j.branch_ids(msgs,VIDEO_QUESTION);zv=j.cached_margin(cache,bids,in_place=True)
    stance='Yes' if zv>0 else 'No';prefix_seconds=tick()-start;start=tick()
    native,B=read_windows(j,cache,msgs,prefix,btext,stance,wins,texts);del cache
    base_seconds=tick()-start;start=tick()
    original={k:v.clone() for k,v in enc.items() if torch.is_tensor(v)}
    ip=j.processor.image_processor
    pixels,noise=corrupt_pixels(enc['pixel_values'],ip.patch_size,ip.temporal_patch_size)
    noise_globals=[];pixel_interventions=[]
    if noise_scope=='full':
        noisy={**enc,'pixel_values':pixels};cache=fresh_prefix(j,noisy)
        assert cache.get_seq_length()==P
        noise_globals.append(j.cached_margin(cache,bids,in_place=True))
        corrupted,V=read_windows(j,cache,msgs,prefix,btext,stance,wins,texts,visual_only=True);del cache
        assert V==len(wins);extra_forwards=3+V
    elif noise_scope=='window':
        assert len(enc['image_grid_thw'])==len(frames)
        corrupted=[{'z_visual':w['z_visual']} for w in native];V=len(wins);K=0
        for i,indices in enumerate(members):
            if not indices:continue
            partial,detail=mix_window_pixels(enc['pixel_values'],pixels,enc['image_grid_thw'],indices)
            if smoke:
                chosen=torch.zeros(len(partial),dtype=torch.bool)
                for a,b in detail['patch_ranges']:chosen[a:b]=True
                assert torch.equal(partial[chosen],pixels[chosen])
                assert torch.equal(partial[~chosen],enc['pixel_values'].float()[~chosen])
            pixel_interventions.append({'window':i,**detail})
            noisy={**enc,'pixel_values':partial};cache=fresh_prefix(j,noisy)
            assert cache.get_seq_length()==P
            noise_globals.append(j.cached_margin(cache,bids,in_place=True))
            one,C=read_windows(j,cache,msgs,prefix,btext,stance,wins,texts,visual_only=True,only_indices=[i])
            assert C==1;corrupted[i]=one[i];K+=1;del cache
        extra_forwards=4*K
    else:raise ValueError(noise_scope)
    assert all(torch.equal(enc[k],v) for k,v in original.items()),'original encoder input was mutated'
    del original
    contrast=[{**a,'z_visual':class_contrast(a['z_visual'],b['z_visual'])} for a,b in zip(native,corrupted)]
    extra_seconds=tick()-start;diagnostic_seconds=0.;verify={}
    if smoke:
        start=tick();cache=fresh_prefix(j,enc);rz=j.cached_margin(cache,bids,in_place=True)
        restored,RB=read_windows(j,cache,msgs,prefix,btext,stance,wins,texts);del cache
        assert rz==zv and RB==B and restored==native,'native restoration parity'
        identity,_=corrupt_pixels(enc['pixel_values'],ip.patch_size,ip.temporal_patch_size,abar=1.)
        repeated,_=corrupt_pixels(enc['pixel_values'],ip.patch_size,ip.temporal_patch_size)
        assert torch.equal(identity,enc['pixel_values'].float()) and torch.equal(repeated,pixels)
        diagnostic_seconds=tick()-start
        verify={'native_restore_exact':True,'pixel_identity_and_determinism':True,'extra_forwards':3+B}
    L=int(math.ceil(dur*FPS));idx=np.clip(((np.arange(L)+.5)/FPS//8).astype(int),0,len(wins)-1)
    records={}
    for arm,ww,calls,seconds in (('base',native,3+B,base_seconds),('contrast',contrast,3+B+extra_forwards,base_seconds+extra_seconds)):
        z=np.array([max(w.values()) for w in ww]);assert np.isfinite(z).all()
        records[arm]={'schema_version':1,'method':'m1_visual_contrast_'+arm,'dataset':ds,'video_id':vid,
            'duration':dur,'native_rate':FPS,'score_curve':z[idx].tolist(),'intervals':[],'error':None,
            'seed':0,'code_path':str(Path(__file__).relative_to(ROOT)),'calls':calls,
            'extra':{'z_video':zv,'stance':stance,'prefix_tokens':P,'prefix_seconds':prefix_seconds,'noise_scope':noise_scope,
                'branch_seconds':seconds,'branch_seconds_note':'contrast includes native reads and all scope-specific prefixes, global QA and visual queries',
                'n_branches':B,'windows':[{'i':i,'start':a,'end':b,'z':float(z[i]),**ww[i]} for i,(a,b) in enumerate(wins)]}}
    check={'dataset':ds,'video_id':vid,'prefix_tokens':P,'native_global':zv,
        'corrupted_globals_diagnostic_only':noise_globals,'noise_scope':noise_scope,
        'corrupted_global_diagnostic_only':noise_globals[0] if noise_scope=='full' else None,
        'pixel_interventions':pixel_interventions,
        'native_branches':B,'visual_branches':V,'noisy_visual_queries':V if noise_scope=='full' else K,
        'noise':noise,'prefix_seconds':prefix_seconds,
        'base_seconds':base_seconds,'extra_seconds':extra_seconds,'actual_forwards':3+B+extra_forwards+verify.get('extra_forwards',0),
        'diagnostic_seconds':diagnostic_seconds,'verify':verify,'original_inputs_unchanged':True,
        'peak_GiB':torch.cuda.max_memory_allocated()/2**30,
        'windows':[{'i':i,'start':a,'end':b,'clean_visual':native[i]['z_visual'],'corrupted_visual':corrupted[i]['z_visual'],
            'contrast_visual':contrast[i]['z_visual'],
            'n_frames':len(members[i])} for i,(a,b) in enumerate(wins)]}
    return records,check


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-name',required=True);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--noise-scope',choices=('full','window'),default='full');a=ap.parse_args()
    torch.manual_seed(0);out=ROOT/'runs/20261003_m1_visual_contrast'/a.run_name;out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',['HateMM','HateClipSeg'])
    if a.smoke:
        selected=[r for ds in ('HateMM','HateClipSeg') for r in [v for v in rows if v['dataset']==ds][:2]]
        selected+=[r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
        assert len({(r['dataset'],r['video_id']) for r in selected})==5;rows=selected
    asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};j=Judge(MODEL)
    import transformers
    config={**vars(a),'date':time.strftime('%Y-%m-%d'),'host':socket.gethostname(),'seed':0,'model':MODEL,
        'torch':torch.__version__,'transformers':transformers.__version__,'GT_in_reader':False,
        'code':'experiments/20261003_m1_visual_contrast/{measure,visual_contrast}.py + src/{mllm_judge,video_inputs}.py, sources2026-10-03',
        'frames':20,'window_seconds':8,'fps':FPS,'noise_step':500,'noise_seed':'CPU0 reset per video',
        'noise_schedule':'1000 sigmoid steps beta1e-5 to.005 over linspace(-6,6)',
        'contrast':'2*clean_visual_class_margin-corrupt_visual_class_margin; speech unchanged; no APC',
        'global_key':'native original','forced_answer':'native original'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');handles={};done={}
    for arm in ('base','contrast'):
        d=out/arm;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**config,'arm':arm},indent=2)+'\n')
        p=d/'predictions.jsonl';done[arm]={(r['dataset'],r['video_id']) for r in map(json.loads,p.open())} if p.exists() else set()
        handles[arm]=p.open('a')
    cp=out/'checks.jsonl';done_checks={(r['dataset'],r['video_id']) for r in map(json.loads,cp.open())} if cp.exists() else set()
    assert done['base']==done['contrast']==done_checks,'partial paired output needs repair before resume'
    checked=cp.open('a');started=time.time()
    for k,row in enumerate(rows):
        key=row['dataset'],row['video_id']
        if key in done['base']:continue
        recs,check=read_video(j,row,asr[key[0]].get(key[1],[]),smoke=a.smoke,noise_scope=a.noise_scope)
        for arm,r in recs.items():handles[arm].write(json.dumps(r)+'\n');handles[arm].flush()
        checked.write(json.dumps(check)+'\n');checked.flush()
        logging.info('progress %d/%d %s %s elapsed=%.1f peak_GiB=%.2f',k+1,len(rows),*key,time.time()-started,check['peak_GiB'])
    for f in handles.values():f.close()
    checked.close();logging.info('RUN_DONE videos=%d elapsed=%.1f',len(rows),time.time()-started)

if __name__=='__main__':main()
