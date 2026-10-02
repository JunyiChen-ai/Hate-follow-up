#!/usr/bin/env python3
"""Native and depth-contrastive window reads from the same forward; no GT."""
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
from contraster import DepthContraster


def sync_time():
    torch.cuda.synchronize();return time.perf_counter()


def read_video(j,engine,row,segments,smoke=False):
    ds,vid,dur=row['dataset'],row['video_id'],float(row['duration'])
    frames=frame_paths(ds,vid,20,'k20');assert frames,(ds,vid)
    wins=fixed_windows(dur,8);texts=[window_text(segments,a,b) for a,b in wins]
    torch.cuda.reset_peak_memory_stats();tick=sync_time()
    msgs,files=j.prefix_messages(frames,segments);prefix,enc=j.encode_prefix(msgs,files)
    if hasattr(j.model.model,'rope_deltas'):j.model.model.rope_deltas=None
    cache=j.prefix_cache(enc);P=cache.get_seq_length()
    bids,btext=j.branch_ids(msgs,VIDEO_QUESTION);zv=j.cached_margin(cache,bids,in_place=True)
    stance='Yes' if zv>0 else 'No'
    answer,atext=j.answer_ids(msgs,VIDEO_QUESTION,stance);j.extend_cache(cache,answer)
    prefix_seconds=sync_time()-tick
    history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',stance)]
    head=prefix+btext+atext;length=cache.get_seq_length();values={a:[{} for _ in wins] for a in ('base','contrast')}
    details=[];B=0;shared_seconds=readout_seconds=diagnostic_seconds=0.
    max_equivalence=max_mature_drift=0.
    for kind in ('visual','speech'):
        for i,((a,b),txt) in enumerate(zip(wins,texts)):
            if kind=='speech' and not txt.strip():continue
            tick=sync_time()
            ids,_=j.branch_ids(msgs,yesno_question(i,len(wins),a,b,txt,kind),history,head_text=head)
            h=engine.forward(cache,ids);native=j.margins_fp32(h[None])[0]
            cache.crop(length);shared_seconds+=sync_time()-tick
            tick=sync_time();z,diagnostic=engine.readout(h);readout_seconds+=sync_time()-tick
            lg=np.asarray(diagnostic['label_logits'][-1],dtype=np.float64);ny=len(j.yes_ids)
            from scipy.special import logsumexp
            full_head_native=float(logsumexp(lg[:ny])-logsumexp(lg[ny:]))
            max_mature_drift=max(max_mature_drift,abs(full_head_native-native))
            max_equivalence=max(max_equivalence,diagnostic['logprob_equivalence_error'])
            if smoke:
                tick=sync_time();reference=j.cached_margin(cache,ids,in_place=True);cache.crop(length)
                assert reference==native,('capture changes native margin',reference,native)
                diagnostic_seconds+=sync_time()-tick
            values['base'][i]['z_'+kind]=native;values['contrast'][i]['z_'+kind]=z;B+=1
            details.append({'i':i,'kind':kind,'native_margin':native,'contrast_margin':z,**diagnostic})
    del cache
    L=int(math.ceil(dur*FPS));idx=np.clip(((np.arange(L)+.5)/FPS//8).astype(int),0,len(wins)-1)
    records={}
    for arm in values:
        ww=values[arm];z=np.array([max(w.values()) for w in ww]);assert np.isfinite(z).all()
        records[arm]={'schema_version':1,'method':'m1_contraster_'+arm,'dataset':ds,'video_id':vid,
            'duration':dur,'native_rate':FPS,'score_curve':z[idx].tolist(),'intervals':[],'error':None,
            'seed':0,'code_path':str(Path(__file__).relative_to(ROOT)),'calls':3+B,
            'extra':{'z_video':zv,'stance':stance,'prefix_tokens':P,'prefix_seconds':prefix_seconds,
                'prefix_seconds_note':'native prefix, global question and forced answer',
                'branch_seconds':shared_seconds+(readout_seconds if arm=='contrast' else 0.),
                'branch_seconds_note':'base is upper bound including intermediate capture; contrast adds full readout',
                'n_branches':B,'windows':[{'i':i,'start':a,'end':b,'z':float(z[i]),**ww[i]} for i,(a,b) in enumerate(wins)]}}
    check={'dataset':ds,'video_id':vid,'prefix_tokens':P,'native_global':zv,'n_branches':B,
        'prefix_seconds':prefix_seconds,'shared_seconds':shared_seconds,'readout_seconds':readout_seconds,
        'actual_forwards':3+B+(B if smoke else 0),'diagnostic_seconds':diagnostic_seconds,
        'max_logprob_equivalence_error':max_equivalence,'max_batched_mature_margin_drift':max_mature_drift,
        'smoke_unhooked_window_parity':True if smoke else None,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,
        'branches':details}
    return records,check


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-name',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    torch.manual_seed(0);out=ROOT/'runs/20261003_m1_contraster'/a.run_name;out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',['HateMM','HateClipSeg'])
    if a.smoke:
        selected=[r for ds in ('HateMM','HateClipSeg') for r in [v for v in rows if v['dataset']==ds][:2]]
        selected+=[r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
        assert len({(r['dataset'],r['video_id']) for r in selected})==5;rows=selected
    asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};j=Judge(MODEL)
    tick=sync_time();engine=DepthContraster(j);head_setup_seconds=sync_time()-tick
    import transformers
    config={**vars(a),'date':time.strftime('%Y-%m-%d'),'host':socket.gethostname(),'seed':0,'model':MODEL,
        'torch':torch.__version__,'transformers':transformers.__version__,'GT_in_reader':False,
        'code':'experiments/20261003_m1_contraster/{measure,contraster}.py + src/{mllm_judge,video_inputs}.py, sources2026-10-03',
        'frames':20,'window_seconds':8,'fps':FPS,'layers':engine.layers,'head_setup_seconds':head_setup_seconds,
        'selection':'standard full-vocabulary JS argmax; shallowest tie','contrast':'token log probability difference; coefficient1; no APC',
        'global_key':'native original','forced_answer':'native original','yes_ids':j.yes_ids,'no_ids':j.no_ids}
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
        recs,check=read_video(j,engine,row,asr[key[0]].get(key[1],[]),smoke=a.smoke)
        for arm,r in recs.items():handles[arm].write(json.dumps(r)+'\n');handles[arm].flush()
        checked.write(json.dumps(check)+'\n');checked.flush()
        logging.info('progress %d/%d %s %s elapsed=%.1f peak_GiB=%.2f',k+1,len(rows),*key,time.time()-started,check['peak_GiB'])
    for f in handles.values():f.close()
    checked.close();engine.close();logging.info('RUN_DONE videos=%d elapsed=%.1f',len(rows),time.time()-started)

if __name__=='__main__':main()
