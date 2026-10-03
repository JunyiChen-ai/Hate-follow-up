#!/usr/bin/env python3
"""Independent native/phase prefix encodings, with coherent per-arm assessments."""
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
from stabilizer import PrefixStabilizer


def synced_time():
    torch.cuda.synchronize();return time.perf_counter()


def read_arm(j,engine,frames,segments,wins,texts,phase=None):
    tick=synced_time();msgs,files=j.prefix_messages(frames,segments,with_frames=True)
    text,enc=j.encode_prefix(msgs,files)
    if hasattr(j.model.model,'rope_deltas'):j.model.model.rope_deltas=None
    if phase is None:cache=j.prefix_cache(enc);diag=None
    else:cache,diag=engine.prefix_cache(enc,phase)
    prefix_seconds=synced_time()-tick;P=cache.get_seq_length();tick=synced_time()
    ids,btext=j.branch_ids(msgs,VIDEO_QUESTION)
    global_z=j.cached_margin(cache,ids,in_place=True)
    stance='Yes' if global_z>0 else 'No'
    ids,atext=j.answer_ids(msgs,VIDEO_QUESTION,stance);j.extend_cache(cache,ids)
    global_seconds=synced_time()-tick
    history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',stance)]
    head=text+btext+atext;n=cache.get_seq_length();rope=j.model.model.rope_deltas.clone()
    reads={};seconds={}
    for kind in ('visual','speech'):
        tick=synced_time();values=[]
        for i,((a,b),txt) in enumerate(zip(wins,texts)):
            if kind=='speech' and not txt.strip():values.append(None);continue
            ids,_=j.branch_ids(msgs,yesno_question(i,len(wins),a,b,txt,kind),history,head_text=head)
            values.append(j.cached_margin(cache,ids,in_place=True));cache.crop(n)
        seconds[kind]=synced_time()-tick;reads[kind]=values
    assert engine.active is None and cache.get_seq_length()==n and torch.equal(j.model.model.rope_deltas,rope)
    if phase is not None:assert engine.visited==list(range(36))
    del cache
    seconds.update(prefix=prefix_seconds,global_qa=global_seconds)
    return {'z_video':global_z,'stance':stance,'prefix_tokens':P,'visual':reads['visual'],
        'speech':reads['speech'],'seconds':seconds,'standalone_seconds':sum(seconds.values()),'phase':diag},enc


def read_video(j,engine,row,segments,smoke=False):
    ds,vid,duration=row['dataset'],row['video_id'],float(row['duration'])
    frames=frame_paths(ds,vid,20,'k20');assert 0<len(frames)<=20
    wins=fixed_windows(duration,8);texts=[window_text(segments,a,b) for a,b in wins]
    V=len(wins);B=V+sum(bool(t.strip()) for t in texts)
    initial=j.forward_calls;torch.cuda.reset_peak_memory_stats()
    base,enc=read_arm(j,engine,frames,segments,wins,texts)
    stable,enc2=read_arm(j,engine,frames,segments,wins,texts,'stable')
    assert all(torch.equal(enc[k],enc2[k]) for k in enc if torch.is_tensor(enc[k]))
    visual=(enc['input_ids'][0]==j.image_token_id).numpy()
    assert visual.sum()==sum(j.img_tokens) and len(j.img_tokens)==len(frames)
    verify={};diagnostic_seconds=0.
    if smoke:
        zero,enc0=read_arm(j,engine,frames,segments,wins,texts,'zero')
        for key in ('z_video','stance','prefix_tokens','visual','speech'):assert zero[key]==base[key]
        assert all(torch.equal(enc[k],enc0[k]) for k in enc if torch.is_tensor(enc[k]))
        assert np.array_equal(zero['phase']['geometry'],np.zeros((36,4)))
        verify={'zero_phase_native_exact':True,'native_restore_exact':True,'inputs_exact':True,'extra_forwards':3+B}
        diagnostic_seconds=zero['standalone_seconds']
    actual=j.forward_calls-initial;assert actual==6+2*B+verify.get('extra_forwards',0)
    length=int(math.ceil(duration*FPS));index=np.clip(((np.arange(length)+.5)/FPS//8).astype(int),0,V-1)
    records={}
    for arm,result in (('base',base),('stable',stable)):
        windows=[]
        for i,((a,b),v,s) in enumerate(zip(wins,result['visual'],result['speech'])):
            w={'i':i,'start':a,'end':b,'z_visual':v,'z':max(v,s) if s is not None else v}
            if s is not None:w['z_speech']=s
            windows.append(w)
        scores=np.asarray([w['z'] for w in windows]);assert np.isfinite(scores).all() and math.isfinite(result['z_video'])
        records[arm]={'schema_version':1,'method':'m1_stabilizer_'+arm,'dataset':ds,'video_id':vid,
            'duration':duration,'native_rate':FPS,'score_curve':scores[index].tolist(),'intervals':[],'error':None,
            'seed':0,'code_path':str(Path(__file__).relative_to(ROOT)),'calls':3+B,
            'extra':{'z_video':result['z_video'],'stance':result['stance'],'prefix_tokens':result['prefix_tokens'],
                'standalone_seconds':result['standalone_seconds'],'n_branches':B,'windows':windows}}
    check={'dataset':ds,'video_id':vid,'prefix_tokens':base['prefix_tokens'],'visual_tokens':int(visual.sum()),
        'visual_queries':V,'native_branches':B,'actual_forwards':actual,'verify':verify,
        'diagnostic_seconds':diagnostic_seconds,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,
        'phase':stable['phase'],'geometry_fields':['max_FP32_pair_squared_norm_error','max_cast_coordinate_delta',
            'mean_pair_squared_norm_before','mean_pair_squared_norm_after_cast'],
        'seconds':{a:r['seconds'] for a,r in (('base',base),('stable',stable))},
        'paired_seconds':base['standalone_seconds']+stable['standalone_seconds']}
    tokens={'input_ids':enc['input_ids'][0].numpy(),'visual':visual,'frame_times':np.array([f[0] for f in frames]),
        'image_counts':np.array(j.img_tokens)}
    return records,check,tokens


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261003_m1_stabilizer'/('r1_smoke' if a.smoke else 'r1_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    torch.manual_seed(0)
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',['HateMM','HateClipSeg'])
    if a.smoke:
        rows=[r for ds in ('HateMM','HateClipSeg') for r in [v for v in rows if v['dataset']==ds][:2]]+[
            r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
        assert len(rows)==5
    asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};j=Judge(MODEL);engine=PrefixStabilizer(j)
    assert len(engine.inv_freq)==64 and len(engine.pairs)==24
    assert j.model.model.language_model.layers[0].self_attn.config.num_attention_heads==32
    j.forward_calls=0
    def count(*_):j.forward_calls+=1
    counter=j.model.model.register_forward_pre_hook(count)
    import transformers
    config={'date':time.strftime('%Y-%m-%d'),'host':socket.gethostname(),'smoke':a.smoke,'seed':0,
        'model':MODEL,'torch':torch.__version__,'transformers':transformers.__version__,'GT_in_reader':False,
        'code':'experiments/20261003_m1_stabilizer/{measure,stabilizer}.py + src/{video_inputs,mllm_judge}.py; sources2026-10-03',
        'layers':list(range(36)),'deltas':[0.,.5]*16,'temporal_pairs':engine.pairs.cpu().tolist(),
        'inv_freq':engine.inv_freq.cpu().tolist(),'pairing':'split-half','axis_layout':'native interleaved',
        'intervention':'image Q temporal coordinates after native RoPE during prefix only',
        'frames':20,'window_seconds':8,'fps':FPS,'video_question':VIDEO_QUESTION,
        'global_and_answer':'own-arm coherent','speech':'own-arm coherent','timing_note':'fresh identical input encoding in each arm'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');handles={};done={}
    for arm in ('base','stable'):
        d=out/arm;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**config,'arm':arm},indent=2)+'\n')
        p=d/'predictions.jsonl';done[arm]={(r['dataset'],r['video_id']) for r in map(json.loads,p.open())} if p.exists() else set()
        handles[arm]=p.open('a')
    cp=out/'checks.jsonl';seen={(r['dataset'],r['video_id']) for r in map(json.loads,cp.open())} if cp.exists() else set()
    assert done['base']==done['stable']==seen,'incomplete paired record'
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
