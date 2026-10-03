#!/usr/bin/env python3
"""Native and preserved window reads, one frozen model and no annotation access."""
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
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0, str(ROOT))
from src.mllm_judge import Judge, MODEL, VIDEO_QUESTION, yesno_question
from src.video_inputs import FPS, frame_paths, load_asr, load_manifest, fixed_windows, window_text
from preserver import AttentionPreserver


def synced_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def prefix(j, frames, segments):
    msgs, files = j.prefix_messages(frames, segments, with_frames=True)
    text, enc = j.encode_prefix(msgs, files)
    j.model.model.rope_deltas = None
    cache = j.prefix_cache(enc)
    P = cache.get_seq_length()
    ids, btext = j.branch_ids(msgs, VIDEO_QUESTION)
    zv = j.cached_margin(cache, ids, in_place=True)
    stance = 'Yes' if zv > 0 else 'No'
    ids, atext = j.answer_ids(msgs, VIDEO_QUESTION, stance)
    j.extend_cache(cache, ids)
    history = [{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]}, j.turn('assistant', stance)]
    return cache, msgs, text+btext+atext, history, zv, stance, enc, P


def query_ids(j, msgs, head, history, wins, texts, kind):
    return [j.branch_ids(msgs, yesno_question(i,len(wins),a,b,t,kind), history, head_text=head)[0]
            for i, ((a,b),t) in enumerate(zip(wins,texts))]


def native_reads(j, cache, ids, available):
    n = cache.get_seq_length()
    values = []
    for tokens, use in zip(ids, available):
        values.append(j.cached_margin(cache, tokens, in_place=True) if use else None)
        cache.crop(n)
    return values


def read_video(j, engine, row, segments, smoke=False):
    ds, vid, duration = row['dataset'], row['video_id'], float(row['duration'])
    frames = frame_paths(ds, vid, 20, 'k20')
    assert 0 < len(frames) <= 20
    wins = fixed_windows(duration, 8)
    texts = [window_text(segments,a,b) for a,b in wins]
    V = len(wins); available = [bool(t.strip()) for t in texts]; B = V+sum(available)
    initial = j.forward_calls
    torch.cuda.reset_peak_memory_stats()
    tick = synced_time()
    cache,msgs,head,history,zv,stance,enc,P = prefix(j,frames,segments)
    full_rope = j.model.model.rope_deltas.clone()
    image_counts = np.asarray(j.img_tokens)
    prefix_seconds = synced_time()-tick
    n = cache.get_seq_length()
    vis_ids = query_ids(j,msgs,head,history,wins,texts,'visual')
    speech_ids = query_ids(j,msgs,head,history,wins,texts,'speech')
    tick = synced_time()
    bv = native_reads(j,cache,vis_ids,[True]*V)
    visual_seconds = synced_time()-tick
    tick = synced_time()
    speech = native_reads(j,cache,speech_ids,available)
    speech_seconds = synced_time()-tick
    tick = synced_time()
    rmsgs,files = j.prefix_messages(frames,segments,with_context=False,with_frames=True)
    _,renc = j.encode_prefix(rmsgs,files)
    assert np.array_equal(j.img_tokens,image_counts)
    assert torch.equal(enc['pixel_values'],renc['pixel_values'])
    assert torch.equal(enc['image_grid_thw'],renc['image_grid_thw'])
    j.model.model.rope_deltas = None
    rc = j.prefix_cache(renc)
    ref_rope = j.model.model.rope_deltas.clone()
    R = rc.get_seq_length()
    reference_prefix_seconds = synced_time()-tick
    rv=[]; pv=[]; rl=[]; pl=[]; geometry=[]
    reference_seconds = mixed_seconds = diagnostic_seconds = 0.
    for i,ids in enumerate(vis_ids):
        tick = synced_time()
        rz,lg,_ = engine.run(rc,ids,ref_rope,'capture')
        reference_seconds += synced_time()-tick
        rv.append(rz); rl.append(lg)
        tick = synced_time()
        pz,lg,geom = engine.run(cache,ids,full_rope,'mix')
        mixed_seconds += synced_time()-tick
        pv.append(pz); pl.append(lg); geometry.append(geom)
        if smoke:
            tick = synced_time()
            zz,_,_ = engine.run(cache,ids,full_rope,'mix',alpha=0.)
            assert zz == bv[i]
            diagnostic_seconds += synced_time()-tick
        engine.clear()
        assert cache.get_seq_length()==n and rc.get_seq_length()==R
    assert engine.mode is None
    del cache,rc
    verify = {}
    if smoke:
        tick = synced_time()
        c,m,h,hi,z,s,e,p = prefix(j,frames,segments)
        assert z==zv and s==stance and p==P
        assert all(torch.equal(enc[k],e[k]) for k in enc if torch.is_tensor(enc[k]))
        assert native_reads(j,c,query_ids(j,m,h,hi,wins,texts,'visual'),[True]*V)==bv
        assert native_reads(j,c,query_ids(j,m,h,hi,wins,texts,'speech'),available)==speech
        del c
        diagnostic_seconds += synced_time()-tick
        verify={'native_windows_exact':True,'native_global_exact':True,'alpha_zero_exact':True,
                'inputs_exact':True,'extra_forwards':3+B+V}
    actual = j.forward_calls-initial
    assert actual == 4+B+2*V+verify.get('extra_forwards',0)
    length = math.ceil(duration*FPS)
    index = np.clip(((np.arange(length)+.5)/FPS//8).astype(int),0,V-1)
    seconds={'prefix':prefix_seconds,'native_visual':visual_seconds,'speech':speech_seconds,
             'reference_prefix':reference_prefix_seconds,'reference_queries':reference_seconds,'mixed':mixed_seconds}
    standalone={'base':prefix_seconds+visual_seconds+speech_seconds,
                'preserve':prefix_seconds+speech_seconds+reference_prefix_seconds+reference_seconds+mixed_seconds}
    records={}
    for arm,values in (('base',bv),('preserve',pv)):
        windows=[]
        for i,((a,b),v,s) in enumerate(zip(wins,values,speech)):
            w={'i':i,'start':a,'end':b,'z_visual':v,'z':max(v,s) if s is not None else v}
            if s is not None:w['z_speech']=s
            windows.append(w)
        scores=np.asarray([w['z'] for w in windows]);assert np.isfinite(scores).all()
        records[arm]={'schema_version':1,'method':'m1_preserver_'+arm,'dataset':ds,'video_id':vid,
            'duration':duration,'native_rate':FPS,'score_curve':scores[index].tolist(),'intervals':[],
            'error':None,'seed':0,'code_path':str(Path(__file__).relative_to(ROOT)),
            'calls':3+B if arm=='base' else 4+B+V,
            'extra':{'z_video':zv,'stance':stance,'prefix_tokens':P,'standalone_seconds':standalone[arm],
                     'n_branches':B,'windows':windows}}
    visual=(enc['input_ids'][0]==j.image_token_id).numpy()
    check={'dataset':ds,'video_id':vid,'prefix_tokens':P,'reference_prefix_tokens':R,
        'visual_tokens':int(visual.sum()),'visual_queries':V,'native_branches':B,'actual_forwards':actual,
        'verify':verify,'diagnostic_seconds':diagnostic_seconds,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,
        'geometry':geometry,'geometry_fields':['ordinary_norm','reference_norm','change_norm'],
        'seconds':seconds,'paired_seconds':sum(seconds.values()),'layer_indices':list(range(36))}
    tokens={'input_ids':enc['input_ids'][0].numpy(),'visual':visual,'image_counts':image_counts,
        'reference_input_ids':renc['input_ids'][0].numpy(),
        'reference_visual':(renc['input_ids'][0]==j.image_token_id).numpy(),
        'full_rope':full_rope.cpu().numpy(),'reference_rope':ref_rope.cpu().numpy(),
        'query_ids':np.concatenate([np.asarray(x,dtype=np.int64) for x in vis_ids]),
        'query_offsets':np.r_[0,np.cumsum([len(x) for x in vis_ids])],
        'frame_times':np.array([f[0] for f in frames]),'yes_ids':np.array(j.yes_ids),'no_ids':np.array(j.no_ids),
        'preserve':np.stack(pl),'reference':np.stack(rl),'reference_margins':np.array(rv)}
    return records,check,tokens


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261003_m1_preserver'/('r1_smoke' if a.smoke else 'r1_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    torch.manual_seed(0)
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',['HateMM','HateClipSeg'])
    if a.smoke:
        rows=[r for ds in ('HateMM','HateClipSeg') for r in [v for v in rows if v['dataset']==ds][:2]]+[
            r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
        assert len(rows)==5
    asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};j=Judge(MODEL);engine=AttentionPreserver(j)
    j.forward_calls=0
    def count(*_):j.forward_calls+=1
    counter=j.model.model.register_forward_pre_hook(count)
    import transformers
    config={'date':time.strftime('%Y-%m-%d'),'host':socket.gethostname(),'smoke':a.smoke,'seed':0,'model':MODEL,
        'torch':torch.__version__,'transformers':transformers.__version__,'GT_in_reader':False,
        'code':'experiments/20261003_m1_preserver/{measure,preserver}.py + src/{video_inputs,mllm_judge}.py; sources2026-10-03',
        'alpha':.5,'layers':list(range(36)),'scope':'all identical query suffix rows, post-o_proj',
        'reference':'same frames and policy, no transcript or global verdict',
        'frames':20,'window_seconds':8,'fps':FPS,'video_question':VIDEO_QUESTION,
        'global_and_answer':'original native','speech':'original native','timing_note':'two prefix encodes, reference queries and mixing included'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');handles={};done={}
    for arm in ('base','preserve'):
        d=out/arm;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**config,'arm':arm},indent=2)+'\n')
        p=d/'predictions.jsonl';done[arm]={(r['dataset'],r['video_id']) for r in map(json.loads,p.open())} if p.exists() else set()
        handles[arm]=p.open('a')
    cp=out/'checks.jsonl';seen={(r['dataset'],r['video_id']) for r in map(json.loads,cp.open())} if cp.exists() else set()
    assert done['base']==done['preserve']==seen,'incomplete paired record'
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
