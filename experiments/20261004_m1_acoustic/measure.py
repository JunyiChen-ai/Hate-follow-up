#!/usr/bin/env python3
"""Paired native and support-conditioned speech margins, no GT."""
import argparse
import copy
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
from alignment import ALGORITHM,ALIGN_MODEL,CACHE_VERSION
from extract import selected_rows,validate_cache,CACHE,DATASETS
from reader import ARMS,PriorReader,arm_support,prefix_word_spans,question_word_spans,token_support


def tick():
    torch.cuda.synchronize();return time.perf_counter()


def existing_records(path,expected):
    rows={}
    if path.exists():
        for line in path.open():
            r=json.loads(line);key=r['dataset'],r['video_id']
            assert key in expected and key not in rows,('unexpected/duplicate',path,key)
            rows[key]=r
    return rows


def validate_pair(base,new,check,detail):
    assert base['dataset']==new['dataset']==check['dataset']
    assert base['video_id']==new['video_id']==check['video_id']
    assert base['duration']==new['duration'] and base['native_rate']==new['native_rate']==FPS
    assert base['error'] is new['error'] is None
    assert base['extra']['z_video']==new['extra']['z_video'] and base['extra']['stance']==new['extra']['stance']
    wins=fixed_windows(base['duration'],8)
    assert len(wins)==len(base['extra']['windows'])==len(new['extra']['windows'])==len(detail['traces'])
    for old,w,t,(a,b) in zip(base['extra']['windows'],new['extra']['windows'],detail['traces'],wins):
        assert old['i']==w['i']==t['window'] and old['start']==w['start']==a and old['end']==w['end']==b
        assert old['z_visual']==w['z_visual'] and w.get('z_speech')==t['margin']
        for item in (old,w): assert item['z']==max(item['z_visual'],item.get('z_speech',float('-inf')))
    for r in (base,new):
        index=np.clip(((np.arange(math.ceil(r['duration']*FPS))+.5)/FPS//8).astype(int),0,len(wins)-1)
        assert np.array_equal(r['score_curve'],np.asarray([w['z'] for w in r['extra']['windows']])[index])
        assert np.isfinite(r['score_curve']).all()
    assert len(detail['words'])==len(detail['support'])


@torch.no_grad()
def read_video(j,reader,row,segments,alignment,arm,smoke):
    ds,vid,duration=row['dataset'],row['video_id'],float(row['duration'])
    frames=frame_paths(ds,vid,20);assert 0<len(frames)<=20
    wins=fixed_windows(duration,8);texts=[window_text(segments,a,b) for a,b in wins]
    p=arm_support(alignment,arm);words=alignment['words']
    map_windows=p.argmax(1) if len(words) else np.empty(0,dtype=int)
    j.model.model.rope_deltas=None;torch.cuda.reset_peak_memory_stats();first=j.forward_calls
    begin=tick();msgs,files=j.prefix_messages(frames,segments);text,enc=j.encode_prefix(msgs,files)
    offsets,spans=prefix_word_spans(j,text,enc,segments,words)
    cache=j.prefix_cache(enc);P=cache.get_seq_length();assert len(offsets)==P
    qid,qtext=j.branch_ids(msgs,VIDEO_QUESTION);zv=j.cached_margin(cache,qid,in_place=True)
    stance='Yes' if zv>0 else 'No';aid,atext=j.answer_ids(msgs,VIDEO_QUESTION,stance);j.extend_cache(cache,aid)
    history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',stance)]
    head=text+qtext+atext;n=cache.get_seq_length();delta=j.model.model.rope_deltas.clone()
    prefix_seconds=tick()-begin;vs=[];ss=[];newss=[];traces=[]
    visual_seconds=speech_seconds=new_seconds=diagnostic_seconds=0.;diagnostic_calls=0;new_calls=0
    for i,((a,b),body) in enumerate(zip(wins,texts)):
        j.model.model.rope_deltas=delta.clone()
        bids,_=j.branch_ids(msgs,yesno_question(i,len(wins),a,b,body,'visual'),history,head_text=head)
        t0=tick();v=j.cached_margin(cache,bids,in_place=True);cache.crop(n);visual_seconds+=tick()-t0;vs.append(v)
        if body.strip():
            q= yesno_question(i,len(wins),a,b,body,'speech')
            bids_s,_=j.branch_ids(msgs,q,history,head_text=head);j.model.model.rope_deltas=delta.clone()
            t0=tick();s=j.cached_margin(cache,bids_s,in_place=True);cache.crop(n);speech_seconds+=tick()-t0
            if smoke and i==0:
                t0=tick();parity=reader.margin(cache,bids_s,np.ones(n+len(bids_s)),delta)
                assert parity==s,'explicit neutral attention differs from native'
                diagnostic_seconds+=tick()-t0;diagnostic_calls+=1
        else:s=None
        ss.append(s)
        selected=np.flatnonzero(map_windows==i).tolist()
        new_body=' '.join(words[k]['text'] for k in selected)
        mass=float(p[:,i].sum()) if len(words) else 0.
        trace=dict(window=i,selected_word_ids=selected,word_mass=mass,body=new_body,
                   native_margin=s,margin=None,attention_layers=0)
        if mass>1e-6:
            t0=tick();question=yesno_question(i,len(wins),a,b,new_body,'speech')
            ids,suffix=j.branch_ids(msgs,question,history,head_text=head)
            qoffsets,qspans=question_word_spans(j,suffix,ids,question,words,selected)
            pp,pm=token_support(offsets,spans,p[:,i]);qp,qm=token_support(qoffsets,qspans,p[:,i])
            support=np.r_[pp,np.ones(n-P),qp]
            if arm=='unweighted':support=np.ones_like(support)
            z=reader.margin(cache,ids,support,delta);new_seconds+=tick()-t0;new_calls+=1
            trace.update(margin=z,attention_layers=len(reader.visited),prefix_bound_tokens=int(pm.sum()),
                         body_bound_tokens=int(qm.sum()),key_support_min=float(support.min()),
                         key_support_max=float(support.max()),suffix_tokens=len(ids))
            if smoke and i==0:
                t0=tick();clone=copy.deepcopy(cache)
                replay=reader.margin(clone,ids,support,delta);assert replay==z,'cloned-cache mismatch'
                diagnostic_calls+=1;del clone
                wrong=support.copy();bound=np.r_[pm,np.zeros(n-P,dtype=bool),qm]
                if len(words) and p.shape[1]>1:
                    shifted=np.roll(p,p.shape[1]//2,axis=1)[:,i]
                    wp,_=token_support(offsets,spans,shifted);wq,_=token_support(qoffsets,qspans,shifted)
                    wrong=np.r_[wp,np.ones(n-P),wq]
                else:wrong[bound]=1.
                altered=reader.margin(cache,ids,wrong,delta);diagnostic_calls+=1
                trace.update(cloned_cache_exact=True,intervention_changes_margin=altered!=z,
                    intervention_changes_support=not np.array_equal(wrong,support),intervention_margin=altered)
                diagnostic_seconds+=tick()-t0
        else:z=None
        newss.append(z);traces.append(trace)
        assert cache.get_seq_length()==n and reader.active is None
    B=len(wins)+sum(s is not None for s in ss)
    assert j.forward_calls-first==3+B+new_calls+diagnostic_calls
    index=np.clip(((np.arange(math.ceil(duration*FPS))+.5)/FPS//8).astype(int),0,len(wins)-1)
    standalone=dict(base=prefix_seconds+visual_seconds+speech_seconds,
        optimized=prefix_seconds+visual_seconds+new_seconds+alignment['align_seconds']+alignment['decode_seconds'])
    records={}
    for name,speech in (('base',ss),('optimized',newss)):
        windows=[]
        for i,((a,b),v,s) in enumerate(zip(wins,vs,speech)):
            w=dict(i=i,start=a,end=b,z_visual=v,z=max(v,s) if s is not None else v)
            if s is not None:w['z_speech']=s
            windows.append(w)
        records[name]=dict(schema_version=1,method='m1_acoustic_'+name,dataset=ds,video_id=vid,
            duration=duration,native_rate=FPS,score_curve=np.asarray([w['z'] for w in windows])[index].tolist(),
            intervals=[],error=None,seed=0,calls=3+len(wins)+sum(s is not None for s in speech),
            code_path=str(Path(__file__).relative_to(ROOT)),extra=dict(z_video=zv,stance=stance,prefix_tokens=P,
            standalone_seconds=standalone[name],n_branches=len(wins)+sum(s is not None for s in speech),windows=windows))
    check=dict(dataset=ds,video_id=vid,visual_queries=len(wins),native_branches=B,new_speech_calls=new_calls,
        actual_forwards=j.forward_calls-first,diagnostic_calls=diagnostic_calls,diagnostic_seconds=diagnostic_seconds,
        standalone_seconds=standalone,peak_GiB=torch.cuda.max_memory_allocated()/2**30,
        align_peak_GiB=alignment['peak_GiB'],align_seconds=alignment['align_seconds'],decode_seconds=alignment['decode_seconds'],
        encoder_calls=alignment['encoder_calls'],decoder_calls=alignment['decoder_calls'],paired_seconds=tick()-begin)
    detail=dict(traces=traces,words=words,support=p.tolist(),prefix_word_spans=spans,
                original_frame_times=[t for t,f in frames],image_counts=list(j.img_tokens))
    del cache
    validate_pair(records['base'],records['optimized'],check,detail)
    return records,check,detail


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--arm',choices=ARMS,default='soft');a=ap.parse_args()
    out=ROOT/'runs/20261004_m1_acoustic'/('r1_'+a.arm+('_smoke' if a.smoke else '_main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    config=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),arm=a.arm,smoke=a.smoke,model=MODEL,
        align_model=ALIGN_MODEL,algorithm=ALGORITHM,cache_version=CACHE_VERSION,GT_in_reader=False,seed=0,torch=torch.__version__,
        transformers=transformers.__version__,frames=20,window_seconds=8,fps=FPS,
        code='experiments/20261004_m1_acoustic/{measure,reader,alignment,extract}.py + src/{video_inputs,mllm_judge}.py; sources2026-10-04')
    cp=out/'config.json'
    if cp.exists():
        old=json.loads(cp.read_text());assert {k:v for k,v in old.items() if k!='date'}=={k:v for k,v in config.items() if k!='date'}
    else:cp.write_text(json.dumps(config,indent=2)+'\n')
    rows=selected_rows(a.smoke);expected={(r['dataset'],r['video_id']):r for r in rows}
    done={name:existing_records(out/name/'predictions.jsonl',expected) for name in ('base','optimized')}
    checks=existing_records(out/'checks.jsonl',expected);assert done['base'].keys()==done['optimized'].keys()==checks.keys()
    asr={ds:load_asr(ds) for ds in DATASETS}
    for key,check in checks.items():
        detail=json.loads((out/'details'/key[0]/(key[1]+'.json')).read_text());validate_pair(done['base'][key],done['optimized'][key],check,detail)
        alignment=json.loads((CACHE/key[0]/(key[1]+'.json')).read_text())
        validate_cache(alignment,expected[key],asr[key[0]].get(key[1],[]))
        assert detail['words']==alignment['words']
        p=arm_support(alignment,a.arm)
        np.testing.assert_array_equal(np.asarray(detail['support']).reshape(p.shape),p)
    torch.manual_seed(0);j=Judge(MODEL);reader=PriorReader(j);j.forward_calls=0
    counter=j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1))
    handles={}
    for name in ('base','optimized'):
        d=out/name;d.mkdir(exist_ok=True);apath=d/'config.json';cfg={**config,'measurement':name}
        if apath.exists():
            old=json.loads(apath.read_text());assert {k:v for k,v in old.items() if k!='date'}=={k:v for k,v in cfg.items() if k!='date'}
        else:apath.write_text(json.dumps(cfg,indent=2)+'\n')
        handles[name]=(d/'predictions.jsonl').open('a')
    h=(out/'checks.jsonl').open('a');started=time.time()
    for k,row in enumerate(rows):
        key=row['dataset'],row['video_id']
        if key in checks:continue
        segments=asr[key[0]].get(key[1],[]);alignment=json.loads((CACHE/key[0]/(key[1]+'.json')).read_text())
        validate_cache(alignment,row,segments)
        recs,check,detail=read_video(j,reader,row,segments,alignment,a.arm,a.smoke)
        dest=out/'details'/key[0];dest.mkdir(parents=True,exist_ok=True);(dest/(key[1]+'.json')).write_text(json.dumps(detail)+'\n')
        for name,r in recs.items():handles[name].write(json.dumps(r)+'\n');handles[name].flush()
        h.write(json.dumps(check)+'\n');h.flush();logging.info('%d/%d %s/%s %.2fs',k+1,len(rows),*key,check['paired_seconds'])
    for f in handles.values():f.close()
    h.close();reader.close();counter.remove();logging.info('MEASUREMENT_DONE coverage=%d elapsed=%.2fs',len(rows),time.time()-started)


if __name__=='__main__':main()
