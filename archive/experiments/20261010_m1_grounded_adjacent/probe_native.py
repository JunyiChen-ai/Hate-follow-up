"""Candidate 41, revision 1 (2026-10-11, README section "Revision 1"): r6's reads plus two text-only probes per
window on the native context. Nothing new is shown to the model. Probe A asks which prefix frames lie inside the
window; probe B asks which frame the answer was based on. Their answers mark windows for an extra read that
probe_native_analyze.py applies offline from the stored reads of the placement controls. No GT is read here."""
import argparse
import json
import logging
import math
import os
import re
import socket
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.stance_cache import build,margin
from src.mllm_judge import Judge,MODEL,yesno_question
from src.video_inputs import frame_paths,load_asr,load_manifest,fixed_windows,window_text

EXP=ROOT/'runs/20261010_m1_grounded_adjacent'
DATASETS=('HateMM','HateClipSeg')
SPEC=dict(version='R1-probe-before-frames-2026-10-11',model=MODEL,native_frames=20,window_seconds=8,seed=0,label='[t={time:.1f}s]\n',
    probe_a='Which of the frames shown above have timestamps inside this window, from {start:.1f}s to {end:.1f}s? List their timestamps exactly as labelled, separated by commas, or reply "none".',
    probe_b='Which single frame shown above did you base that answer on? Reply with its timestamp exactly as labelled, or "none".',
    max_new_tokens_a=48,max_new_tokens_b=16,match_tolerance=.05,
    rule_a='flag when the matched set differs from the prefix frames inside [start,end); an empty set is correct only with "none" and no matched time',
    rule_b='flag when no matched time lies inside [start,end)',
    reads='native V and S from the prefix cache exactly as r6; probes text only on the same cache; extra reads applied offline from placement_controls_main records',
    scope='HateMM215/HateClipSeg118 test;4fps sole evaluator;fixed r6;development-selected;no GT in reading')
IMPLEMENTATION='probe_native.py 2026-10-11: margin() for V and S as r6; probes with Judge.cached_generate in place on the same cache; cache cropped and rope_deltas restored after each probe; forward and vision counts asserted'
NUMBER=re.compile(r'\d+(?:\.\d+)?')
PREDICTIONS=('base',)


def selected_rows(smoke=False):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:
        rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def windows_for(row,segments):
    return [dict(i=i,start=a,end=b,body=window_text(segments,a,b)) for i,(a,b) in enumerate(fixed_windows(float(row['duration']),SPEC['window_seconds']))]


def prediction(row,ctx,visual,speech,seconds,name):
    ww=[]
    for w,v,s in zip(windows_for(row,[]),visual,speech):
        one=dict(i=w['i'],start=w['start'],end=w['end'],z_visual=v,z=max(v,s) if s is not None else v)
        if s is not None:one['z_speech']=s
        ww.append(one)
    idx=np.clip(((np.arange(math.ceil(float(row['duration'])*4))+.5)/4//SPEC['window_seconds']).astype(int),0,len(ww)-1)
    return dict(schema_version=1,method=name,dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),native_rate=4,
        score_curve=np.asarray([w['z'] for w in ww])[idx].tolist(),intervals=[],error=None,seed=0,calls=3+len(ww)+sum(s is not None for s in speech),
        code_path='experiments/20261010_m1_grounded_adjacent/probe_native.py',
        extra=dict(z_video=ctx['global_margin'],stance=ctx['stance'],prefix_tokens=ctx['prefix_tokens'],stance_cache_tokens=ctx['stance_cache_tokens'],
            stance_cache_logical_start=ctx['stance_cache_logical_start'],windows=ww,standalone_seconds=seconds))


def match(reply,frame_times):
    numbers=[float(x) for x in NUMBER.findall(reply)]
    matched=sorted({t for t in frame_times if any(abs(x-t)<=SPEC['match_tolerance']+1e-9 for x in numbers)})  # +1e-9: labels round .x5 up
    return numbers,matched,'none' in reply.lower()


def parse_a(reply,frame_times,truth):
    numbers,matched,none=match(reply,frame_times);truth=sorted(truth)
    correct=(matched==truth) if truth else (none and not matched)
    return dict(reply_a=reply,numbers_a=numbers,matched_a=matched,none_a=none,correct_a=bool(correct),missed_a=len(set(truth)-set(matched)),extra_a=len(set(matched)-set(truth)),
        parse_failure_a=(not numbers and not none),flag_a=not bool(correct))


def parse_b(reply,frame_times,start,end):
    numbers,matched,none=match(reply,frame_times);inside=[t for t in matched if start<=t<end]
    return dict(reply_b=reply,numbers_b=numbers,matched_b=matched,none_b=none,inside_b=bool(inside),parse_failure_b=(not numbers and not none),flag_b=not bool(inside))


def tick(j):
    if j.device.type=='cuda':torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def probes(j,cache,ctx,question,v,start,end):
    """Two text-only probes after the native read of this window, on the same cache. Returns {'a':(reply,generated,prefill_tokens),'b':...}."""
    n=cache.get_seq_length();assert n==ctx['stance_cache_tokens'];head=ctx['head']
    conv=j.conv(ctx['msgs'],question,ctx['history']);base=j.render(conv,True);assert base.startswith(head)
    bids,_=j.branch_ids(ctx['msgs'],question,ctx['history'],head_text=head);assert j.tok(base[len(head):],add_special_tokens=False)['input_ids']==bids  # same turn as the native read
    pa=SPEC['probe_a'].format(start=start,end=end);pb=SPEC['probe_b']
    full_a=j.render(conv+[j.turn('user',pa)],True);full_b=j.render(conv+[j.turn('assistant','Yes' if v>0 else 'No'),j.turn('user',pb)],True)
    assert full_a.startswith(head) and full_b.startswith(head) and full_a.count(pa)==1 and full_b.count(pb)==1 and question in full_a and question in full_b
    out={}
    for key,text,mx in (('a',full_a[len(head):],SPEC['max_new_tokens_a']),('b',full_b[len(head):],SPEC['max_new_tokens_b'])):
        ids=j.tok(text,add_special_tokens=False)['input_ids'];first=j.forward_calls;vision=j.vision_calls;j.model.model.rope_deltas=ctx['rope'].clone()
        try:reply,gen=j.cached_generate(cache,ids,max_new_tokens=mx,in_place=True)
        finally:cache.crop(n);j.model.model.rope_deltas=ctx['rope'].clone()
        assert j.forward_calls-first==1+len(gen) and j.vision_calls==vision and cache.get_seq_length()==n
        out[key]=(reply,len(gen),len(ids))
    return out


@torch.no_grad()
def read(j,row,segments):
    begin=tick(j);first=j.forward_calls;vision=j.vision_calls
    if j.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    native=frame_paths(row['dataset'],row['video_id'],SPEC['native_frames']);assert 0<len(native)<=SPEC['native_frames'];frame_times=[t for t,_ in native]
    stamp=tick(j);cache,ctx=build(j,native,segments);times=dict(prefix=tick(j)-stamp,native_visual=0.,native_speech=0.,probes=0.)
    windows=windows_for(row,segments);visual=[];speech=[];traces=[];generated=0
    for w in windows:
        q=yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')
        stamp=tick(j);v=margin(j,cache,ctx,q);times['native_visual']+=tick(j)-stamp
        if w['body'].strip():
            stamp=tick(j);s=margin(j,cache,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'speech'));times['native_speech']+=tick(j)-stamp
        else:s=None
        truth=[t for t in frame_times if w['start']<=t<w['end']]
        stamp=tick(j);pr=probes(j,cache,ctx,q,v,w['start'],w['end']);times['probes']+=tick(j)-stamp
        trace=dict(i=w['i'],bounds=[w['start'],w['end']],native_visual=v,native_speech=s,inside_times=truth,**parse_a(pr['a'][0],frame_times,truth),**parse_b(pr['b'][0],frame_times,w['start'],w['end']),
            generated_a=pr['a'][1],generated_b=pr['b'][1],prefill_tokens_a=pr['a'][2],prefill_tokens_b=pr['b'][2])
        generated+=pr['a'][1]+pr['b'][1];visual.append(v);speech.append(s);traces.append(trace)
    native_seconds=sum(times[k] for k in ('prefix','native_visual','native_speech'))
    preds=dict(base=prediction(row,ctx,visual,speech,native_seconds,'m1_native'))
    checks=dict(GT_read=False,host=socket.gethostname(),times=times,actual_forwards=j.forward_calls-first,actual_vision=j.vision_calls-vision,windows=len(windows),
        covered=sum(bool(t['inside_times']) for t in traces),generated_tokens=generated,correct_a=sum(t['correct_a'] for t in traces),flag_a=sum(t['flag_a'] for t in traces),
        flag_b=sum(t['flag_b'] for t in traces),none_a=sum(t['none_a'] for t in traces),none_b=sum(t['none_b'] for t in traces),inside_b=sum(t['inside_b'] for t in traces),
        parse_failures=sum(t['parse_failure_a']+t['parse_failure_b'] for t in traces),peak_GiB=torch.cuda.max_memory_allocated()/2**30 if j.device.type=='cuda' else 0.,pipeline_seconds=tick(j)-begin)
    assert checks['actual_forwards']==preds['base']['calls']+2*len(windows)+generated and checks['actual_vision']==1
    return dict(version=SPEC['version'],implementation=IMPLEMENTATION,spec=SPEC,checks=checks,
        native=dict(global_margin=ctx['global_margin'],stance=ctx['stance'],stance_cache_tokens=ctx['stance_cache_tokens'],frame_times=frame_times),predictions=preds,traces=traces)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=EXP/('probe_smoke' if a.smoke else 'probe_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),spec=SPEC,model=MODEL,implementation=IMPLEMENTATION,GT_read=False,smoke=a.smoke,
        torch=torch.__version__,transformers=transformers.__version__,source='experiments/20261010_m1_grounded_adjacent/probe_native.py;2026-10-11',command='python -u '+' '.join(sys.argv))
    number=1
    while (out/f'pipeline_attempt_{number:04d}.json').exists():number+=1
    audit_path=out/f'pipeline_attempt_{number:04d}.json';audit=dict(host=cfg['host'],completed=False,GT_read=False,videos=[]);start=time.perf_counter();j=None;hooks=[]
    results={name:[] for name in PREDICTIONS}
    try:
        path=out/'config.json'
        if path.exists():
            old=json.loads(path.read_text());assert all(old[k]==cfg[k] for k in cfg if k not in ('date','command'))
        else:path.write_text(json.dumps(cfg,indent=2)+'\n')
        asr={ds:load_asr(ds) for ds in DATASETS};stamp=time.perf_counter();torch.manual_seed(SPEC['seed']);torch.set_num_threads(4);j=Judge(MODEL)
        audit['model_load_seconds']=time.perf_counter()-stamp;j.forward_calls=j.vision_calls=0
        hooks=[j.model.model.language_model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),
            j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
        rows=selected_rows(a.smoke)
        for ordinal,row in enumerate(rows,1):
            stamp=time.perf_counter();first=j.forward_calls
            path=out/'records'/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True);reused=path.exists()
            if reused:bundle=json.loads(path.read_text())
            else:
                bundle=read(j,row,asr[row['dataset']].get(row['video_id'],[]))
                partial=path.with_suffix('.partial');partial.write_text(json.dumps(bundle,allow_nan=False)+'\n');partial.replace(path)
            for name in PREDICTIONS:results[name].append(bundle['predictions'][name])
            c=bundle['checks']
            audit['videos'].append(dict(dataset=row['dataset'],video_id=row['video_id'],reused=reused,elapsed_seconds=time.perf_counter()-stamp,current_actual_forwards=j.forward_calls-first,
                windows=c['windows'],covered=c['covered'],correct_a=c['correct_a'],flag_a=c['flag_a'],flag_b=c['flag_b'],none_a=c['none_a'],inside_b=c['inside_b'],parse_failures=c['parse_failures'],peak_GiB=c['peak_GiB']))
            audit['elapsed_seconds']=time.perf_counter()-start;audit_path.write_text(json.dumps(audit,indent=2)+'\n')
            logging.info('%d/%d %s/%s windows=%d covered=%d correct_a=%d flag_a=%d flag_b=%d none_a=%d inside_b=%d parse_fail=%d gen=%d probes=%.1fs peak=%.2fGiB',ordinal,len(rows),row['dataset'],row['video_id'],
                c['windows'],c['covered'],c['correct_a'],c['flag_a'],c['flag_b'],c['none_a'],c['inside_b'],c['parse_failures'],c['generated_tokens'],c['times']['probes'],c['peak_GiB'])
        for name,rr in results.items():
            folder=out/name;folder.mkdir(exist_ok=True);(folder/'config.json').write_text(json.dumps({**cfg,'measurement':name},indent=2)+'\n')
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
        audit['completed']=True;logging.info('DONE coverage=%d',len(rows))
    finally:
        for h in hooks:h.remove()
        audit['elapsed_seconds']=time.perf_counter()-start;audit['actual_forwards']=j.forward_calls if j else 0;audit['actual_vision']=j.vision_calls if j else 0
        audit_path.write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
