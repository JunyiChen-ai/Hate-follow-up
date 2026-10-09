"""M1 candidate 41: grounded acceptance of adjacent-frame reads (README, declared 2026-10-10). No annotations.

Per window with at least one prefix frame inside: the adjacent read (frames shown again before r6's visual question,
margin z), then on the same context a probe asking which shown frame, if any, contains violating content. The window
keeps z when (z > v and the reply cites a shown frame) or (z <= v and it does not); otherwise the native v.
"""
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
from PIL import Image
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.stance_cache import build,margin,positions
from src.mllm_judge import Judge,MODEL,yesno_question
from src.video_inputs import frame_paths,load_asr,load_manifest,fixed_windows,window_text

IMPLEMENTATION='candidate41 grounded acceptance of adjacent reads;2026-10-10'
EXP=ROOT/'runs/20261010_m1_grounded_adjacent'
DATASETS=('HateMM','HateClipSeg')
SPEC=dict(version='R1-grounded-adjacent-2026-10-10',model=MODEL,native_frames=20,window_seconds=8,seed=0,label='[t={time:.1f}s]\n',
    probe='Among the frames shown for this window, which one, if any, contains content that violates the rules above? Reply with that frame\'s timestamp exactly as labelled, or "none".',
    max_new_tokens=16,match_tolerance=.05,decision='z if (z>v and grounded) or (z<=v and not grounded) else v',
    no_frame_inside='native v, no probe',speech='native S',global_stance='native G and own hard stance',fusion='max(V,S)',decoder='fixed r6_bma',
    scope='HateMM215/HateClipSeg118 test;4fps sole evaluator;fixed r6;development-selected;no GT in reading')
NUMBER=re.compile(r'\d+(?:\.\d+)?')
PREDICTIONS=('base','accept_all','grounded')


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
        code_path='experiments/20261010_m1_grounded_adjacent/grounded.py',
        extra=dict(z_video=ctx['global_margin'],stance=ctx['stance'],prefix_tokens=ctx['prefix_tokens'],stance_cache_tokens=ctx['stance_cache_tokens'],
            stance_cache_logical_start=ctx['stance_cache_logical_start'],windows=ww,standalone_seconds=seconds))


def parse(reply,shown_times):
    numbers=[float(x) for x in NUMBER.findall(reply)]
    cited=[t for t in shown_times if any(abs(x-t)<=SPEC['match_tolerance'] for x in numbers)]
    return dict(numbers=numbers,cited=cited,grounded=bool(cited),says_none='none' in reply.lower(),parse_failure=not numbers and 'none' not in reply.lower())


def tick(j):
    if j.device.type=='cuda':torch.cuda.synchronize()
    return time.perf_counter()


def open_images(paths):
    images=[]
    try:
        for p in paths:
            with Image.open(p) as original:images.append(original.convert('RGB'))
        return images
    except Exception:
        for image in images:image.close()
        raise


@torch.no_grad()
def adjacent_probe(j,cache,ctx,shown,question):
    """shown: [(time, path)] prefix frames inside the window. Returns the adjacent margin, the probe reply and details."""
    n=cache.get_seq_length();assert shown and n==ctx['stance_cache_tokens']
    content=[]
    for t,p in shown:content+=[dict(type='text',text=SPEC['label'].format(time=t)),dict(type='image',image=str(p))]
    content.append(dict(type='text',text=question));conv=ctx['msgs']+ctx['history']+[dict(role='user',content=content)]
    full_u=j.render(conv,False);full_q=j.render(conv,True);full_p=j.render(conv+[j.turn('user',SPEC['probe'])],True)
    assert full_u.startswith(ctx['head']) and full_q.startswith(full_u) and full_p.startswith(full_u)
    text_u=full_u[len(ctx['head']):];header=full_q[len(full_u):];text_p=full_p[len(full_u):]
    assert text_u.count(question)==1 and SPEC['probe'] in text_p and header and all(SPEC['label'].format(time=t) in text_u for t,_ in shown)
    images=open_images([p for _,p in shown])
    try:enc=j.encode(text_u+header,images)
    finally:
        for image in images:image.close()
    ids=enc['input_ids'].to(j.device);grid=enc['image_grid_thw'].to(j.device);assert len(grid)==len(shown)
    counts=[int(torch.tensor(g).prod())//j.processor.image_processor.merge_size**2 for g in grid.tolist()];assert int((ids==j.image_token_id).sum())==sum(counts)
    hid=j.tok(header,add_special_tokens=False)['input_ids'];assert ids[0,-len(hid):].tolist()==hid;L=ids.shape[1]-len(hid)
    pid=j.tok(text_p,add_special_tokens=False)['input_ids'];probe_ids=torch.tensor([pid],device=j.device)
    rel_uh,_=positions(j,ids,grid);rel_up,_=positions(j,torch.cat([ids[:,:L],probe_ids],1),grid)
    assert torch.equal(rel_up[:,:,:L],rel_uh[:,:,:L]);start=ctx['stance_cache_logical_start'];old=j.model.model.rope_deltas.clone()
    first=j.forward_calls;vision=j.vision_calls;gen=[]
    try:
        kwargs=j.model_inputs(enc);kwargs.pop('attention_mask',None)
        out=j.model.model(**kwargs,position_ids=rel_uh+start,past_key_values=cache,use_cache=True)
        z=j.margins_fp32(out.last_hidden_state[0,-1][None].clone())[0];del out
        assert cache.get_seq_length()==n+ids.shape[1];cache.crop(n+L)
        pos_p=rel_up[:,:,L:]+start;assert bool((pos_p[:,0,-1]==pos_p[0,0,-1]).all())
        out=j.model.model(input_ids=probe_ids,position_ids=pos_p,past_key_values=cache,use_cache=True);h=out.last_hidden_state[0,-1];del out
        W=j.model.get_output_embeddings().weight;nxt_pos=int(pos_p[0,0,-1])+1
        for _ in range(SPEC['max_new_tokens']):
            lg=h.float()@W.float().T
            if j.softcap:lg=torch.tanh(lg/j.softcap)*j.softcap
            nxt=int(lg.argmax())
            if nxt in j.eos_ids:break
            gen.append(nxt)
            out=j.model.model(input_ids=torch.tensor([[nxt]],device=j.device),position_ids=torch.full((3,1,1),nxt_pos,device=j.device),past_key_values=cache,use_cache=True)
            h=out.last_hidden_state[0,-1];del out;nxt_pos+=1
        reply=j.tok.decode(gen,skip_special_tokens=True).strip()
        assert j.forward_calls-first==2+len(gen) and j.vision_calls-vision==1
    finally:cache.crop(n);j.model.model.rope_deltas=old
    return z,reply,dict(user_tokens=int(L),probe_tokens=len(pid),generated=len(gen),image_counts=counts,shown_times=[t for t,_ in shown])


@torch.no_grad()
def read(j,row,segments):
    begin=tick(j);first=j.forward_calls;vision=j.vision_calls
    if j.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    native=frame_paths(row['dataset'],row['video_id'],SPEC['native_frames']);assert 0<len(native)<=SPEC['native_frames']
    stamp=tick(j);cache,ctx=build(j,native,segments);times=dict(prefix=tick(j)-stamp,native_visual=0.,native_speech=0.,adjacent_probe=0.)
    windows=windows_for(row,segments);visual=[];speech=[]
    for w in windows:
        stamp=tick(j);visual.append(margin(j,cache,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')));times['native_visual']+=tick(j)-stamp
        if w['body'].strip():
            stamp=tick(j);s=margin(j,cache,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'speech'));times['native_speech']+=tick(j)-stamp
        else:s=None
        speech.append(s)
    traces=[];accept_all=[];final=[];generated=0;covered=0
    for w,v,s in zip(windows,visual,speech):
        shown=[(t,p) for t,p in native if w['start']<=t<w['end']];question=yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')
        trace=dict(i=w['i'],bounds=[w['start'],w['end']],native_visual=v,native_speech=s,shown_times=[t for t,_ in shown])
        if shown:
            stamp=tick(j);z,reply,d=adjacent_probe(j,cache,ctx,shown,question);times['adjacent_probe']+=tick(j)-stamp
            p=parse(reply,d['shown_times']);accepted=(z>v and p['grounded']) or (z<=v and not p['grounded']);f=z if accepted else v
            trace.update(adjacent=z,reply=reply,**p,accepted=accepted,final=f,**d);generated+=d['generated'];covered+=1
        else:z=v;f=v;trace.update(adjacent=None,reply=None,accepted=None,final=v)
        accept_all.append(z);final.append(f);traces.append(trace)
    native_seconds=sum(times[k] for k in ('prefix','native_visual','native_speech'))
    preds=dict(base=prediction(row,ctx,visual,speech,native_seconds,'m1_native'),
        accept_all=prediction(row,ctx,accept_all,speech,native_seconds-times['native_visual']+times['adjacent_probe'],'m1_c41_accept_all'),
        grounded=prediction(row,ctx,final,speech,native_seconds+times['adjacent_probe'],'m1_c41_grounded'))
    checks=dict(GT_read=False,host=socket.gethostname(),times=times,actual_forwards=j.forward_calls-first,actual_vision=j.vision_calls-vision,windows=len(windows),
        covered=covered,generated_tokens=generated,accepted=sum(bool(t['accepted']) for t in traces),grounded=sum(bool(t.get('grounded')) for t in traces),
        says_none=sum(bool(t.get('says_none')) for t in traces),parse_failures=sum(bool(t.get('parse_failure')) for t in traces),
        peak_GiB=torch.cuda.max_memory_allocated()/2**30 if j.device.type=='cuda' else 0.,pipeline_seconds=tick(j)-begin)
    assert checks['actual_forwards']==preds['base']['calls']+2*covered+generated and checks['actual_vision']==1+covered
    return dict(version=SPEC['version'],implementation=IMPLEMENTATION,spec=SPEC,checks=checks,
        native=dict(global_margin=ctx['global_margin'],stance=ctx['stance'],stance_cache_tokens=ctx['stance_cache_tokens'],frame_times=[t for t,_ in native]),
        predictions=preds,traces=traces)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=EXP/('smoke' if a.smoke else 'main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),spec=SPEC,model=MODEL,implementation=IMPLEMENTATION,GT_read=False,smoke=a.smoke,
        torch=torch.__version__,transformers=transformers.__version__,source='experiments/20261010_m1_grounded_adjacent/grounded.py;2026-10-10',command='python -u '+' '.join(sys.argv))
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
                covered=c['covered'],accepted=c['accepted'],grounded=c['grounded'],says_none=c['says_none'],parse_failures=c['parse_failures'],peak_GiB=c['peak_GiB']))
            audit['elapsed_seconds']=time.perf_counter()-start;audit_path.write_text(json.dumps(audit,indent=2)+'\n')
            logging.info('%d/%d %s/%s covered=%d accepted=%d grounded=%d none=%d parse_fail=%d peak=%.2fGiB',ordinal,len(rows),row['dataset'],row['video_id'],
                c['covered'],c['accepted'],c['grounded'],c['says_none'],c['parse_failures'],c['peak_GiB'])
        for name,rr in results.items():
            folder=out/name;folder.mkdir(exist_ok=True);(folder/'config.json').write_text(json.dumps({**cfg,'measurement':name},indent=2)+'\n')
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
        audit['completed']=True;logging.info('DONE coverage=%d',len(rows))
    finally:
        for h in hooks:h.remove()
        audit['elapsed_seconds']=time.perf_counter()-start;audit['actual_forwards']=j.forward_calls if j else 0;audit['actual_vision']=j.vision_calls if j else 0
        audit_path.write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
