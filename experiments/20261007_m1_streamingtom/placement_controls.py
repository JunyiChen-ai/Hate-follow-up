"""Separate "more frames inside the window" from "frames placed next to the question" (user question 2026-10-09).

Every arm shares the fresh native prefix of r6 (20 frames + full ASR + rules), r6's G, own stance and S of each
video. Only the visual read of windows that have the arm's frames changes; other windows keep the native V.
- base: r6's native V (must equal r6 exactly).
- local_clean_replay: E of local_controls (the window's LOCAL 0.5 fps frames with "Actual LOCAL source at ..."
  labels right before the question) re-read through the model's own forward with fresh vision; checks this
  job's suffix path against the stored E.
- adjacent_local: the same LOCAL frames right before the question, labelled like the prefix frames ("[t=12.0s]").
- prefix_local: the same LOCAL frames inserted into the prefix frame list in time order with "[t=12.0s]" labels;
  the question is r6's plain question. One standalone forward per window (prefix with 20 + k frames, transcript,
  rules, whole-video question, native stance answer, window question); no shared cache.
- adjacent_native: the native prefix frames whose time falls inside the window are shown again right before the
  question with their "[t=..s]" labels; no new frame.
- standalone_native (diagnostic, one window per video): r6's exact input as one standalone forward; its
  difference from the cached V is the numeric floor of the no-cache path used by prefix_local.
"""
import argparse
import json
import logging
import os
import socket
import sys
import time
from PIL import Image
import torch
from inputs import ROOT,SPEC,DATASETS,selected_rows,frames_for,windows_for,local_ids
from collect import tick
from measure import prediction
from src.stance_cache import build,margin,positions
from src.mllm_judge import Judge,MODEL,VIDEO_QUESTION,yesno_question
from src.video_inputs import frame_paths,load_asr

IMPLEMENTATION='placement controls local_clean_replay/adjacent_local/prefix_local/adjacent_native;2026-10-09'
EXP=ROOT/'runs/20261007_m1_streamingtom'
LOCAL_RUN=EXP/'local_controls_main'
ARMS=('local_clean_replay','adjacent_local','prefix_local','adjacent_native')
LOCAL_LABEL='Actual LOCAL source at {time:.3f} seconds.'
NATIVE_LABEL='[t={time:.1f}s]\n'


def native_in_window(native,window):
    return [(t,p) for t,p in native if window['start']<=t<window['end']]


def open_images(paths):
    images=[]
    try:
        for p in paths:
            with Image.open(p) as original:images.append(original.convert('RGB'))
        return images
    except Exception:
        for image in images:image.close()
        raise


def forward_margin(j,enc,position_ids=None,cache=None):
    """Model's own forward (vision + placement + DeepStack inside); Yes/No margin at the last row."""
    kwargs=j.model_inputs(enc)
    if cache is not None:kwargs.pop('attention_mask',None)  # the processor's mask covers the suffix only; the model builds the causal mask over cache + suffix
    if position_ids is not None:kwargs['position_ids']=position_ids
    out=j.model.model(**kwargs,past_key_values=cache,use_cache=cache is not None)
    hidden=out.last_hidden_state[0,-1].clone();del out
    return j.margins_fp32(hidden[None])[0]


@torch.no_grad()
def suffix_margin(j,cache,ctx,labelled,question):
    """Images with their text labels, then the question, as one user turn on the shared native cache."""
    assert labelled and cache.get_seq_length()==ctx['stance_cache_tokens']
    content=[]
    for label,path in labelled:content+=[dict(type='text',text=label),dict(type='image',image=str(path))]
    content.append(dict(type='text',text=question))
    full=j.render(ctx['msgs']+ctx['history']+[dict(role='user',content=content)],True);assert full.startswith(ctx['head']);text=full[len(ctx['head']):]
    assert all(label in text for label,_ in labelled) and text.count(question)==1
    images=open_images([p for _,p in labelled])
    try:enc=j.encode(text,images)
    finally:
        for image in images:image.close()
    ids=enc['input_ids'].to(j.device);grid=enc['image_grid_thw'].to(j.device);assert len(grid)==len(labelled)
    counts=[int(torch.tensor(g).prod())//j.processor.image_processor.merge_size**2 for g in grid.tolist()]
    assert int((ids==j.image_token_id).sum())==sum(counts)
    relative,_=positions(j,ids,grid);n=cache.get_seq_length();old=j.model.model.rope_deltas.clone();first=j.forward_calls;vision=j.vision_calls
    try:
        z=forward_margin(j,enc,relative+ctx['stance_cache_logical_start'],cache)
        assert j.forward_calls-first==1 and j.vision_calls-vision==1
        return z,dict(suffix_tokens=int(ids.shape[1]),image_counts=counts,labels=[label for label,_ in labelled])
    finally:cache.crop(n);j.model.model.rope_deltas=old


@torch.no_grad()
def standalone_margin(j,frames,segments,history,question):
    """r6's conversation rebuilt from scratch with `frames` as the prefix frame list; one full forward, no cache."""
    msgs,files=j.prefix_messages(frames,segments);text=j.render(j.conv(msgs,question,history),True)
    assert text.count(question)==1 and text.count(VIDEO_QUESTION)==1
    images=open_images(files)
    try:enc=j.encode(text,images)
    finally:
        for image in images:image.close()
    grid=enc['image_grid_thw'];assert len(grid)==len(files)
    counts=[int(torch.tensor(g).prod())//j.processor.image_processor.merge_size**2 for g in grid.tolist()]
    assert int((enc['input_ids'][0]==j.image_token_id).sum())==sum(counts)
    j.model.model.rope_deltas=None;first=j.forward_calls;vision=j.vision_calls
    z=forward_margin(j,enc);assert j.forward_calls-first==1 and j.vision_calls-vision==1
    return z,dict(tokens=int(enc['input_ids'].shape[1]),images=len(files),image_counts=counts,input_ids=enc['input_ids'][0].tolist())


@torch.no_grad()
def read(j,row,segments,reference):
    begin=tick(j);first=j.forward_calls;vision=j.vision_calls
    if j.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    _,frames=frames_for(row);native=frame_paths(row['dataset'],row['video_id'],SPEC['native_frames']);assert len(native)==SPEC['native_frames']
    stamp=tick(j);cache,ctx=build(j,native,segments)
    times=dict(prefix=tick(j)-stamp,native_visual=0.,native_speech=0.,standalone_native=0.,**{a:0. for a in ARMS})
    windows=windows_for(row,segments);visual=[];speech=[]
    for w in windows:
        stamp=tick(j);visual.append(margin(j,cache,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')));times['native_visual']+=tick(j)-stamp
        if w['body'].strip():
            stamp=tick(j);s=margin(j,cache,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'speech'));times['native_speech']+=tick(j)-stamp
        else:s=None
        speech.append(s)
    if reference is not None:
        assert reference['frames']==frames and len(reference['traces'])==len(windows)
    # Diagnostic: r6's exact input for the first covered window as one standalone forward.
    probe=next((w for w in windows if local_ids(frames,w)),windows[0]);question=yesno_question(probe['i'],len(windows),probe['start'],probe['end'],probe['body'],'visual')
    stamp=tick(j);z,details=standalone_margin(j,native,segments,ctx['history'],question);times['standalone_native']=tick(j)-stamp
    ids,_=j.branch_ids(ctx['msgs'],question,ctx['history'],head_text=ctx['head'])
    diagnostic=dict(i=probe['i'],cached_visual=visual[probe['i']],standalone_visual=z,abs_difference=abs(z-visual[probe['i']]),
        tokens=details['tokens'],cached_tokens=ctx['stance_cache_tokens']+len(ids),tokens_equal=details['tokens']==ctx['stance_cache_tokens']+len(ids),
        branch_ids_equal=details['input_ids'][-len(ids):]==ids,images=details['images'],image_counts_equal=details['image_counts']==ctx['image_counts'])
    j.model.model.rope_deltas=ctx['rope'].clone()
    values={a:[] for a in ARMS};traces=[];counts=dict(covered_local=0,covered_native=0)
    for w,v,s in zip(windows,visual,speech):
        local=local_ids(frames,w);inside=native_in_window(native,w);question=yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')
        trace=dict(i=w['i'],bounds=[w['start'],w['end']],LOCAL=local,native_inside=[t for t,_ in inside],native_visual=v,native_speech=s)
        if reference is not None:
            old=reference['traces'][w['i']];assert old['i']==w['i'] and old['LOCAL']==local and old['native_visual']==v
            trace['stored_local_clean']=old['local_clean']['z'] if old['local_clean'] else None
        counts['covered_local']+=bool(local);counts['covered_native']+=bool(inside)
        for arm in ARMS:
            stamp=tick(j)
            if arm=='local_clean_replay' and local:
                z,d=suffix_margin(j,cache,ctx,[(LOCAL_LABEL.format(time=frames[i]['time']),ROOT/frames[i]['path']) for i in local],question)
            elif arm=='adjacent_local' and local:
                z,d=suffix_margin(j,cache,ctx,[(NATIVE_LABEL.format(time=frames[i]['time']),ROOT/frames[i]['path']) for i in local],question)
            elif arm=='adjacent_native' and inside:
                z,d=suffix_margin(j,cache,ctx,[(NATIVE_LABEL.format(time=t),p) for t,p in inside],question)
            elif arm=='prefix_local' and local:
                merged=sorted([(t,p,0) for t,p in native]+[(frames[i]['time'],ROOT/frames[i]['path'],1) for i in local],key=lambda x:(x[0],x[2]))
                z,d=standalone_margin(j,[(t,p) for t,p,_ in merged],segments,ctx['history'],question);d.pop('input_ids')
                d['inserted_positions']=[k for k,(_,_,kind) in enumerate(merged) if kind]
                j.model.model.rope_deltas=ctx['rope'].clone()
            else:z,d=v,None
            times[arm]+=tick(j)-stamp;values[arm].append(z);trace[arm]=dict(z=z,**d) if d else None
        if trace.get('stored_local_clean') is not None:trace['replay_abs']=abs(trace['local_clean_replay']['z']-trace['stored_local_clean'])
        traces.append(trace)
    base=prediction(row,ctx,visual,speech,sum(times[k] for k in ('prefix','native_visual','native_speech')),'m1_native')
    predictions={}
    for a in ARMS:
        p=prediction(row,ctx,values[a],speech,sum(times[k] for k in ('prefix','native_speech',a)),'m1_place_'+a)
        p['code_path']='experiments/20261007_m1_streamingtom/placement_controls.py';predictions[a]=p
    checks=dict(GT_read=False,host=socket.gethostname(),times=times,actual_forwards=j.forward_calls-first,actual_vision=j.vision_calls-vision,**counts,
        windows=len(windows),peak_GiB=torch.cuda.max_memory_allocated()/2**30 if j.device.type=='cuda' else 0.,pipeline_seconds=tick(j)-begin,
        vision_scope='fresh vision inside every arm forward; the native prefix vision is paid once per video')
    extra=3*counts['covered_local']+counts['covered_native']+1
    assert checks['actual_forwards']==base['calls']+extra and checks['actual_vision']==1+extra
    return dict(version=SPEC['version'],implementation=IMPLEMENTATION,arms=list(ARMS),spec=SPEC,checks=checks,diagnostic=diagnostic,
        native=dict(global_margin=ctx['global_margin'],stance=ctx['stance'],stance_cache_tokens=ctx['stance_cache_tokens'],frame_times=[t for t,_ in native]),
        base=base,predictions=predictions,frames=frames,traces=traces)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=EXP/f"placement_controls_{'smoke' if a.smoke else 'main'}";out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import av,transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),spec=SPEC,model=MODEL,implementation=IMPLEMENTATION,arms=list(ARMS),
        GT_read=False,smoke=a.smoke,reference=str(LOCAL_RUN.relative_to(ROOT)),torch=torch.__version__,transformers=transformers.__version__,av=av.__version__,
        source='experiments/20261007_m1_streamingtom/{placement_controls,measure}.py;2026-10-09',command='python -u '+' '.join(sys.argv))
    number=1
    while (out/f'pipeline_attempt_{number:04d}.json').exists():number+=1
    audit_path=out/f'pipeline_attempt_{number:04d}.json';audit=dict(host=cfg['host'],completed=False,GT_read=False,videos=[]);start=time.perf_counter();j=None;hooks=[]
    results={name:[] for name in ('base',*ARMS)}
    try:
        path=out/'config.json'
        if path.exists():
            old=json.loads(path.read_text());assert all(old[k]==cfg[k] for k in cfg if k not in ('date','command'))
        else:path.write_text(json.dumps(cfg,indent=2)+'\n')
        asr={ds:load_asr(ds) for ds in DATASETS};stamp=time.perf_counter();torch.manual_seed(0);torch.set_num_threads(4);j=Judge(MODEL)
        audit['model_load_seconds']=time.perf_counter()-stamp;j.forward_calls=j.vision_calls=0
        hooks=[j.model.model.language_model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),
            j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
        rows=selected_rows(a.smoke)
        for ordinal,row in enumerate(rows,1):
            stamp=time.perf_counter();first=j.forward_calls
            path=out/'records'/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True);reused=path.exists()
            segments=asr[row['dataset']].get(row['video_id'],[])
            if reused:bundle=json.loads(path.read_text())
            else:
                reference=json.loads((LOCAL_RUN/'records'/row['dataset']/(row['video_id']+'.json')).read_text())
                bundle=read(j,row,segments,reference);del reference
                partial=path.with_suffix('.partial');partial.write_text(json.dumps(bundle,allow_nan=False)+'\n');partial.replace(path)
            results['base'].append(bundle['base'])
            for name in ARMS:results[name].append(bundle['predictions'][name])
            replay=max((t['replay_abs'] for t in bundle['traces'] if 'replay_abs' in t),default=0.)
            audit['videos'].append(dict(dataset=row['dataset'],video_id=row['video_id'],reused=reused,elapsed_seconds=time.perf_counter()-stamp,
                current_actual_forwards=j.forward_calls-first,replay_max_abs=replay,standalone_native_abs=bundle['diagnostic']['abs_difference'],
                peak_GiB=bundle['checks']['peak_GiB']))
            audit['elapsed_seconds']=time.perf_counter()-start;audit_path.write_text(json.dumps(audit,indent=2)+'\n')
            logging.info('%d/%d %s/%s replay_max_abs=%.6f standalone_native_abs=%.6f peak=%.2fGiB',ordinal,len(rows),row['dataset'],row['video_id'],
                replay,bundle['diagnostic']['abs_difference'],bundle['checks']['peak_GiB'])
        for name,rr in results.items():
            folder=out/name;folder.mkdir(exist_ok=True);(folder/'config.json').write_text(json.dumps({**cfg,'measurement':name},indent=2)+'\n')
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
        audit['completed']=True;logging.info('DONE coverage=%d',len(rows))
    finally:
        for h in hooks:h.remove()
        audit['elapsed_seconds']=time.perf_counter()-start;audit['actual_forwards']=j.forward_calls if j else 0;audit['actual_vision']=j.vision_calls if j else 0
        audit_path.write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
