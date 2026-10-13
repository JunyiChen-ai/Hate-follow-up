"""M1 candidate 42 (declared 2026-10-13, README of this directory): evidence-bound pairwise comparison of windows.

r6's reads (prefix cache, G, own stance, V and S per window) are taken exactly as r6. In addition, pairs of windows of
the same video are put to the model as one A/B/C question on the same cache, with the LOCAL 0.5 fps frames of BOTH
windows shown right before the question (the placement finding of 2026-10-10: the model cannot find frames by their
timestamps, so the frames must be shown). Each edge of a fixed sparse graph is asked in both orders. Nothing here reads
labels; compare_analyze.py aggregates the comparisons into per-window values offline and runs the sole evaluator."""
import argparse
import json
import logging
import math
import os
import socket
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.stance_cache import build,margin,positions
from src.mllm_judge import Judge,MODEL,yesno_question
from src.video_inputs import frame_paths,load_asr,load_manifest,fixed_windows,window_text

EXP=ROOT/'runs/20261013_m1_pairwise_evidence'
PLACEMENT=ROOT/'runs/20261007_m1_streamingtom/placement_controls_main'
DATASETS=('HateMM','HateClipSeg')
CHOICES=('A','B','C')
SPEC=dict(version='C42-pairwise-evidence-2026-10-13',model=MODEL,native_frames=20,window_seconds=8,seed=0,ring_distances=[1,2],complete_graph_max_windows=4,
    label='[t={time:.1f}s]\n',
    header='Window {letter}, from {start:.1f}s to {end:.1f}s of this video. Frames inside window {letter}:\n',
    header_empty='Window {letter}, from {start:.1f}s to {end:.1f}s of this video. (No frames are available for this window.)\n',
    question=('Consider the two windows above, A and B. Judge the visual content alone (imagery, gestures, symbols, on-screen text), '
              'ignoring the speech. Use the rest of the video only to resolve who or what is being referred to.\n\n'
              'Which window more clearly shows visual content that violates the above rules?\n\n'
              'Answer "A", "B", or "C", where C means they are equal or there is not enough evidence.'),
    graph='ring over a seeded permutation of the windows (default_rng([seed, video_index])), ring distances 1 and 2 (2N edges, degree 4); complete graph when N <= 4; every edge asked in both orders',
    readout='d = log((pA + pC/2) / (pB + pC/2)) over the softmax restricted to the A/B/C first tokens; l_ij = (d(i as A, j as B) - d(j as A, i as B)) / 2',
    frames='the stored LOCAL 0.5 fps frames of placement_controls_main (data/temporal_retrieved_kv), all frames inside the window, labelled like the prefix frames',
    reads='native V and S on the prefix cache exactly as r6 (bound to the placement records); each comparison is one user turn on the same cache, cropped afterwards',
    scope='HateMM215/HateClipSeg118 test;4fps sole evaluator;fixed r6;development-selected;no GT in reading')
IMPLEMENTATION=('compare.py 2026-10-13: margin() for V and S as r6; comparisons as one user turn with labelled images of both windows and the question, '
    'model forward on the shared cache with explicit 3D positions (as placement suffix_margin), A/B/C log-probs from choice_logprobs; cache cropped and rope_deltas restored after each; forward and vision counts asserted')


def selected_rows(smoke=False,datasets=DATASETS):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    assert len(rows)==333
    indexed=[(vi,r) for vi,r in enumerate(rows)]  # video_index over the full 333, independent of the --datasets filter
    if smoke:
        indexed=[(vi,r) for ds in DATASETS for vi,r in [x for x in indexed if x[1]['dataset']==ds][:2]]+[(vi,r) for vi,r in indexed if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
        assert len(indexed)==5
    return [(vi,r) for vi,r in indexed if r['dataset'] in datasets]


def windows_for(row,segments):
    return [dict(i=i,start=a,end=b,body=window_text(segments,a,b)) for i,(a,b) in enumerate(fixed_windows(float(row['duration']),SPEC['window_seconds']))]


def edges_for(n,vi):
    """Fixed sparse comparison graph of a video with n windows; label-free and reproducible."""
    if n<=1:return []
    if n<=SPEC['complete_graph_max_windows']:return [(i,j) for i in range(n) for j in range(i+1,n)]
    perm=np.random.default_rng([SPEC['seed'],vi]).permutation(n).tolist();edges=set()
    for k in range(n):
        for d in SPEC['ring_distances']:
            a,b=perm[k],perm[(k+d)%n];edges.add((min(a,b),max(a,b)))
    return sorted(edges)


def prediction(row,ctx,visual,speech,seconds,name):
    ww=[]
    for w,v,s in zip(windows_for(row,[]),visual,speech):
        one=dict(i=w['i'],start=w['start'],end=w['end'],z_visual=v,z=max(v,s) if s is not None else v)
        if s is not None:one['z_speech']=s
        ww.append(one)
    idx=np.clip(((np.arange(math.ceil(float(row['duration'])*4))+.5)/4//SPEC['window_seconds']).astype(int),0,len(ww)-1)
    return dict(schema_version=1,method=name,dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),native_rate=4,
        score_curve=np.asarray([w['z'] for w in ww])[idx].tolist(),intervals=[],error=None,seed=0,calls=3+len(ww)+sum(s is not None for s in speech),
        code_path='experiments/20261013_m1_pairwise_evidence/compare.py',
        extra=dict(z_video=ctx['global_margin'],stance=ctx['stance'],prefix_tokens=ctx['prefix_tokens'],stance_cache_tokens=ctx['stance_cache_tokens'],
            stance_cache_logical_start=ctx['stance_cache_logical_start'],windows=ww,standalone_seconds=seconds))


def d_from_logprobs(lp):
    p=np.exp(np.asarray(lp,dtype=float));p=p/p.sum()
    return float(np.log((p[0]+p[2]/2)/(p[1]+p[2]/2))),float(p[2])


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
def compare(j,cache,ctx,choices,first,second):
    """One A/B/C comparison (first window shown as A, second as B) as one user turn on the shared native cache.
    Returns (log-probs aligned with CHOICES, details). Cache and rope_deltas are restored afterwards."""
    n=cache.get_seq_length();assert n==ctx['stance_cache_tokens']
    content=[];paths=[]
    for letter,seg in (('A',first),('B',second)):
        if seg['frames']:
            content.append(dict(type='text',text=SPEC['header'].format(letter=letter,start=seg['start'],end=seg['end'])))
            for t,p in seg['frames']:content+=[dict(type='text',text=SPEC['label'].format(time=t)),dict(type='image',image=str(p))];paths.append(p)
        else:content.append(dict(type='text',text=SPEC['header_empty'].format(letter=letter,start=seg['start'],end=seg['end'])))
    content.append(dict(type='text',text=SPEC['question']))
    full=j.render(ctx['msgs']+ctx['history']+[dict(role='user',content=content)],True);assert full.startswith(ctx['head']);text=full[len(ctx['head']):]
    assert text.count(SPEC['question'])==1 and text.count('Window A,')==1 and text.count('Window B,')==1
    j.model.model.rope_deltas=ctx['rope'].clone();first_call=j.forward_calls;vision=j.vision_calls
    try:
        if not paths:
            ids=j.tok(text,add_special_tokens=False)['input_ids'];lp=j.cached_choices(cache,ids,choices,in_place=True)
            assert j.forward_calls-first_call==1 and j.vision_calls==vision
            return lp,dict(suffix_tokens=len(ids),images=0,image_counts=[])
        images=open_images(paths)
        try:enc=j.encode(text,images)
        finally:
            for image in images:image.close()
        ids=enc['input_ids'].to(j.device);grid=enc['image_grid_thw'].to(j.device);assert len(grid)==len(paths)
        counts=[int(torch.tensor(g).prod())//j.processor.image_processor.merge_size**2 for g in grid.tolist()]
        assert int((ids==j.image_token_id).sum())==sum(counts)
        relative,_=positions(j,ids,grid)
        kwargs=j.model_inputs(enc);kwargs.pop('attention_mask',None);kwargs['position_ids']=relative+ctx['stance_cache_logical_start']
        out=j.model.model(**kwargs,past_key_values=cache,use_cache=True);hidden=out.last_hidden_state[0,-1].clone();del out
        lp=j.choice_logprobs(hidden,choices)
        assert j.forward_calls-first_call==1 and j.vision_calls-vision==1
        return lp,dict(suffix_tokens=int(ids.shape[1]),images=len(paths),image_counts=counts)
    finally:cache.crop(n);j.model.model.rope_deltas=ctx['rope'].clone()


def placement_record(row):
    rec=json.loads((PLACEMENT/'records'/row['dataset']/(row['video_id']+'.json')).read_text())
    assert rec['checks']['GT_read'] is False and len(rec['frames'])>0
    return rec


@torch.no_grad()
def read(j,vi,row,segments,choices):
    begin=tick(j);first=j.forward_calls;vision=j.vision_calls
    if j.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    native=frame_paths(row['dataset'],row['video_id'],SPEC['native_frames']);assert 0<len(native)<=SPEC['native_frames']
    stamp=tick(j);cache,ctx=build(j,native,segments);times=dict(prefix=tick(j)-stamp,native_visual=0.,native_speech=0.,compare=0.)
    windows=windows_for(row,segments);visual=[];speech=[]
    for w in windows:
        stamp=tick(j);visual.append(margin(j,cache,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')));times['native_visual']+=tick(j)-stamp
        if w['body'].strip():
            stamp=tick(j);s=margin(j,cache,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'speech'));times['native_speech']+=tick(j)-stamp
        else:s=None
        speech.append(s)
    rec=placement_record(row);frames=rec['frames'];assert len(rec['traces'])==len(windows)
    segs=[]
    for w,v,t in zip(windows,visual,rec['traces']):
        assert t['i']==w['i'] and t['bounds']==[w['start'],w['end']]
        assert t['native_visual']==v,f"native V differs from the placement record in window {w['i']}: {v} vs {t['native_visual']}"
        local=[i for i,f in enumerate(frames) if w['start']<=f['time']<w['end']];assert local==t['LOCAL']
        segs.append(dict(i=w['i'],start=w['start'],end=w['end'],frames=[(frames[i]['time'],ROOT/frames[i]['path']) for i in local]))
    edges=edges_for(len(windows),vi);records=[];with_images=0
    for i,k in edges:
        stamp=tick(j)
        lp_ij,d1=compare(j,cache,ctx,choices,segs[i],segs[k]);lp_ji,d2=compare(j,cache,ctx,choices,segs[k],segs[i])
        times['compare']+=tick(j)-stamp;with_images+=bool(d1['images'])+bool(d2['images'])
        d_ij,pc_ij=d_from_logprobs(lp_ij);d_ji,pc_ji=d_from_logprobs(lp_ji)
        records.append(dict(i=i,j=k,lp_ij=lp_ij,lp_ji=lp_ji,d_ij=d_ij,d_ji=d_ji,l_ij=(d_ij-d_ji)/2,s_ij=(d_ij+d_ji)/2,pC_ij=pc_ij,pC_ji=pc_ji,
            images=[d1['images'],d2['images']],suffix_tokens=[d1['suffix_tokens'],d2['suffix_tokens']]))
    native_seconds=sum(times[k] for k in ('prefix','native_visual','native_speech'))
    preds=dict(base=prediction(row,ctx,visual,speech,native_seconds,'m1_native'))
    swap=sum(np.sign(e['d_ij'])==-np.sign(e['d_ji']) for e in records if e['d_ij']!=0 and e['d_ji']!=0)
    checks=dict(GT_read=False,host=socket.gethostname(),video_index=vi,times=times,actual_forwards=j.forward_calls-first,actual_vision=j.vision_calls-vision,windows=len(windows),
        edges=len(records),comparisons_with_images=with_images,swap_agreements=int(swap),mean_pC=float(np.mean([x for e in records for x in (e['pC_ij'],e['pC_ji'])])) if records else None,
        covered_local=sum(bool(s['frames']) for s in segs),peak_GiB=torch.cuda.max_memory_allocated()/2**30 if j.device.type=='cuda' else 0.,pipeline_seconds=tick(j)-begin)
    assert checks['actual_forwards']==preds['base']['calls']+2*len(records) and checks['actual_vision']==1+with_images
    return dict(version=SPEC['version'],implementation=IMPLEMENTATION,spec=SPEC,checks=checks,
        native=dict(global_margin=ctx['global_margin'],stance=ctx['stance'],stance_cache_tokens=ctx['stance_cache_tokens'],frame_times=[t for t,_ in native]),
        windows=[dict(i=s['i'],bounds=[s['start'],s['end']],LOCAL=[t for t,_ in s['frames']],native_visual=v,native_speech=sp) for s,v,sp in zip(segs,visual,speech)],
        edges=records,predictions=preds)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--datasets',nargs='+',default=list(DATASETS),choices=DATASETS);a=ap.parse_args()
    out=EXP/('smoke' if a.smoke else 'main');out.mkdir(parents=True,exist_ok=True);tag='_'.join(a.datasets)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/f'run_{tag}.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/f'run_{tag}.pid').write_text(str(os.getpid()))
    import transformers
    cfg=dict(spec=SPEC,model=MODEL,implementation=IMPLEMENTATION,GT_read=False,smoke=a.smoke,placement=str(PLACEMENT.relative_to(ROOT)),
        torch=torch.__version__,transformers=transformers.__version__,source='experiments/20261013_m1_pairwise_evidence/compare.py;2026-10-13')
    number=1
    while (out/f'pipeline_attempt_{tag}_{number:04d}.json').exists():number+=1
    audit_path=out/f'pipeline_attempt_{tag}_{number:04d}.json';audit=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),datasets=a.datasets,command='python -u '+' '.join(sys.argv),completed=False,GT_read=False,videos=[])
    start=time.perf_counter();j=None;hooks=[];base=[]
    try:
        path=out/'config.json'
        if path.exists():assert json.loads(path.read_text())==cfg,'config differs from the existing run directory'
        else:path.write_text(json.dumps(cfg,indent=2)+'\n')
        asr={ds:load_asr(ds) for ds in a.datasets};stamp=time.perf_counter();torch.manual_seed(SPEC['seed']);torch.set_num_threads(4);j=Judge(MODEL)
        audit['model_load_seconds']=time.perf_counter()-stamp;j.forward_calls=j.vision_calls=0
        hooks=[j.model.model.language_model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),
            j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
        choices=[j.label_ids(c) for c in CHOICES];flat=[t for c in choices for t in c];assert len(flat)==len(set(flat)),f'A/B/C token sets overlap: {choices}'
        audit['choice_ids']=dict(zip(CHOICES,choices));logging.info('choice ids %s',audit['choice_ids'])
        rows=selected_rows(a.smoke,a.datasets)
        for ordinal,(vi,row) in enumerate(rows,1):
            stamp=time.perf_counter();first=j.forward_calls
            path=out/'records'/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True);reused=path.exists()
            if reused:bundle=json.loads(path.read_text())
            else:
                bundle=read(j,vi,row,asr[row['dataset']].get(row['video_id'],[]),choices)
                partial=path.with_suffix('.partial');partial.write_text(json.dumps(bundle,allow_nan=False)+'\n');partial.replace(path)
            base.append(bundle['predictions']['base']);c=bundle['checks']
            audit['videos'].append(dict(dataset=row['dataset'],video_id=row['video_id'],video_index=vi,reused=reused,elapsed_seconds=time.perf_counter()-stamp,current_actual_forwards=j.forward_calls-first,
                windows=c['windows'],edges=c['edges'],swap_agreements=c['swap_agreements'],mean_pC=c['mean_pC'],compare_seconds=c['times']['compare'],peak_GiB=c['peak_GiB']))
            audit['elapsed_seconds']=time.perf_counter()-start;audit_path.write_text(json.dumps(audit,indent=2)+'\n')
            logging.info('%d/%d %s/%s windows=%d edges=%d swap=%d meanpC=%s compare=%.1fs peak=%.2fGiB',ordinal,len(rows),row['dataset'],row['video_id'],c['windows'],c['edges'],c['swap_agreements'],
                None if c['mean_pC'] is None else round(c['mean_pC'],3),c['times']['compare'],c['peak_GiB'])
        folder=out/'base';folder.mkdir(exist_ok=True);(folder/f'predictions_{tag}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in base))
        audit['completed']=True;logging.info('DONE coverage=%d',len(rows))
    finally:
        for h in hooks:h.remove()
        audit['elapsed_seconds']=time.perf_counter()-start;audit['actual_forwards']=j.forward_calls if j else 0;audit['actual_vision']=j.vision_calls if j else 0
        audit_path.write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
