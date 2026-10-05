#!/usr/bin/env python3
"""Independent native and TTF visual readings; no annotation access."""
import argparse
import copy
import json
import logging
import math
import os
from pathlib import Path
import socket
import time
import numpy as np
import torch
from ttf import ROOT,SPEC,select,capture_native,layout,prefill,dense_plan
from src.mllm_judge import Judge,MODEL,yesno_question
from src.stance_cache import build,margin
from src.video_inputs import load_manifest,load_asr,frame_paths,fixed_windows,window_text

DATASETS=('HateMM','HateClipSeg')
CACHE=ROOT/'data/temporal_token_fusion'


def selected_rows(smoke=False):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def tick():
    torch.cuda.synchronize();return time.perf_counter()


def prediction(row,ctx,visual,speech,seconds,method):
    wins=fixed_windows(float(row['duration']),8);windows=[]
    for i,((a,b),v,s) in enumerate(zip(wins,visual,speech)):
        w=dict(i=i,start=a,end=b,z_visual=v,z=max(v,s) if s is not None else v)
        if s is not None:w['z_speech']=s
        windows.append(w)
    idx=np.clip(((np.arange(math.ceil(float(row['duration'])*4))+.5)/4//8).astype(int),0,len(wins)-1)
    return dict(schema_version=1,method=method,dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),native_rate=4,
        score_curve=np.asarray([w['z'] for w in windows])[idx].tolist(),intervals=[],error=None,seed=0,
        calls=(6 if method=='m1_ttf' else 3)+len(wins)+sum(s is not None for s in speech),code_path='experiments/20261005_m1_ttf/measure.py',
        extra=dict(z_video=ctx['global_margin'],stance=ctx['stance'],prefix_tokens=ctx['prefix_tokens'],
            stance_cache_tokens=ctx['stance_cache_tokens'],stance_cache_logical_start=ctx['stance_cache_logical_start'],windows=windows,standalone_seconds=seconds))


@torch.no_grad()
def read_video(j,row,segments,smoke):
    frames=frame_paths(row['dataset'],row['video_id'],20);wins=fixed_windows(float(row['duration']),8)
    before=j.language_calls;before_vision=j.vision_calls;torch.cuda.reset_peak_memory_stats();start=tick()
    with capture_native(j) as captured:cache,ctx=build(j,frames,segments)
    times=dict(native_prefix=tick()-start,native_visual=0.,native_speech=0.,matching=0.,layout=0.,new_prefix=0.,new_visual=0.,diagnostic=0.)
    native=dict(visual=[],speech=[]);new_visual=[];traces=[];repeat_checked=set();diagnostic_calls=0
    for i,(a,b) in enumerate(wins):
        body=window_text(segments,a,b);trace=dict(i=i,start=a,end=b,body=body,native={},optimized={})
        for kind in ('visual','speech'):
            if kind=='speech' and not body.strip():native[kind].append(None);continue
            q=yesno_question(i,len(wins),a,b,body,kind);start=tick();z=margin(j,cache,ctx,q)
            times['native_'+kind]+=tick()-start;native[kind].append(z);trace['native'][kind]=dict(question=q,margin=z)
        traces.append(trace)
    del cache
    start=tick();cpu_features=captured['features'].cpu();merge=j.model.config.vision_config.spatial_merge_size
    plan=select(cpu_features,captured['encoded']['image_grid_thw'],merge);times['matching']=tick()-start
    # Feature cache is an audit input only: measurement always captures this video's current native forward.
    folder=CACHE/row['dataset']/row['video_id'];folder.mkdir(parents=True,exist_ok=True)
    feature_file=folder/'projector.pt'
    if feature_file.exists():assert torch.equal(torch.load(feature_file,map_location='cpu',weights_only=True),cpu_features)
    else:torch.save(cpu_features,feature_file)
    metadata=dict(version=SPEC['version'],model=MODEL,dataset=row['dataset'],video_id=row['video_id'],host=socket.gethostname(),
        frames=[dict(time=t,path=str(p.relative_to(ROOT))) for t,p in frames],segments=[list(s) for s in segments],
        grid_thw=captured['encoded']['image_grid_thw'].tolist(),merge=merge,feature_shape=list(cpu_features.shape),dtype=str(cpu_features.dtype),GT_read=False)
    metadata_file=folder/'metadata.json'
    if metadata_file.exists():
        previous=json.loads(metadata_file.read_text())
        assert {k:v for k,v in previous.items() if k!='host'}=={k:v for k,v in metadata.items() if k!='host'}
        metadata=previous  # Preserve first acquisition host; don't overwrite smoke provenance on reuse.
    else:metadata_file.write_text(json.dumps(metadata)+'\n')
    dense_check=None
    if smoke:
        start=tick();dense=dense_plan(plan);msgs,text,packed,evidence=layout(j,frames,segments,captured,dense,native_order=True)
        assert evidence['packed_ids']==evidence['original_prefix_ids']
        check_cache,check_ctx=prefill(j,msgs,text,packed,ctx);diagnostic_calls+=3
        q=traces[0]['native']['visual']['question'];z=margin(j,check_cache,check_ctx,q);diagnostic_calls+=1
        assert z==native['visual'][0],(row['video_id'],'dense native-order manual != native',z,native['visual'][0])
        dense_check=dict(native_order_exact=True,margin=z,positions=evidence['packed_positions'],native_margin=native['visual'][0])
        del check_cache,packed;times['diagnostic']+=tick()-start
    start=tick();msgs,text,packed,evidence=layout(j,frames,segments,captured,plan);times['layout']=tick()-start
    start=tick();new_cache,new_ctx=prefill(j,msgs,text,packed,ctx);times['new_prefix']=tick()-start
    del packed,captured
    for i,((a,b),trace) in enumerate(zip(wins,traces)):
        q=trace['native']['visual']['question'];ids,suffix=j.branch_ids(msgs,q,new_ctx['history'],head_text=new_ctx['head'])
        start=tick();z=margin(j,new_cache,new_ctx,q);times['new_visual']+=tick()-start;new_visual.append(z)
        t=dict(question=q,suffix_ids=ids,suffix_text=suffix,margin=z,cache_restored=new_cache.get_seq_length()==new_ctx['stance_cache_tokens'],rope_restored=torch.equal(j.model.model.rope_deltas,new_ctx['rope']))
        assert t['cache_restored'] and t['rope_restored']
        if smoke and 'visual' not in repeat_checked:
            start=tick();clone=copy.deepcopy(new_cache)
            replay=margin(j,clone,new_ctx,q);assert replay==z and clone.get_seq_length()==new_ctx['stance_cache_tokens']
            del clone;times['diagnostic']+=tick()-start;diagnostic_calls+=1;repeat_checked.add('visual');t['repeat_exact']=True
        trace['optimized']['visual']=t
    base_seconds=times['native_prefix']+times['native_visual']+times['native_speech']
    new_seconds=sum(times[k] for k in ('native_prefix','native_speech','matching','layout','new_prefix','new_visual'))
    base=prediction(row,ctx,native['visual'],native['speech'],base_seconds,'m1_native')
    new=prediction(row,new_ctx,new_visual,native['speech'],new_seconds,'m1_ttf')
    checks=dict(GT_read=False,host=socket.gethostname(),actual_language_forwards=j.language_calls-before,
        actual_vision_forwards=j.vision_calls-before_vision,diagnostic_forwards=diagnostic_calls,times=times,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
    assert checks['actual_language_forwards']==base['calls']+3+len(wins)+diagnostic_calls
    assert checks['actual_vision_forwards']==1
    b=dict(base=base,optimized=new,checks=checks,plan=plan,layout=evidence,dense_check=dense_check,traces=traces,
        segments=[list(s) for s in segments],native_ctx={k:v for k,v in ctx.items() if k not in ('rope','positions','files')},
        new_ctx={k:v for k,v in new_ctx.items() if k!='rope'},new_rope=new_ctx['rope'].tolist(),source_metadata=metadata)
    del new_cache
    return b


def validate_bundle(row,b,segments,renderer,smoke):
    base,new=b['base'],b['optimized'];wins=fixed_windows(float(row['duration']),8)
    assert b['checks']['GT_read'] is False and b['segments']==[list(s) for s in segments]
    assert b['checks']['actual_vision_forwards']==1 and b['checks']['diagnostic_forwards']==(5 if smoke else 0)
    assert b['checks']['actual_language_forwards']==base['calls']+3+len(wins)+b['checks']['diagnostic_forwards']
    assert all(math.isfinite(v) and v>=0 for v in b['checks']['times'].values())
    frames=frame_paths(row['dataset'],row['video_id'],20);meta=b['source_metadata'];folder=CACHE/row['dataset']/row['video_id']
    assert meta==json.loads((folder/'metadata.json').read_text()) and meta['GT_read'] is False and meta['version']==SPEC['version']
    assert meta['frames']==[dict(time=t,path=str(p.relative_to(ROOT))) for t,p in frames] and meta['segments']==b['segments']
    features=torch.load(folder/'projector.pt',map_location='cpu',weights_only=True)
    plan=select(features,torch.tensor(meta['grid_thw']),meta['merge']);assert plan==b['plan']
    assert b['layout']['source_indices']==plan['retained'] and b['layout']['visual_source_positions_exact'] and b['layout']['deepstack_gather_exact'] and b['layout']['text_and_boundaries_preserved']
    assert base['extra']['z_video']==new['extra']['z_video'] and base['extra']['stance']==new['extra']['stance']
    t=b['checks']['times']
    assert abs(base['extra']['standalone_seconds']-sum(t[k] for k in ('native_prefix','native_visual','native_speech')))<1e-6
    assert abs(new['extra']['standalone_seconds']-sum(t[k] for k in ('native_prefix','native_speech','matching','layout','new_prefix','new_visual')))<1e-6
    assert base['calls']==3+len(wins)+sum(bool(window_text(segments,a,z).strip()) for a,z in wins) and new['calls']==base['calls']+3
    assert len(b['traces'])==len(base['extra']['windows'])==len(new['extra']['windows'])==len(wins)
    for name,r in [('base',base),('optimized',new)]:
        assert (r['dataset'],r['video_id'],r['duration'],r['native_rate'])==(row['dataset'],row['video_id'],float(row['duration']),4)
        assert r['error'] is None and np.isfinite(r['score_curve']).all() and r['extra']['standalone_seconds']>=0
        for i,(w,(a,z),trace) in enumerate(zip(r['extra']['windows'],wins,b['traces'])):
            assert (w['i'],w['start'],w['end'])==(i,a,z) and trace['body']==window_text(segments,a,z)
            v=trace['native' if name=='base' else 'optimized']['visual'];q=yesno_question(i,len(wins),a,z,trace['body'],'visual')
            assert v['question']==q and v['margin']==w['z_visual']
            if name=='optimized':
                ids,suffix=renderer.branch_ids(b['new_ctx']['msgs'],q,b['new_ctx']['history'],head_text=b['new_ctx']['head'])
                assert v['suffix_ids']==ids and v['suffix_text']==suffix and v['cache_restored'] and v['rope_restored']
            if trace['body'].strip():assert w['z_speech']==trace['native']['speech']['margin']
            else:assert 'z_speech' not in w
            assert w['z']==max(w['z_visual'],w['z_speech']) if 'z_speech' in w else w['z']==w['z_visual']
        idx=np.clip(((np.arange(len(r['score_curve']))+.5)/4//8).astype(int),0,len(wins)-1)
        assert np.array_equal(r['score_curve'],np.asarray([w['z'] for w in r['extra']['windows']])[idx])
    e=b['layout'];assert new['extra']['prefix_tokens']==e['compressed_prefix_tokens']
    assert b['new_rope']==[[max(max(axis) for axis in e['packed_positions'])+1-e['compressed_prefix_tokens']]]
    if smoke:assert b['dense_check']['native_order_exact'] and b['dense_check']['margin']==b['dense_check']['native_margin'] and sum(t['optimized']['visual'].get('repeat_exact',False) for t in b['traces'])==1


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261005_m1_ttf'/('r1_full_smoke' if a.smoke else 'r1_full_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,spec=SPEC,GT_read=False,smoke=a.smoke,torch=torch.__version__,transformers=transformers.__version__,code='experiments/20261005_m1_ttf/{ttf,measure}.py; sources2026-10-05',command='python -u '+' '.join(__import__('sys').argv))
    config=out/'config.json'
    if config.exists():assert {k:v for k,v in json.loads(config.read_text()).items() if k!='date'}=={k:v for k,v in cfg.items() if k!='date'}
    else:config.write_text(json.dumps(cfg,indent=2)+'\n')
    CACHE.mkdir(parents=True,exist_ok=True)
    (CACHE/'PROVENANCE.md').write_text('# TTF audit projector features\n\nGenerated2026-10-05 by `experiments/20261005_m1_ttf/measure.py`, sourceversion'+SPEC['version']+', Qwen/Qwen3-VL-8B-Instruct. Each metadata records actualhost, originalnativeJPEG paths/timestamps, literalASR, processorgrid anddtype/shape. Actualnativevisionforward is captured withoutmodification; measurementalways recomputes it, thiscacheonlysupportsCPUsource/selectionreplay. Newvideo visioncost chargedonce; noGT/no contentchecksums.\n')
    rows=selected_rows(a.smoke);asr={ds:load_asr(ds) for ds in DATASETS};torch.manual_seed(0);torch.set_num_threads(4)
    j=Judge(MODEL);j.language_calls=j.vision_calls=0
    hooks=[j.model.model.language_model.register_forward_pre_hook(lambda *_:setattr(j,'language_calls',j.language_calls+1)),j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    results={name:[] for name in ('base','optimized')}
    for number,row in enumerate(rows,1):
        p=out/'records'/row['dataset']/(row['video_id']+'.json');p.parent.mkdir(parents=True,exist_ok=True)
        segments=asr[row['dataset']].get(row['video_id'],[])
        if p.exists():b=json.loads(p.read_text())
        else:
            b=read_video(j,row,segments,a.smoke);validate_bundle(row,b,segments,j,a.smoke);p.write_text(json.dumps(b)+'\n')
        validate_bundle(row,b,segments,j,a.smoke)
        for name in results:results[name].append(b[name])
        logging.info('%d/%d %s/%s %.2fs keep=%d/%d',number,len(rows),row['dataset'],row['video_id'],b['optimized']['extra']['standalone_seconds'],b['plan']['retained_count'],b['plan']['original_count'])
    for name,rr in results.items():
        d=out/name;d.mkdir(exist_ok=True);(d/'config.json').write_text(json.dumps({**cfg,'measurement':name},indent=2)+'\n')
        (d/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
    for hook in hooks:hook.remove()
    logging.info('DONE coverage=%d',len(rows))


if __name__=='__main__':main()
