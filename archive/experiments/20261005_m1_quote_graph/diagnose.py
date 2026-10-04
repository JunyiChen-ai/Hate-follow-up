#!/usr/bin/env python3
"""Postscore GT error analysis only; never imported into extraction/reading."""
import json
import logging
import socket
import numpy as np
from extract import ROOT,CACHE,DATASETS,selected_rows
from analyze import read,METRICS
from src.eval.evaluate import within_video_macro


def within(vid,y,curve):return within_video_macro({vid:y},{vid:curve})[METRICS[-1]]


def main():
    out=ROOT/'runs/20261005_m1_quote_graph/r1_error_analysis';out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    rawroot=ROOT/'runs/20261005_m1_quote_graph/r1_full_main';raw={a:read(rawroot/a/'predictions.jsonl') for a in ('base','optimized')}
    final=json.loads((ROOT/'runs/20261005_m1_quote_graph/r1_full_main_analysis/per_video.json').read_text());bykey={(r['dataset'],r['video_id']):r for r in final}
    result=dict(GT_read=True,scope='Development-selected postscore diagnostic; no alternative method scores, fit or parameter selection',datasets={},per_video=[])
    for ds in DATASETS:
        gt=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True)
        ys={str(v):np.asarray(gt['y4'][i]) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        strata={kind:[] for kind in ('no_context_video','some_context_video','context_frames','no_context_frames')}
        shifts={name:[] for name in ('context_pos','context_neg','empty_pos','empty_neg')};counts={k:0 for k in ('windows','context_windows','speech_windows')}
        for row in [r for r in selected_rows(False) if r['dataset']==ds]:
            key=ds,row['video_id'];m=json.loads((CACHE/ds/(key[1]+'.json')).read_text());b,n=raw['base'][key],raw['optimized'][key]
            y=ys[key[1]];wins=b['extra']['windows'];idx=np.clip(((np.arange(len(b['score_curve']))+.5)/4//8).astype(int),0,len(wins)-1)
            context=np.asarray([bool(p['selected']) for p in m['packets']]);counts['windows']+=len(wins);counts['context_windows']+=context.sum()
            delta=np.asarray([v.get('z_speech',0.)-u.get('z_speech',0.) for u,v in zip(wins,n['extra']['windows'])])[idx]
            speech=np.asarray(['z_speech' in w for w in wins])[idx];counts['speech_windows']+=sum('z_speech' in w for w in wins)
            length=min(len(y),len(delta));y=y[:length];delta=delta[:length];has=context[idx][:length];speech=speech[:length]
            for prefix,mask in (('context',has),('empty',~has)):
                for label,name in ((1,'pos'),(0,'neg')):shifts[prefix+'_'+name].extend(delta[mask & speech & (y==label)].tolist())
            if key not in bykey:continue
            r=bykey[key];change=r['optimized']['final']-r['base']['final'];cohort='some_context_video' if context.any() else 'no_context_video'
            strata[cohort].append(change)
            detail=dict(dataset=ds,video_id=key[1],context_windows=int(context.sum()),windows=len(wins),final_delta=change,
                rawmax_delta=r['optimized']['raw_max']-r['base']['raw_max'],stance=b['extra']['stance'],subset_delta={})
            for name,mask in (('context_frames',has),('no_context_frames',~has)):
                if not mask.any():continue
                ub=within(key[1],y[mask],np.asarray(b['score_curve'])[:length][mask]);un=within(key[1],y[mask],np.asarray(n['score_curve'])[:length][mask])
                if ub is not None and un is not None:strata[name].append(un-ub);detail['subset_delta'][name]=un-ub
            result['per_video'].append(detail)
        def stats(values):
            x=np.asarray(values)
            return dict(n=len(x),mean=float(x.mean()) if len(x) else None,median=float(np.median(x)) if len(x) else None)
        result['datasets'][ds]=dict(counts={k:int(v) for k,v in counts.items()},final_and_raw_subset_changes={k:stats(v) for k,v in strata.items()},speech_margin_shifts={k:stats(v) for k,v in shifts.items()})
    result['read_files']=['runs/20261005_m1_quote_graph/r1_full_main/{base,optimized}/predictions.jsonl',
        'runs/20261005_m1_quote_graph/r1_full_main_analysis/{summary,per_video,alignment}.json',
        'data/temporal_quotation_graph/{HateMM,HateClipSeg}/*.json','data/gt_4fps/{HateMM,HateClipSeg}.npz']
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='per_video'},indent=2));logging.info('DIAGNOSIS_DONE')


if __name__=='__main__':main()
