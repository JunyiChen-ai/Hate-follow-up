"""Post-whole descriptive error analysis, using canonical metrics only.

Never imported by source construction, prediction, fitting, or thresholds.
"""
import json
import argparse
import math
import os
from pathlib import Path
import socket
import numpy as np
from retrieval import ROOT
from src.eval.evaluate import pooled,within_video_macro


def read(path):
    return {(r['dataset'],r['video_id']):r for line in path.open() if line.strip() for r in [json.loads(line)]}


def one(y,s):return within_video_macro({'video':y},{'video':s})['within_video_macro_ROC_AUC']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--revision',type=int,choices=(1,2),default=1);args=ap.parse_args()
    stem=f'r{args.revision}_full_main_B'
    out=ROOT/'runs/20261006_m1_ordered_slots'/(stem+'_error_analysis');out.mkdir(parents=True,exist_ok=True)
    print(socket.gethostname(),flush=True);(out/'run.pid').write_text(str(os.getpid()))
    run=ROOT/'runs/20261006_m1_ordered_slots'/stem;decoded=run.parent/(stem+'_decoded')
    raw={a:read(run/a/'predictions.jsonl') for a in ('base','optimized')};final={a:read(decoded/a/'predictions.jsonl') for a in raw}
    sources=[str((run/a/'predictions.jsonl').relative_to(ROOT)) for a in raw]+[str((decoded/a/'predictions.jsonl').relative_to(ROOT)) for a in final]
    result=dict(GT_read=True,scope='development-selected post-whole error analysis only; canonical metric calls, no scoring/fitting/threshold input',sources=sources,datasets={})
    rows=[];window_rows=[];random=np.random.default_rng(0)
    for ds in ('HateMM','HateClipSeg'):
        path=ROOT/'data/gt_4fps'/f'{ds}.npz';result['sources'].append(str(path.relative_to(ROOT)))
        with np.load(path,allow_pickle=True) as gt:y={str(v):np.asarray(gt['y4'][i],np.int8) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        arrays={a:{k:{} for k in ('V','S','max','final')} for a in raw};differences={k:[] for k in ('V','S','max','final')}
        frame_groups={}
        for key,b in raw['base'].items():
            if key[0]!=ds:continue
            n=min(len(y[key[1]]),len(b['score_curve']));yy=y[key[1]][:n];idx=np.clip(((np.arange(n)+.5)/4//8).astype(int),0,len(b['extra']['windows'])-1)
            bundle=json.loads((run/'records'/ds/(key[1]+'.json')).read_text())
            for arm in raw:
                w=raw[arm][key]['extra']['windows']
                for kind,field in [('V','z_visual'),('S','z_speech'),('max','z')]:arrays[arm][kind][key[1]]=np.array([x.get(field,np.nan) for x in w])[idx]
                arrays[arm]['final'][key[1]]=np.asarray(final[arm][key]['score_curve'])[:n]
            vd=dict(dataset=ds,video_id=key[1],has_positive=bool(yy.any()),stance=b['extra']['stance'],global_margin=b['extra']['z_video'],within={})
            for kind in arrays['base']:
                pair={arm:one(yy,arrays[arm][kind][key[1]]) for arm in raw};delta=pair['optimized']-pair['base'] if all(v is not None for v in pair.values()) else None
                vd['within'][kind]=dict(**pair,delta=delta)
                if delta is not None:differences[kind].append(delta)
            vd['final_mean_delta']=float((arrays['optimized']['final'][key[1]]-arrays['base']['final'][key[1]]).mean());rows.append(vd)
            for t in bundle['traces']:
                wi=t['i'];mask=idx==wi;labels=yy[mask]
                record=dict(dataset=ds,video_id=key[1],i=wi,bounds=[t['start'],t['end']],stance=vd['stance'],global_margin=vd['global_margin'],
                    GT_positive_fraction=float(labels.mean()) if len(labels) else None,frame_count=len(labels),
                    native_visual=t['native_visual'],new_visual=t['new_visual'],native_speech=t['native_speech'],new_speech=t['new_speech'],
                    final_mean_delta=float((arrays['optimized']['final'][key[1]][mask]-arrays['base']['final'][key[1]][mask]).mean()) if len(labels) else None)
                window_rows.append(record)
                for positive in (False,True):
                    count=int((labels==positive).sum())
                    group=frame_groups.setdefault(f"{vd['stance']}_{'positive' if positive else 'negative'}",dict(n=0,V_delta_sum=0.,S_delta_sum=0.,S_n=0,final_delta_sum=0.))
                    group['n']+=count;group['V_delta_sum']+=count*(t['new_visual']-t['native_visual'])
                    if t['native_speech'] is not None:group['S_n']+=count;group['S_delta_sum']+=count*(t['new_speech']-t['native_speech'])
                    group['final_delta_sum']+=float((arrays['optimized']['final'][key[1]][mask]-arrays['base']['final'][key[1]][mask])[labels==positive].sum())
        table={arm:{kind:{**pooled(y,scores),**within_video_macro(y,scores)} for kind,scores in arrays[arm].items()} for arm in arrays}
        uncertainty={}
        for kind,delta in differences.items():
            delta=np.asarray(delta);samples=delta[random.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
            uncertainty[kind]=dict(n=len(delta),mean=float(delta.mean()),paired_bootstrap95=np.quantile(samples,[.025,.975]).tolist(),seed=0,draws=10000)
        for group in frame_groups.values():
            group['V_mean_delta']=group['V_delta_sum']/group['n'] if group['n'] else None
            group['S_mean_delta']=group['S_delta_sum']/group['S_n'] if group['S_n'] else None
            group['final_mean_delta']=group['final_delta_sum']/group['n'] if group['n'] else None
        result['datasets'][ds]=dict(canonical_raw_table=table,paired_within=uncertainty,frame_groups=frame_groups)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(out/'per_video.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'windows.json').write_text(json.dumps(window_rows,indent=2)+'\n')
    print(json.dumps({ds:r['paired_within'] for ds,r in result['datasets'].items()},indent=2));print('ERROR_ANALYSIS_DONE',flush=True)


if __name__=='__main__':main()
