#!/usr/bin/env python3
"""Post-scoring M1 branch diagnostics, using the canonical within metric."""
import argparse
import json
from pathlib import Path
import socket
import sys
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.eval.evaluate import within_video_macro
KEY='within_video_macro_ROC_AUC'


def read(path):
    rows={}
    for r in map(json.loads,path.open()):
        key=r['dataset'],r['video_id'];assert key not in rows and not r.get('error')
        rows[key]=r
    return rows


def auc(y,s):
    return within_video_macro({'video':y},{'video':s})[KEY]


def boot(values):
    values=np.asarray(values,float)
    if not len(values):return {'n':0,'mean':None,'ci95':None}
    if len(values)==1:return {'n':1,'mean':float(values[0]),'ci95':None}
    rng=np.random.default_rng(0)
    means=values[rng.integers(len(values),size=(2000,len(values)))].mean(axis=1)
    return {'n':len(values),'mean':float(values.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()}


def run(source,arms,output):
    raw={a:read(source/a/'predictions.jsonl') for a in arms}
    decoded={a:read(source.parent/(source.name+'_decoded')/a/'predictions.jsonl') for a in arms}
    assert arms[0]=='base'
    expected={k for k in read(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl') if k[0] in ('HateMM','HateClipSeg')}
    assert all(r.keys()==expected for r in [*raw.values(),*decoded.values()])
    result={'host':socket.gethostname(),'scope':'post-scoring development-selected diagnostic; no fitting or primary-metric replacement',
        'input':str(source.relative_to(ROOT)),'speech_scope':'only frames in windows with an available speech read',
        'single_window_scope':'raw constant score cannot add within-window timing; final r6 may reorder temporal cells',
        'datasets':{},'per_video':[]}
    for ds in ('HateMM','HateClipSeg'):
        with np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True) as gt:
            truth={str(v):np.asarray(gt['y4'][i]) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        records=[]
        for key,base in raw['base'].items():
            if key[0]!=ds:continue
            y=truth[key[1]];windows=base['extra']['windows'];n=len(windows)
            gt_length=len(y);prediction_length=len(base['score_curve'])
            assert base['native_rate']==4
            # Canonical within_video_macro compares the shared available extent.
            y=y[:min(gt_length,prediction_length)]
            times=(np.arange(len(y))+.5)/4
            indices=np.clip(np.searchsorted([w['start'] for w in windows],times,side='right')-1,0,n-1)
            row={'dataset':ds,'video_id':key[1],'n_windows':n,'GT_length':gt_length,
                'prediction_length':prediction_length,'evaluated_length':len(y),'arms':{}}
            availability={}
            for arm in arms:
                r=raw[arm][key];d=decoded[arm][key];ws=r['extra']['windows']
                assert len(r['score_curve'])==len(d['score_curve'])==prediction_length and r['native_rate']==d['native_rate']==4
                assert [(w['start'],w['end']) for w in ws]==[(w['start'],w['end']) for w in windows]
                values={'raw_max':auc(y,np.asarray(r['score_curve'])),'final':auc(y,np.asarray(d['score_curve']))}
                for branch in ('visual','speech'):
                    curve=np.asarray([w.get('z_'+branch,np.nan) for w in ws])[indices]
                    valid=np.isfinite(curve)
                    if arm=='base':availability[branch]=valid
                    else:assert np.array_equal(valid,availability[branch])
                    values[branch]=auc(y[valid],curve[valid]) if valid.any() else None
                row['arms'][arm]=values
            if row['arms']['base']['final'] is not None:records.append(row)
        summary={'eligible_final_videos':len(records),'arms':{}}
        for arm in arms:
            stats={}
            for name in ('visual','speech','raw_max','final'):
                pairs=[(r['arms'][arm][name],r['arms']['base'][name]) for r in records
                    if r['arms'][arm][name] is not None and r['arms']['base'][name] is not None]
                stats[name]={'n':len(pairs),'mean':float(np.mean([p[0] for p in pairs])) if pairs else None,
                    'paired_delta':boot([a-b for a,b in pairs])}
            for group,predicate in (('single_window',lambda r:r['n_windows']==1),('multiple_windows',lambda r:r['n_windows']>1)):
                deltas=[r['arms'][arm]['final']-r['arms']['base']['final'] for r in records if predicate(r)]
                stats[group]={'paired_delta':boot(deltas),'contribution_to_full_mean_delta':float(sum(deltas)/len(records))}
            summary['arms'][arm]=stats
        result['datasets'][ds]=summary;result['per_video']+=records
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print('DIAGNOSTICS_DONE',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--arms',nargs='+',required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.run.resolve(),args.arms,args.out.resolve())
