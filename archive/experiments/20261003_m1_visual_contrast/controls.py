#!/usr/bin/env python3
"""Predeclared cached controls; labels used only by canonical evaluation/report."""
import argparse
import copy
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import numpy as np
from analyze import ROOT, DATASETS, METRICS, read, metrics, boot
from src.eval.evaluate import within_video_macro

PARENT=ROOT/'runs/20261003_m1_visual_contrast'
ARMS=('scale','video_shift','shuffle','common_shift')
OUT=PARENT/'r1_controls_analysis'


def construct(base,check,arm):
    r=copy.deepcopy(base);ws=r['extra']['windows'];cw=check['windows'];n=len(ws)
    a=np.array([w['z_visual'] for w in ws]);b=np.array([w['corrupted_visual'] for w in cw])
    c=np.array([w.get('z_speech',-np.inf) for w in ws]);perm=np.random.default_rng(0).permutation(n)
    d=float(np.maximum(2*a-b,c).mean()-np.maximum(a,c).mean())
    visual={'scale':2*a,'video_shift':2*a-b.mean(),'shuffle':2*a-b[perm],
            'common_shift':a+d}[arm]
    for i,w in enumerate(ws):
        w['z_visual']=float(visual[i])
        if arm=='common_shift' and 'z_speech' in w:w['z_speech']+=d
        w['z']=max(w[k] for k in ('z_visual','z_speech') if k in w)
    L=len(base['score_curve']);idx=np.clip(((np.arange(L)+.5)/4//8).astype(int),0,n-1)
    r['score_curve']=np.array([w['z'] for w in ws])[idx].tolist()
    r['method']='m1_visual_contrast_control_'+arm
    r['code_path']=str(Path(__file__).relative_to(ROOT))
    r['calls']=None
    r['extra'].update(control=arm,source_read='r1_main',control_new_model_calls=0)
    for key in ('prefix_seconds','branch_seconds','branch_seconds_note'):r['extra'].pop(key,None)
    diagnostic={'dataset':r['dataset'],'video_id':r['video_id'],'windows':n,'common_shift':d,
                'shuffle_index_changes':int(np.count_nonzero(perm!=np.arange(n))),
                'shuffle_value_changes':int(np.count_nonzero(b[perm]!=b))}
    if arm=='common_shift':
        expected=np.asarray(base['score_curve'])+d
        assert np.allclose(r['score_curve'],expected,rtol=0,atol=1e-12)
        assert math.isclose(np.mean([w['z'] for w in ws]),np.maximum(2*a-b,c).mean(),abs_tol=1e-12)
    if arm=='scale':assert np.array_equal(visual,2*a)
    assert np.isfinite(r['score_curve']).all()
    return r,diagnostic


def prepare():
    base=read(PARENT/'r1_main/base/predictions.jsonl');check=read(PARENT/'r1_main/checks.jsonl')
    assert len(base)==333 and base.keys()==check.keys()
    diagnostics=[]
    for arm in ARMS:
        target=PARENT/'r1_controls'/arm;target.mkdir(parents=True,exist_ok=True)
        with (target/'predictions.jsonl').open('w') as f:
            for k,b in base.items():
                r,d=construct(b,check[k],arm);f.write(json.dumps(r)+'\n')
                if arm=='shuffle':diagnostics.append(d)
        config={'host':socket.gethostname(),'arm':arm,'source':'runs/20261003_m1_visual_contrast/r1_main',
                'code':'archive/experiments/20261003_m1_visual_contrast/controls.py,2026-10-03',
                'GT_in_construction':False,'new_model_calls':0,'seed':'default_rng0 reset per video',
                'definition':'README R1 cached mechanism controls'}
        (target/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (OUT/'construction.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print('CONTROLS_PREPARED',flush=True)


def evaluate(arm):
    from analyze import evaluate as run_evaluation
    run_evaluation(PARENT/'r1_controls',PARENT/'r1_controls_decoded',arm)


def report():
    paths={'base':(PARENT/'r1_main/base',PARENT/'r1_main_decoded/base'),
           'contrast':(PARENT/'r1_main/contrast',PARENT/'r1_main_decoded/contrast')}
    paths.update({a:(PARENT/'r1_controls'/a,PARENT/'r1_controls_decoded'/a) for a in ARMS})
    raw={a:read(p/'predictions.jsonl') for a,(p,_) in paths.items()}
    pred={a:read(p/'predictions.jsonl') for a,(_,p) in paths.items()}
    mm={a:metrics(p/'metrics.json') for a,(_,p) in paths.items()}
    rm={a:metrics(p/'metrics.json') for a,(p,_) in paths.items()}
    for a in paths:
        assert raw[a].keys()==pred[a].keys()==raw['base'].keys()
        for k,r in pred[a].items():
            assert r['native_rate']==4 and np.isfinite(r['score_curve']).all()
            assert len(r['score_curve'])==len(raw['base'][k]['score_curve'])
            assert r['extra']['z_video']==raw['base'][k]['extra']['z_video']
    results={'scope':'development-selected controls; no method promotion','datasets':{},
             'sources':{a:[str(p.relative_to(ROOT)/'metrics.json') for p in paths[a]] for a in paths}}
    lines=['dataset\tarm\tROC\tPR\twithin\traw_within\n'];pervideo=[]
    for ds in DATASETS:
        g=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True)
        ys={str(v):np.asarray(g['y4'][i]) for i,v in enumerate(g['video_ids']) if str(g['split'][i])=='test'}
        rows=[]
        for k,b in raw['base'].items():
            if k[0]!=ds:continue
            y=ys[k[1]];row={'dataset':ds,'video_id':k[1],'global_positive':b['extra']['z_video']>0}
            for a in paths:
                row[a]=within_video_macro({k[1]:y},{k[1]:np.asarray(pred[a][k]['score_curve'])})[METRICS[-1]]
                row[a+'_raw']=within_video_macro({k[1]:y},{k[1]:np.asarray(raw[a][k]['score_curve'])})[METRICS[-1]]
                ws=raw[a][k]['extra']['windows'];L=min(len(y),len(raw[a][k]['score_curve']))
                idx=np.clip(((np.arange(L)+.5)/4//8).astype(int),0,len(ws)-1)
                for branch in ('visual','speech'):
                    values=np.array([w.get('z_'+branch,np.nan) for w in ws])[idx]
                    valid=np.isfinite(values)
                    row[a+'_'+branch+'_raw']=within_video_macro(
                        {k[1]:y[:L][valid]},{k[1]:values[valid]})[METRICS[-1]] if valid.any() else None
            if row['base'] is not None:rows.append(row)
        result={}
        for a in paths:
            result[a]={'final':{m:mm[a][ds][m] for m in METRICS},'raw':{m:rm[a][ds][m] for m in METRICS},
                       'delta_final_vs_base':{m:mm[a][ds][m]-mm['base'][ds][m] for m in METRICS},
                       'within_vs_base':boot([r[a]-r['base'] for r in rows]),
                       'within_contrast_minus_arm':boot([r['contrast']-r[a] for r in rows]),
                       'raw_within_vs_base':boot([r[a+'_raw']-r['base_raw'] for r in rows]),
                       'correct_yes':boot([r[a]-r['base'] for r in rows if r['global_positive']]),
                       'wrong_no':boot([r[a]-r['base'] for r in rows if not r['global_positive']]),
                       'largest_gains':sorted(rows,key=lambda r:r[a]-r['base'],reverse=True)[:5],
                       'largest_losses':sorted(rows,key=lambda r:r[a]-r['base'])[:5]}
            for branch in ('visual','speech'):
                vv=[r[a+'_'+branch+'_raw'] for r in rows if r[a+'_'+branch+'_raw'] is not None]
                result[a][branch+'_raw']={'n':len(vv),'mean':float(np.mean(vv)) if vv else None,
                    'scope':'within-video AUC over available branch frames only; unavailable speech omitted'}
            lines.append(ds+'\t'+a+'\t'+'\t'.join(f'{mm[a][ds][m]:.6f}' for m in METRICS)+f'\t{rm[a][ds][METRICS[-1]]:.6f}\n')
        results['datasets'][ds]=result;pervideo+=rows
        # Strictly positive affine visual transform must leave normal-score temporal input unchanged.
        assert max(abs(r['scale']-r['base']) for r in rows)<1e-10,'scale r6 within invariance failed'
    (OUT/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    (OUT/'per_video.json').write_text(json.dumps(pervideo,indent=2)+'\n')
    (OUT/'table.tsv').write_text(''.join(lines));print(''.join(lines));print('CONTROLS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True)
    ap.add_argument('--arm',choices=ARMS);args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    print('host',socket.gethostname(),flush=True)
    if args.stage=='prepare':prepare()
    elif args.stage=='evaluate':
        if args.arm is None:ap.error('--arm required')
        evaluate(args.arm)
    else:report()


if __name__=='__main__':main()
