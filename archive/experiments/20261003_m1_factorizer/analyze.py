#!/usr/bin/env python3
"""Canonical complete-pair evaluation; GT only enters evaluation/reporting."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.eval.evaluate import within_video_macro
DATASETS=('HateMM','HateClipSeg');ARMS=('base','causal','factor')
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')


def read(path):
    out={}
    for r in map(json.loads,path.open()):
        k=r['dataset'],r['video_id'];assert k not in out and not r.get('error'),k;out[k]=r
    return out


def boot(values):
    v=np.asarray(values,float)
    if not len(v):return {'n':0}
    rng=np.random.default_rng(0);means=v[rng.integers(len(v),size=(2000,len(v)))].mean(axis=1)
    return {'n':len(v),'mean':float(v.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),
        'improved':int((v>1e-10).sum()),'worse':int((v< -1e-10).sum())}


def validate(raw,pred):
    for a in ARMS:
        assert pred[a].keys()==raw['base'].keys()
        for k,b in raw['base'].items():
            r=pred[a][k];assert r['native_rate']==4 and np.isfinite(r['score_curve']).all()
            assert len(r['score_curve'])==len(b['score_curve'])
            assert r['extra']['z_video']==b['extra']['z_video']


def prepare(root,out):
    raw={a:read(root/a/'predictions.jsonl') for a in ARMS};checks=read(root/'checks.jsonl')
    expected={k for k in read(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl') if k[0] in DATASETS}
    old=read(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl')
    assert raw['base'].keys()==raw['causal'].keys()==raw['factor'].keys()==checks.keys()==expected
    validate(raw,raw);maximum=0.
    for k,b in raw['base'].items():
        f=raw['factor'][k];h=old[k];assert b['extra']['z_video']==h['extra']['z_video']
        assert len(b['score_curve'])==len(h['score_curve'])
        assert len(b['extra']['windows'])==len(f['extra']['windows'])==len(h['extra']['windows'])
        assert len(raw['causal'][k]['extra']['windows'])==len(b['extra']['windows'])
        for wb,wc in zip(b['extra']['windows'],raw['causal'][k]['extra']['windows']):
            assert (wb['start'],wb['end'])==(wc['start'],wc['end'])
            assert ('z_speech' in wb)==('z_speech' in wc)
        for wb,wf,wh in zip(b['extra']['windows'],f['extra']['windows'],h['extra']['windows']):
            assert (wb['start'],wb['end'])==(wf['start'],wf['end'])==(wh['start'],wh['end'])
            assert ('z_speech' in wb)==('z_speech' in wf)==('z_speech' in wh)
            for m in ('z_visual','z_speech'):
                if m in wb:maximum=max(maximum,abs(wb[m]-wh[m]))
        with np.load(root/'tokens'/k[0]/(k[1]+'.npz')) as t:
            media=t['visual']|t['speech']
            assert len(t['groups'])==len(t['input_ids'])==checks[k]['prefix_tokens']
            assert np.all(t['groups'][~media]==-1) and np.all(t['groups'][media]>=0)
            assert t['local'].shape==(len(b['extra']['windows']),len(media))
    assert maximum==0.,('baseline read drift',maximum)
    (out/'alignment.json').write_text(json.dumps({'videos':len(expected),'native_global_exact':True,'native_windows_exact':True},indent=2)+'\n')
    print('PREPARED',len(expected),flush=True)


def evaluate(root,decoded,arm):
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(root/arm/'predictions.jsonl'),
        '--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(root/arm/'metrics.json')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(root/arm),
        '--datasets',*DATASETS,'--noleak','--transform','nscore','--key','calib','--duration','bma',
        '--bma-prior','length','--min-windows','2','--bma-grid','6','--arm','m2','--out-root',str(decoded),'--tag',arm],cwd=ROOT,check=True)


def metrics(path):return {r['dataset']:r for r in json.load(path.open())['per_dataset']}


def report(root,decoded,out):
    raw={a:read(root/a/'predictions.jsonl') for a in ARMS};pred={a:read(decoded/a/'predictions.jsonl') for a in ARMS}
    validate(raw,pred);checks=read(root/'checks.jsonl');assert checks.keys()==raw['base'].keys()
    mm={a:metrics(decoded/a/'metrics.json') for a in ARMS};rm={a:metrics(root/a/'metrics.json') for a in ARMS}
    old=metrics(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json')
    result={'datasets':{},'scope':'development-selected; unchanged r6 algorithm independently refit per arm',
        'metric_sources':{a:str((decoded/a/'metrics.json').relative_to(ROOT)) for a in ARMS},'mechanism_supported':False}
    allrows=[];cost={};lines=['dataset\tarm\tROC\tPR\twithin\traw_within\n']
    for ds in DATASETS:
        g=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True)
        ys={str(v):np.asarray(g['y4'][i]) for i,v in enumerate(g['video_ids']) if str(g['split'][i])=='test'}
        rows=[]
        for k,b in raw['base'].items():
            if k[0]!=ds:continue
            y=ys[k[1]];r={'dataset':ds,'video_id':k[1],'positive_fraction':float(y.mean()),'global_positive':b['extra']['z_video']>0}
            for a in ARMS:
                r[a]=within_video_macro({k[1]:y},{k[1]:np.asarray(pred[a][k]['score_curve'])})[METRICS[-1]]
                r[a+'_raw']=within_video_macro({k[1]:y},{k[1]:np.asarray(raw[a][k]['score_curve'])})[METRICS[-1]]
            if r['base'] is not None:r['delta']=r['factor']-r['base'];r['raw_delta']=r['factor_raw']-r['base_raw'];rows.append(r)
        allrows.extend(rows)
        result['datasets'][ds]={'final':{a:{m:mm[a][ds][m] for m in METRICS} for a in ARMS},
            'raw':{a:{m:rm[a][ds][m] for m in METRICS} for a in ARMS},
            'delta':{m:mm['factor'][ds][m]-mm['base'][ds][m] for m in METRICS},
            'delta_vs_causal':{m:mm['factor'][ds][m]-mm['causal'][ds][m] for m in METRICS},
            'delta_vs_current':{m:mm['factor'][ds][m]-old[ds][m] for m in METRICS},
            'within_paired':boot([r['delta'] for r in rows]),'raw_within_paired':boot([r['raw_delta'] for r in rows]),
            'within_factor_minus_causal':boot([r['factor']-r['causal'] for r in rows]),
            'correct_yes':boot([r['delta'] for r in rows if r['global_positive']]),
            'wrong_no':boot([r['delta'] for r in rows if not r['global_positive']]),
            'largest_gains':sorted(rows,key=lambda r:r['delta'],reverse=True)[:5],
            'largest_losses':sorted(rows,key=lambda r:r['delta'])[:5]}
        cc=[r for k,r in checks.items() if k[0]==ds]
        result['datasets'][ds]['encoding']={'videos':len(cc),
            'shared_tokens':sum(r['mapping']['shared_tokens'] for r in cc),
            'unassigned_media':sum(r['mapping']['unassigned_media'] for r in cc),
            'causal_global_max_abs_drift':max(abs(r['causal_global_diagnostic_only']-r['native_global']) for r in cc),
            'causal_window_max_abs_drift':max(r['causal_window_max_abs_diff'] for r in cc),
            'global_diagnostic_mean_abs_change':float(np.mean([abs(r['factor_global_diagnostic_only']-r['native_global']) for r in cc]))}
        cost[ds]={a:{'standalone_seconds':sum(r['extra']['prefix_seconds']+r['extra']['branch_seconds'] for k,r in raw[a].items() if k[0]==ds),
            'mean_forwards':float(np.mean([r['calls'] for k,r in raw[a].items() if k[0]==ds]))} for a in ARMS}
        cost[ds]['peak_GiB']=max(r['peak_GiB'] for r in cc)
        for a in ARMS:lines.append(ds+'\t'+a+'\t'+'\t'.join(f'{mm[a][ds][m]:.6f}' for m in METRICS)+f'\t{rm[a][ds][METRICS[-1]]:.6f}\n')
    result['gates']={'any_qualifying_gain':any(r['delta'][m]>=.01 for r in result['datasets'].values() for m in METRICS),
        'performance_pass':all(all(r[d][METRICS[-1]]>=.01 for d in ('delta','delta_vs_current','delta_vs_causal')) and
            all(r[d][m]>=(-.01 if m==METRICS[-1] else -.005) for d in ('delta','delta_vs_current','delta_vs_causal') for m in METRICS) for r in result['datasets'].values())}
    for name,value in (('summary',result),('cost',cost),('per_video',allrows)):(out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    (out/'table.tsv').write_text(''.join(lines));print(''.join(lines));print(result['gates']);print('ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);ap.add_argument('--arm',choices=ARMS);a=ap.parse_args()
    if a.stage=='evaluate' and not a.arm:ap.error('evaluate requires arm')
    parent=ROOT/'runs/20261003_m1_factorizer';root=parent/'r1_main';decoded=parent/'r1_main_decoded';out=parent/'r1_main_analysis'
    out.mkdir(parents=True,exist_ok=True);print('host',socket.gethostname(),flush=True)
    if a.stage=='prepare':prepare(root,out)
    elif a.stage=='evaluate':evaluate(root,decoded,a.arm)
    else:report(root,decoded,out)

if __name__=='__main__':main()
