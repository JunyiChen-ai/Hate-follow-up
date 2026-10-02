#!/usr/bin/env python3
"""Canonical evaluation and diagnostics of the visual contrast."""
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
DATASETS=('HateMM','HateClipSeg')
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')
ARMS=('base','contrast')


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


def prepare(root,out):
    raw={a:read(root/a/'predictions.jsonl') for a in ARMS}
    expected={k for k in read(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl') if k[0] in DATASETS}
    old=read(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl')
    checks=read(root/'checks.jsonl')
    assert raw['base'].keys()==raw['contrast'].keys()==checks.keys()==expected
    maximum=0.
    for k,b in raw['base'].items():
        e=raw['contrast'][k];h=old[k]
        assert b['extra']['z_video']==e['extra']['z_video']==h['extra']['z_video'],k
        assert b['native_rate']==e['native_rate']==4
        assert len(b['score_curve'])==len(e['score_curve'])==len(h['score_curve'])
        assert np.isfinite(b['score_curve']).all() and np.isfinite(e['score_curve']).all()
        assert len(b['extra']['windows'])==len(e['extra']['windows'])==len(h['extra']['windows'])
        cc=checks[k]
        assert cc['visual_branches']==len(cc['windows'])==len(b['extra']['windows'])
        if cc.get('noise_scope','full')=='full':expected_forwards=6+cc['native_branches']+cc['visual_branches']
        else:
            nonempty=sum(w['n_frames']>0 for w in cc['windows'])
            expected_forwards=3+cc['native_branches']+4*nonempty
            assert len(cc['pixel_interventions'])==len(cc['corrupted_globals_diagnostic_only'])==nonempty
            assert sorted(v['window'] for v in cc['pixel_interventions'])==[i for i,w in enumerate(cc['windows']) if w['n_frames']]
            for w in cc['windows']:
                if not w['n_frames']:assert w['clean_visual']==w['corrupted_visual']==w['contrast_visual']
        assert cc['actual_forwards']==expected_forwards
        assert cc['original_inputs_unchanged'] and cc['noise']['same_temporal_noise']
        count=0
        for wb,we,wh,d in zip(b['extra']['windows'],e['extra']['windows'],h['extra']['windows'],cc['windows']):
            assert (wb['start'],wb['end'])==(we['start'],we['end'])==(wh['start'],wh['end'])==(d['start'],d['end'])
            assert ('z_speech' in wb)==('z_speech' in we)==('z_speech' in wh)
            for key in ('z_visual','z_speech'):
                if key in wb:maximum=max(maximum,abs(wb[key]-wh[key]));count+=1
            if 'z_speech' in wb:assert wb['z_speech']==we['z_speech']
            assert wb['z_visual']==d['clean_visual'] and we['z_visual']==d['contrast_visual']
            assert we['z_visual']==2*d['clean_visual']-d['corrupted_visual']
            assert wb['z']==max(wb[m] for m in ('z_visual','z_speech') if m in wb)
            assert we['z']==max(we[m] for m in ('z_visual','z_speech') if m in we)
        assert count==cc['native_branches']
    assert maximum==0.,('baseline read drift',maximum)
    (out/'alignment.json').write_text(json.dumps({'videos':len(expected),'global_exact':True,'window_exact':True},indent=2)+'\n')
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
    mm={a:metrics(decoded/a/'metrics.json') for a in ARMS};rm={a:metrics(root/a/'metrics.json') for a in ARMS}
    old=metrics(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json');checks=read(root/'checks.jsonl')
    assert all(r.keys()==raw['base'].keys() for r in [*pred.values(),checks])
    for arm in ARMS:
        for key,r in pred[arm].items():
            assert r['native_rate']==4 and np.isfinite(r['score_curve']).all(),(arm,key)
            assert len(r['score_curve'])==len(raw['base'][key]['score_curve'])
            assert r['extra']['z_video']==raw['base'][key]['extra']['z_video']
    result={'datasets':{},'scope':'development-selected; fixed r6 algorithm with independent unsupervised fits',
        'metric_sources':{a:str((decoded/a/'metrics.json').relative_to(ROOT)) for a in ARMS},'mechanism_supported':False}
    allrows=[];cost={};lines=['dataset\tarm\tROC\tPR\twithin\traw_within\n']
    for ds in DATASETS:
        g=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True)
        ys={str(v):np.asarray(g['y4'][i]) for i,v in enumerate(g['video_ids']) if str(g['split'][i])=='test'}
        rows=[]
        for k,b in raw['base'].items():
            if k[0]!=ds or k[1] not in ys:continue
            y=ys[k[1]];r={'dataset':ds,'video_id':k[1],'positive_fraction':float(y.mean()),'global_positive':b['extra']['z_video']>0}
            for a in ARMS:
                assert len(pred[a][k]['score_curve'])==len(b['score_curve']) and pred[a][k]['extra']['z_video']==b['extra']['z_video']
                r[a]=within_video_macro({k[1]:y},{k[1]:np.asarray(pred[a][k]['score_curve'])})[METRICS[-1]]
                r[a+'_raw']=within_video_macro({k[1]:y},{k[1]:np.asarray(raw[a][k]['score_curve'])})[METRICS[-1]]
            if r['base'] is not None:r['delta']=r['contrast']-r['base'];r['raw_delta']=r['contrast_raw']-r['base_raw'];rows.append(r)
        allrows.extend(rows)
        result['datasets'][ds]={'final':{a:{m:mm[a][ds][m] for m in METRICS} for a in ARMS},
            'raw':{a:{m:rm[a][ds][m] for m in METRICS} for a in ARMS},
            'delta':{m:mm['contrast'][ds][m]-mm['base'][ds][m] for m in METRICS},
            'delta_vs_current':{m:mm['contrast'][ds][m]-old[ds][m] for m in METRICS},
            'within_paired':boot([r['delta'] for r in rows]),'raw_within_paired':boot([r['raw_delta'] for r in rows]),
            'correct_yes':boot([r['delta'] for r in rows if r['global_positive']]),
            'wrong_no':boot([r['delta'] for r in rows if not r['global_positive']]),
            'largest_gains':sorted(rows,key=lambda r:r['delta'],reverse=True)[:5],'largest_losses':sorted(rows,key=lambda r:r['delta'])[:5]}
        cc=[r for k,r in checks.items() if k[0]==ds];ww=[w for r in cc for w in r['windows']]
        dominance={a:sum(w['z_visual']>=w.get('z_speech',-np.inf) for k,r in raw[a].items() if k[0]==ds for w in r['extra']['windows']) for a in ARMS}
        result['datasets'][ds]['visual_intervention']={'windows':len(ww),
            'no_frame_windows':sum(w['n_frames']==0 for w in ww),'visual_dominant_windows':dominance,
            'mean_abs_visual_correction':float(np.mean([abs(w['clean_visual']-w['corrupted_visual']) for w in ww])),
            'mean_within_video_std_corrupted_read':float(np.mean([np.std([w['corrupted_visual'] for w in r['windows']]) for r in cc])),
            'mean_abs_global_diagnostic_change':float(np.mean([abs(r['native_global']-z) for r in cc
                for z in r.get('corrupted_globals_diagnostic_only',[r.get('corrupted_global_diagnostic_only')])])),
            'global_diagnostic_scope':'mean across corrupted prefix evaluations, not a downstream key'}
        cost[ds]={a:{'standalone_seconds':sum(r['extra']['prefix_seconds']+r['extra']['branch_seconds'] for k,r in raw[a].items() if k[0]==ds),
            'mean_forwards':float(np.mean([r['calls'] for k,r in raw[a].items() if k[0]==ds]))} for a in ARMS}
        cost[ds]['peak_GiB']=max(r['peak_GiB'] for r in cc)
        cost[ds]['extra_prefix_and_visual_seconds']=sum(r['extra_seconds'] for r in cc)
        for a in ARMS:lines.append(ds+'\t'+a+'\t'+'\t'.join(f'{mm[a][ds][m]:.6f}' for m in METRICS)+f'\t{rm[a][ds][METRICS[-1]]:.6f}\n')
    result['gates']={'any_qualifying_gain':any(r['delta'][m]>=.01 for r in result['datasets'].values() for m in METRICS),
        'performance_pass':all(r['delta'][METRICS[-1]]>=.01 and r['delta_vs_current'][METRICS[-1]]>=.01 and
            all(r[d][m]>=(-.01 if m==METRICS[-1] else -.005) for d in ('delta','delta_vs_current') for m in METRICS) for r in result['datasets'].values())}
    for name,value in (('summary',result),('cost',cost),('per_video',allrows)):(out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    (out/'table.tsv').write_text(''.join(lines));print(''.join(lines));print(result['gates']);print('ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);ap.add_argument('--arm',choices=ARMS)
    ap.add_argument('--run-name',default='r1_main');a=ap.parse_args()
    if a.stage=='evaluate' and not a.arm:ap.error('evaluate requires arm')
    parent=ROOT/'runs/20261003_m1_visual_contrast';root=parent/a.run_name;decoded=parent/(a.run_name+'_decoded');out=parent/(a.run_name+'_analysis')
    out.mkdir(parents=True,exist_ok=True);print('host',socket.gethostname(),flush=True)
    if a.stage=='prepare':prepare(root,out)
    elif a.stage=='evaluate':evaluate(root,decoded,a.arm)
    else:report(root,decoded,out)

if __name__=='__main__':main()
