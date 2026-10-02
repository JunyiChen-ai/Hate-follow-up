#!/usr/bin/env python3
"""Canonical paired evaluation; annotations only enter post-scoring reporting."""
import argparse
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import numpy as np
from scipy.special import logsumexp
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.eval.evaluate import within_video_macro
from src.video_inputs import fixed_windows
DATASETS=('HateMM','HateClipSeg');ARMS=('base','full')
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')


def read(path):
    out={}
    for r in map(json.loads,path.open()):
        k=r['dataset'],r['video_id'];assert k not in out and not r.get('error');out[k]=r
    return out


def boot(v):
    v=np.asarray(v,float);rng=np.random.default_rng(0)
    return {'n':len(v),'mean':float(v.mean()),'ci95':np.quantile(v[rng.integers(len(v),size=(2000,len(v)))].mean(1),[.025,.975]).tolist()}


def prepare(root,out,smoke=False):
    cfg=json.load((root/'config.json').open())
    assert cfg['smoke']==smoke and cfg['lambda']==.17 and cfg['gamma']==.3
    assert cfg['layers']==list(range(36)) and cfg['sla_indices']==list(range(30,35))
    assert cfg['model']=='Qwen/Qwen3-VL-8B-Instruct' and cfg['frames']==20 and cfg['window_seconds']==8 and cfg['fps']==4
    assert cfg['GT_in_reader'] is False and cfg['global_and_answer']==cfg['speech']=='original native'
    for a in ARMS:assert json.load((root/a/'config.json').open())=={**cfg,'arm':a}
    raw={a:read(root/a/'predictions.jsonl') for a in ARMS};checks=read(root/'checks.jsonl')
    manifest=read(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl')
    if smoke:
        expected={k for ds in DATASETS for k in [k for k in manifest if k[0]==ds][:2]}|{('HateMM','hate_video_114')}
        assert len(expected)==5
    else:expected={k for k in manifest if k[0] in DATASETS};assert len(expected)==333
    assert all(v.keys()==expected for v in [*raw.values(),checks])
    old=read(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');count=0;rows=[]
    for k,b in raw['base'].items():
        h=old[k];assert b['extra']['z_video']==h['extra']['z_video'] and np.array_equal(b['score_curve'],h['score_curve'])
        ws=b['extra']['windows'];c=checks[k];dur=float(manifest[k]['duration']);wins=fixed_windows(dur,8);V=len(wins)
        B=sum(1+('z_speech' in w) for w in ws)
        assert len(ws)==len(h['extra']['windows'])==V
        assert c['actual_forwards']==6+B+2*V+(3+B if smoke else 0)
        assert c['visual_queries']==V and c['native_branches']==B
        assert c['query_tokens_match'] and c['text_prefix_has_no_images'] and c['native_mRoPE_restored']
        assert c['layer_indices']==list(range(36)) and c['sla_indices']==list(range(30,35))
        norms=np.asarray(c['direction_norms']);assert norms.shape==(V,36) and np.isfinite(norms).all() and (norms>=0).all()
        if smoke:assert c['verify']['native_windows_exact'] and c['verify']['native_global_exact']
        for w,hw in zip(ws,h['extra']['windows']):
            for name in ('start','end','z_visual','z_speech'):
                assert w.get(name)==hw.get(name);count+=name.startswith('z_') and name in w
        with np.load(root/'tokens'/k[0]/(k[1]+'.npz')) as t:
            ny=len(t['yes_ids']);N=ny+len(t['no_ids'])
            for name in ('native_final','native_previous','steered_final','steered_previous'):
                assert t[name].shape==(V,N) and np.isfinite(t[name]).all()
            assert len(t['visual'])==len(t['input_ids'])==c['prefix_tokens'] and t['visual'].sum()==c['visual_tokens']
            logits={'base':t['native_final'],'full':.7*t['steered_final']+.3*t['steered_previous']}
            for a in ARMS:
                r=raw[a][k];assert r['extra']['z_video']==b['extra']['z_video'] and r['extra']['stance']==b['extra']['stance']
                assert r['duration']==dur and r['native_rate']==4 and len(r['score_curve'])==math.ceil(4*dur) and np.isfinite(r['score_curve']).all()
                assert len(r['extra']['windows'])==V and r['calls']==(3+B if a=='base' else 6+B+2*V)
                for i,(w,v,extent) in enumerate(zip(ws,r['extra']['windows'],wins)):
                    assert (v['i'],v['start'],v['end'])==(i,*extent) and w.get('z_speech')==v.get('z_speech')
                    assert abs(v['z_visual']-(logsumexp(logits[a][i,:ny])-logsumexp(logits[a][i,ny:])))<2e-5
                    assert v['z']==max(v['z_visual'],v.get('z_speech',-np.inf))
        rows.append({'dataset':k[0],'video_id':k[1],'peak_GiB':c['peak_GiB'],'prefix_tokens':c['prefix_tokens'],
            'max_visual_delta':max(abs(w['z_visual']-v['z_visual']) for w,v in zip(ws,raw['full'][k]['extra']['windows'])),
            'mean_direction_norm':float(norms.mean()),
            'standalone_seconds':{a:raw[a][k]['extra']['standalone_seconds'] for a in ARMS}})
    result={'no_GT':True,'videos':len(expected),'native_branches_exact':count,'native_globals_exact':True,'rows':rows}
    if smoke:
        result['estimated_seconds']={ds:{a:n*np.mean([r['standalone_seconds'][a] for r in rows if r['dataset']==ds and r['video_id']!='hate_video_114']) for a in ARMS} for ds,n in [('HateMM',215),('HateClipSeg',118)]}
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(result,indent=2)+'\n')
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
    result={'scope':'development-selected, fixed r6 independently refit per arm','datasets':{},'mechanism_supported':False,
        'metric_sources':{a:str((decoded/a/'metrics.json').relative_to(ROOT)) for a in ARMS}}
    costs={};allrows=[];lines=['dataset\tarm\tROC\tPR\twithin\traw_within\n']
    assert all(v.keys()==raw['base'].keys() for v in [*raw.values(),*pred.values(),checks])
    for a in ARMS:
        for k,b in raw['base'].items():
            r=pred[a][k]
            assert np.isfinite(r['score_curve']).all()
            assert r['extra']['z_video']==b['extra']['z_video']
            if 'stance' in r['extra']:assert r['extra']['stance']==b['extra']['stance']
    for ds in DATASETS:
        gt=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True)
        ys={str(v):np.asarray(gt['y4'][i]) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        rows=[]
        for k,b in raw['base'].items():
            if k[0]!=ds:continue
            row={'dataset':ds,'video_id':k[1],'global_positive':b['extra']['z_video']>0}
            for a in ARMS:
                assert len(pred[a][k]['score_curve'])==len(b['score_curve']) and pred[a][k]['native_rate']==4
                for suffix,source in (('',pred),('_raw',raw)):
                    row[a+suffix]=within_video_macro({k[1]:ys[k[1]]},{k[1]:np.asarray(source[a][k]['score_curve'])})[METRICS[-1]]
            if row['base'] is not None:rows.append(row)
        allrows+=rows
        result['datasets'][ds]={'final':{a:{m:mm[a][ds][m] for m in METRICS} for a in ARMS},
            'raw':{a:{m:rm[a][ds][m] for m in METRICS} for a in ARMS},
            'delta':{m:mm['full'][ds][m]-mm['base'][ds][m] for m in METRICS},
            'delta_vs_current':{m:mm['full'][ds][m]-old[ds][m] for m in METRICS},
            'within_paired':boot([r['full']-r['base'] for r in rows]),
            'raw_within_paired':boot([r['full_raw']-r['base_raw'] for r in rows])}
        costs[ds]={a:{'standalone_seconds':sum(r['extra']['standalone_seconds'] for k,r in raw[a].items() if k[0]==ds),
            'mean_forwards':float(np.mean([r['calls'] for k,r in raw[a].items() if k[0]==ds]))} for a in ARMS}
        costs[ds]['peak_GiB']=max(c['peak_GiB'] for k,c in checks.items() if k[0]==ds)
        for a in ARMS:lines.append(ds+'\t'+a+'\t'+'\t'.join(f'{mm[a][ds][m]:.6f}' for m in METRICS)+f'\t{rm[a][ds][METRICS[-1]]:.6f}\n')
    result['gates']={'any_qualifying_gain':any(r['delta'][m]>=.01 for r in result['datasets'].values() for m in METRICS),
        'performance_pass':all(all(r[d][METRICS[-1]]>=.01 for d in ('delta','delta_vs_current')) and
            all(r[d][m]>=(-.01 if m==METRICS[-1] else -.005) for d in ('delta','delta_vs_current') for m in METRICS) for r in result['datasets'].values())}
    for name,value in (('summary',result),('cost',costs),('per_video',allrows)):(out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    (out/'table.tsv').write_text(''.join(lines));print(''.join(lines));print(result['gates']);print('ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);ap.add_argument('--arm',choices=ARMS);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.stage=='evaluate' and not a.arm:ap.error('evaluate requires arm')
    if a.smoke and a.stage!='prepare':ap.error('smoke is plumbing only')
    parent=ROOT/'runs/20261003_m1_reinforcer';root=parent/('r1_smoke' if a.smoke else 'r1_main');decoded=parent/'r1_main_decoded';out=root if a.smoke else parent/'r1_main_analysis'
    out.mkdir(parents=True,exist_ok=True);print('host',socket.gethostname(),flush=True)
    if a.stage=='prepare':prepare(root,out,a.smoke)
    elif a.stage=='evaluate':evaluate(root,decoded,a.arm)
    else:report(root,decoded,out)


if __name__=='__main__':main()
