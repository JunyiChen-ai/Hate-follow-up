#!/usr/bin/env python3
"""Canonical paired evaluation; annotations only enter post-scoring reporting."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import numpy as np
from scipy.special import logsumexp
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.eval.evaluate import within_video_macro
DATASETS=('HateMM','HateClipSeg');ARMS=('base','eager','pai')
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')


def read(path):
    out={}
    for r in map(json.loads,path.open()):
        k=r['dataset'],r['video_id'];assert k not in out and not r.get('error');out[k]=r
    return out


def boot(v):
    v=np.asarray(v,float);rng=np.random.default_rng(0)
    return {'n':len(v),'mean':float(v.mean()),'ci95':np.quantile(v[rng.integers(len(v),size=(2000,len(v)))].mean(1),[.025,.975]).tolist()}


def prepare(root,out):
    raw={a:read(root/a/'predictions.jsonl') for a in ARMS};checks=read(root/'checks.jsonl')
    expected={k for k in read(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl') if k[0] in DATASETS}
    assert all(v.keys()==expected for v in [*raw.values(),checks])
    old=read(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');count=0
    for k,b in raw['base'].items():
        h=old[k];assert b['extra']['z_video']==h['extra']['z_video']
        assert np.array_equal(b['score_curve'],h['score_curve'])
        ws=b['extra']['windows'];c=checks[k];V=len(ws);B=sum(1+('z_speech' in w) for w in ws)
        assert len(h['extra']['windows'])==V and len(c['attention'])==V
        assert c['actual_forwards']==6+B+3*V and c['visual_queries']==V and c['native_branches']==B
        assert c['query_tokens_match'] and c['text_prefix_has_no_images']
        for w,hw in zip(ws,h['extra']['windows']):
            for name in ('start','end','z_visual','z_speech'):
                assert w.get(name)==hw.get(name);count+=name.startswith('z_') and name in w
        with np.load(root/'tokens'/k[0]/(k[1]+'.npz')) as t:
            ny=len(t['yes_ids']);assert t['eager'].shape==t['amplified'].shape==t['reference'].shape==(V,ny+len(t['no_ids']))
            assert len(t['visual'])==len(t['input_ids'])==c['prefix_tokens'] and t['visual'].sum()==c['visual_tokens']
            logits={'eager':t['eager'],'pai':1.1*t['amplified']-.1*t['reference']}
            for a in ARMS:
                r=raw[a][k];assert r['extra']['z_video']==b['extra']['z_video'] and r['extra']['stance']==b['extra']['stance']
                assert r['native_rate']==4 and len(r['score_curve'])==len(b['score_curve']) and np.isfinite(r['score_curve']).all()
                assert len(r['extra']['windows'])==V
                assert r['calls']==(3+B if a!='pai' else 6+B+V)
                for i,(w,v) in enumerate(zip(ws,r['extra']['windows'])):
                    assert (w['start'],w['end'],w.get('z_speech'))==(v['start'],v['end'],v.get('z_speech'))
                    if a!='base':assert abs(v['z_visual']-(logsumexp(logits[a][i,:ny])-logsumexp(logits[a][i,ny:])))<2e-5
        for d in c['attention']:
            for a in ('eager','boost'):assert d[a]['layer_indices']==list(range(2,36))
            assert d['eager']['image_attention_mass_before_after'][0]==d['eager']['image_attention_mass_before_after'][1]
            assert d['boost']['image_attention_mass_before_after'][1]>=d['boost']['image_attention_mass_before_after'][0]-1e-6
    (out/'alignment.json').write_text(json.dumps({'videos':len(expected),'native_branches_exact':count,'native_globals_exact':True,'speech_preserved':True},indent=2)+'\n')
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
            'delta':{m:mm['pai'][ds][m]-mm['base'][ds][m] for m in METRICS},
            'delta_vs_eager':{m:mm['pai'][ds][m]-mm['eager'][ds][m] for m in METRICS},
            'delta_vs_current':{m:mm['pai'][ds][m]-old[ds][m] for m in METRICS},
            'within_paired':boot([r['pai']-r['base'] for r in rows]),
            'within_pai_minus_eager':boot([r['pai']-r['eager'] for r in rows]),
            'raw_within_paired':boot([r['pai_raw']-r['base_raw'] for r in rows])}
        costs[ds]={a:{'standalone_seconds':sum(r['extra']['standalone_seconds'] for k,r in raw[a].items() if k[0]==ds),
            'mean_forwards':float(np.mean([r['calls'] for k,r in raw[a].items() if k[0]==ds]))} for a in ARMS}
        costs[ds]['peak_GiB']=max(c['peak_GiB'] for k,c in checks.items() if k[0]==ds)
        for a in ARMS:lines.append(ds+'\t'+a+'\t'+'\t'.join(f'{mm[a][ds][m]:.6f}' for m in METRICS)+f'\t{rm[a][ds][METRICS[-1]]:.6f}\n')
    result['gates']={'any_qualifying_gain':any(r['delta'][m]>=.01 for r in result['datasets'].values() for m in METRICS),
        'performance_pass':all(all(r[d][METRICS[-1]]>=.01 for d in ('delta','delta_vs_eager','delta_vs_current')) and
            all(r[d][m]>=(-.01 if m==METRICS[-1] else -.005) for d in ('delta','delta_vs_eager','delta_vs_current') for m in METRICS) for r in result['datasets'].values())}
    for name,value in (('summary',result),('cost',costs),('per_video',allrows)):(out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    (out/'table.tsv').write_text(''.join(lines));print(''.join(lines));print(result['gates']);print('ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);ap.add_argument('--arm',choices=ARMS);a=ap.parse_args()
    if a.stage=='evaluate' and not a.arm:ap.error('evaluate requires arm')
    parent=ROOT/'runs/20261003_m1_amplifier';root=parent/'r1_main';decoded=parent/'r1_main_decoded';out=parent/'r1_main_analysis'
    out.mkdir(parents=True,exist_ok=True);print('host',socket.gethostname(),flush=True)
    if a.stage=='prepare':prepare(root,out)
    elif a.stage=='evaluate':evaluate(root,decoded,a.arm)
    else:report(root,decoded,out)


if __name__=='__main__':main()
