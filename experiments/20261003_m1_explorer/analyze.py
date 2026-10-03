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
from src.video_inputs import fixed_windows,frame_paths
DATASETS=('HateMM','HateClipSeg');ARMS=('base','explore')
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')


def read(path):
    out={}
    for r in map(json.loads,path.open()):
        k=r['dataset'],r['video_id'];assert k not in out and not r.get('error');out[k]=r
    return out


def boot(v):
    v=np.asarray(v,float);rng=np.random.default_rng(0)
    return {'n':len(v),'mean':float(v.mean()),'ci95':np.quantile(v[rng.integers(len(v),size=(2000,len(v)))].mean(1),[.025,.975]).tolist()}


def prepare(root,out,smoke=False,version='r1'):
    cfg=json.load((root/'config.json').open())
    assert cfg['smoke']==smoke and cfg['layers']==list(range(36))
    assert cfg.get('revision','r1')==version and cfg.get('support_override',True)==(version=='r1')
    assert cfg['entropy_thresholds']==[.1,.3] and cfg['frames_per_round']==2 and cfg['round_limit']==2
    assert cfg['attention_exponent']==.5 and cfg['GT_in_reader'] is False
    assert cfg['model']=='Qwen/Qwen3-VL-8B-Instruct' and cfg['frames']==20 and cfg['window_seconds']==8 and cfg['fps']==4
    assert cfg['global_and_answer']==cfg['speech']=='original native'
    for arm in ARMS:assert json.load((root/arm/'config.json').open())=={**cfg,'arm':arm}
    raw={arm:read(root/arm/'predictions.jsonl') for arm in ARMS};checks=read(root/'checks.jsonl')
    manifest=read(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl')
    expected={k for k in manifest if k[0] in DATASETS}
    if smoke:
        expected={k for ds in DATASETS for k in [k for k in manifest if k[0]==ds][:2]}|{('HateMM','hate_video_114')}
    assert len(expected)==(5 if smoke else 333)
    assert all(v.keys()==expected for v in [*raw.values(),checks])
    old=read(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');count=0;rows=[]
    for k,b in raw['base'].items():
        h=old[k];assert b['extra']['z_video']==h['extra']['z_video'] and np.array_equal(b['score_curve'],h['score_curve'])
        c=checks[k];dur=float(manifest[k]['duration']);wins=fixed_windows(dur,8);ws=b['extra']['windows'];V=len(wins)
        B=sum(1+('z_speech' in w) for w in ws)
        details=json.load((root/'details'/k[0]/(k[1]+'.json')).open());traces=details['traces']
        assert len(ws)==len(h['extra']['windows'])==V==len(traces)
        frames=frame_paths(*k,20,'k20')
        assert details['original_frame_times']==[f[0] for f in frames]
        assert len(details['image_counts'])==len(frames)==c['actual_original_frames']
        assert c['actual_forwards']==3+B+c['acquisition_reads']+(V if smoke else 0)
        assert c['acquisition_reads']==sum(len(t['rounds']) for t in traces)
        assert c['visual_queries']==V and c['native_branches']==B and c['baseline_instrumented'] is True
        if smoke:assert c['native_replay_exact'] is True
        for i,(w,hw,t,(start,end)) in enumerate(zip(ws,h['extra']['windows'],traces,wins)):
            for name in ('start','end','z_visual','z_speech'):
                assert w.get(name)==hw.get(name);count+=name.startswith('z_') and name in w
            assert t['window']==i and t['initial_z']==w['z_visual'] and len(t['rounds'])<=2
            support=sum(start<=ft<end for ft,_ in frames);assert t['nominal_support']==support
            prior=np.asarray(t['initial_prior']);assert len(prior)==len(frames) and np.isfinite(prior).all() and abs(prior.sum()-1)<1e-5
            last=w['z_visual'];used=set()
            for ri,step in enumerate(t['rounds']):
                assert 1<=len(step['added'])<=2
                x=abs(last);e=math.exp(-x);entropy=math.log1p(e)+x*e/(1+e)
                assert (ri==0 and support==0 and cfg.get('support_override',True)) or entropy>=(.1,.3)[ri]
                for f in step['added']:
                    assert start<=f['time']<end and f['index'] not in used
                    assert f['index'] not in details['source']['legacy_excluded_indices']
                    assert f==details['source']['frames'][f['index']]
                    used.add(f['index'])
                assert {f['index'] for f in step['acquired']}==used and len(used)<=4
                assert [f['time'] for f in step['acquired']]==sorted(f['time'] for f in step['acquired'])
                assert len(step['paths'])==len(step['image_counts'])==len(step['image_grid_thw'])==len(used)
                assert all((ROOT/p).is_file() for p in step['paths'])
                assert np.isfinite(step['z']) and np.isfinite(step['prior']).all()
                assert len(step['prior'])==len(frames)+len(used) and abs(sum(step['prior'])-1)<1e-5
                last=step['z']
            assert t['final_z']==last==raw['explore'][k]['extra']['windows'][i]['z_visual']
        for arm in ARMS:
            r=raw[arm][k];assert r['extra']['z_video']==b['extra']['z_video'] and r['extra']['stance']==b['extra']['stance']
            assert r['duration']==dur and r['native_rate']==4 and len(r['score_curve'])==math.ceil(4*dur) and np.isfinite(r['score_curve']).all()
            assert r['calls']==3+B+(c['acquisition_reads'] if arm=='explore' else 0)
            for i,(w,v,extent) in enumerate(zip(ws,r['extra']['windows'],wins)):
                assert (v['i'],v['start'],v['end'])==(i,*extent) and w.get('z_speech')==v.get('z_speech')
                assert v['z']==max(v['z_visual'],v.get('z_speech',-np.inf)) and np.isfinite(v['z_visual'])
        rows.append({'dataset':k[0],'video_id':k[1],'peak_GiB':c['peak_GiB'],
            'standalone_seconds':c['standalone_seconds'],'paired_seconds':c['paired_seconds'],
            'acquisition_reads':c['acquisition_reads']})
    result={'no_GT':True,'videos':len(expected),'native_branches_exact':count,'native_globals_exact':True,'rows':rows}
    if version=='r2':
        replayed=read(ROOT/'runs/20261003_m1_explorer/r2_cache/explore/predictions.jsonl')
        for k,r in raw['explore'].items():
            h=replayed[k]
            assert r['extra']['windows']==h['extra']['windows'] and r['calls']==h['calls']
            assert r['extra']['z_video']==h['extra']['z_video'] and np.array_equal(r['score_curve'],h['score_curve'])
        result['r2_cached_replay_exact']=True
    if smoke:
        result['estimated_seconds']={ds:{arm:n*np.mean([r['standalone_seconds'][arm] for r in rows if r['dataset']==ds and r['video_id']!='hate_video_114']) for arm in ARMS} for ds,n in [('HateMM',215),('HateClipSeg',118)]}
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
            'delta':{m:mm['explore'][ds][m]-mm['base'][ds][m] for m in METRICS},
            'delta_vs_current':{m:mm['explore'][ds][m]-old[ds][m] for m in METRICS},
            'within_paired':boot([r['explore']-r['base'] for r in rows]),
            'raw_within_paired':boot([r['explore_raw']-r['base_raw'] for r in rows])}
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
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);ap.add_argument('--arm',choices=ARMS);ap.add_argument('--smoke',action='store_true');ap.add_argument('--version',choices=('r1','r2'),default='r1');a=ap.parse_args()
    if a.stage=='evaluate' and not a.arm:ap.error('evaluate requires arm')
    if a.smoke and a.stage!='prepare':ap.error('smoke is plumbing only')
    parent=ROOT/'runs/20261003_m1_explorer';root=parent/(a.version+('_smoke' if a.smoke else '_main'));decoded=parent/(a.version+'_main_decoded');out=root if a.smoke else parent/(a.version+'_main_analysis')
    out.mkdir(parents=True,exist_ok=True);print('host',socket.gethostname(),flush=True)
    if a.stage=='prepare':prepare(root,out,a.smoke,a.version)
    elif a.stage=='evaluate':evaluate(root,decoded,a.arm)
    else:report(root,decoded,out)


if __name__=='__main__':main()
