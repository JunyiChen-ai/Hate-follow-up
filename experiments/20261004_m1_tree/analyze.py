#!/usr/bin/env python3
"""Canonical evaluator orchestration; annotation access starts after scoring."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.eval.evaluate import within_video_macro
from src.video_inputs import load_asr
from extract import DATASETS,CACHE,selected_rows,validate_cached
from measure import existing,validate_records,validate_extension_inputs
from src.mllm_renderer import cpu_renderer
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')


def read(path):
    rows={}
    for line in path.open():
        r=json.loads(line);key=r['dataset'],r['video_id'];assert key not in rows;rows[key]=r
    return rows


def metrics(path):return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']}


def prepare(root,out,smoke,revision='r1'):
    renderer=cpu_renderer() if revision=='r3' else None
    rows=selected_rows(smoke);expected={(r['dataset'],r['video_id']):r for r in rows}
    cfg=json.loads((root/'config.json').read_text());assert cfg['GT_in_reader'] is False and cfg['smoke']==smoke
    assert cfg.get('reader_revision','r1')==revision
    records={name:existing(root/name/'predictions.jsonl',expected) for name in ('base','optimized')}
    checks=existing(root/'checks.jsonl',expected)
    assert all(r.keys()==expected.keys() for r in [*records.values(),checks])
    old=read(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');asr={ds:load_asr(ds) for ds in DATASETS}
    changed_global=changed_visual=changed_speech=verified=cloned=speech_cloned=extension_verified=0
    for key,row in expected.items():
        metadata,features=validate_cached(CACHE/key[0]/key[1],row)
        detail=json.loads((root/'details'/key[0]/(key[1]+'.json')).read_text())
        assert detail.get('reader_revision','r1')==revision
        if revision=='r3':extension_verified+=detail['extension']['fresh_render_verified']
        base,new=records['base'][key],records['optimized'][key]
        validate_records(row,base,new,checks[key],detail,metadata)
        if renderer is not None:validate_extension_inputs(renderer,row,asr[key[0]].get(key[1],[]),base,detail,metadata)
        assert detail['segments']==[list(s) for s in asr[key[0]].get(key[1],[])]
        assert base['extra']['z_video']==old[key]['extra']['z_video']
        assert base['extra']['windows']==old[key]['extra']['windows']
        assert np.array_equal(base['score_curve'],old[key]['score_curve'])
        changed_global+=base['extra']['z_video']!=new['extra']['z_video']
        for a,b,t in zip(base['extra']['windows'],new['extra']['windows'],detail['traces']):
            changed_visual+=a['z_visual']!=b['z_visual'];changed_speech+=a.get('z_speech')!=b.get('z_speech')
            verified+=t['fresh_render_verified']
            cloned+='cloned_margin' in t
            speech_cloned+='cloned_speech_margin' in t
    assert changed_visual>0
    if revision=='r2':assert changed_global==changed_speech==0
    elif revision=='r3':assert changed_global==0 and changed_speech>0
    else:assert changed_global>0
    if smoke:assert verified==len(rows)
    if revision=='r2':assert cloned==(len(rows) if smoke else 0)
    if revision=='r3':
        assert cloned==extension_verified==(len(rows) if smoke else 0)
        expected_speech=sum(any('z_speech' in w for w in r['extra']['windows']) for r in records['base'].values()) if smoke else 0
        assert speech_cloned==expected_speech
    result=dict(coverage=len(rows),native_exact=True,GT_read=False,changed_global=changed_global,
        changed_visual_windows=changed_visual,changed_speech_windows=changed_speech,
        fresh_render_checks=verified,cost={},mechanism_supported=False)
    if revision=='r2':result.update(reader_revision='r2',cloned_margin_checks=cloned)
    if revision=='r3':result.update(reader_revision='r3',cloned_margin_checks=cloned,
        cloned_speech_checks=speech_cloned,extension_fresh_render_checks=extension_verified)
    for ds in DATASETS:
        rr=[r for key,r in checks.items() if key[0]==ds]
        result['cost'][ds]=dict(standalone_seconds={a:sum(r['standalone_seconds'][a] for r in rr) for a in ('base','optimized')},
            peak_GiB=max(r['peak_GiB'] for r in rr),new_prefix_tokens_max=max(r['new_prefix_tokens'] for r in rr),
            **{k:sum(r[k] for r in rr) for k in ('read_seconds','preprocessing_seconds','actual_forwards','caption_count','caption_tokens','added_local_images','feature_images')},
            input_actual_forwards={k:sum(r['input_actual_forwards'][k] for r in rr) for k in ('vision','language')})
        if revision=='r2':result['cost'][ds].update({k:sum(r[k] for r in rr) for k in (
            'prefix_seconds','native_visual_seconds','shared_speech_seconds','new_visual_seconds','diagnostic_forwards','diagnostic_seconds')})
        if revision=='r3':result['cost'][ds].update({k:sum(r[k] for r in rr) for k in (
            'prefix_seconds','reference_branch_seconds','extension_seconds','new_visual_seconds','new_speech_seconds',
            'diagnostic_forwards','diagnostic_seconds')})
        if smoke:
            sample=[r for r in rr if r['video_id']!='hate_video_114']
            result['cost'][ds]['rough_full_seconds']=(215 if ds=='HateMM' else 118)*np.mean([r['standalone_seconds']['optimized'] for r in sample])
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def evaluate(root,decoded,name):
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(root/name/'predictions.jsonl'),
        '--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(root/name/'metrics.json')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(root/name),
        '--datasets',*DATASETS,'--noleak','--transform','nscore','--key','calib','--duration','bma',
        '--bma-prior','length','--min-windows','2','--bma-grid','6','--arm','m2',
        '--out-root',str(decoded),'--tag',name],cwd=ROOT,check=True)


def bootstrap(values):
    x=np.asarray(values);rng=np.random.default_rng(0)
    if not len(x):return dict(n=0,mean=None,CI95=None)
    means=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(n=len(x),mean=float(x.mean()),CI95=np.quantile(means,[.025,.975]).tolist())


def report(root,decoded,out):
    mm={a:metrics(decoded/a/'metrics.json') for a in ('base','optimized')}
    current=metrics(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json')
    raw={a:read(root/a/'predictions.jsonl') for a in mm};final={a:read(decoded/a/'predictions.jsonl') for a in mm}
    result=dict(scope='development-selected; unchanged r6 independently applied',
        metric_sources={a:str((decoded/a/'metrics.json').relative_to(ROOT)) for a in mm},datasets={},mechanism_supported=False)
    per_video=[]
    for ds in DATASETS:
        assert all(mm['base'][ds][m]==current[ds][m] for m in METRICS),'native six metrics mismatch'
        gt=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True)
        ys={str(v):np.asarray(gt['y4'][i]) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        paired={kind:[] for kind in ('final','raw_max','raw_visual','raw_speech_shared')}
        for key in [k for k in raw['base'] if k[0]==ds]:
            y=ys[key[1]];r=dict(dataset=ds,video_id=key[1]);ww=raw['base'][key]['extra']['windows']
            idx=np.clip(((np.arange(len(raw['base'][key]['score_curve']))+.5)/4//8).astype(int),0,len(ww)-1)
            shared=np.asarray(['z_speech' in w for w in ww])[idx]
            for a in mm:
                wins=raw[a][key]['extra']['windows']
                curves=dict(final=np.asarray(final[a][key]['score_curve']),raw_max=np.asarray(raw[a][key]['score_curve']),
                    raw_visual=np.asarray([w['z_visual'] for w in wins])[idx])
                r[a]={kind:within_video_macro({key[1]:y},{key[1]:curve})[METRICS[-1]] for kind,curve in curves.items()}
                speech=np.asarray([w.get('z_speech',0.) for w in wins])[idx];n=min(len(y),len(speech));use=shared[:n]
                r[a]['raw_speech_shared']=within_video_macro({key[1]:y[:n][use]},{key[1]:speech[:n][use]})[METRICS[-1]]
            if r['base']['final'] is not None:
                per_video.append(r)
                for kind in paired:
                    if r['base'][kind] is not None and r['optimized'][kind] is not None:paired[kind].append(r['optimized'][kind]-r['base'][kind])
        delta={m:mm['optimized'][ds][m]-current[ds][m] for m in METRICS}
        result['datasets'][ds]=dict(final={a:{m:mm[a][ds][m] for m in METRICS} for a in mm},delta=delta,
            n_eligible=len(paired['final']),within_paired={kind:bootstrap(v) for kind,v in paired.items()})
    common=[m for m in METRICS if all(result['datasets'][ds]['delta'][m]>=.01 for ds in DATASETS)]
    no_losses=all(result['datasets'][ds]['delta'][m]>=(-.01 if m==METRICS[-1] else -.005) for ds in DATASETS for m in METRICS)
    result['gates']=dict(common_gain_metrics=common,no_losses_beyond_noise=no_losses,performance_pass=bool(common and no_losses),
        any_qualifying_gain=any(result['datasets'][ds]['delta'][m]>=.01 for ds in DATASETS for m in METRICS))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(out/'per_video.json').write_text(json.dumps(per_video,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True);print('ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--name',choices=('base','optimized'))
    ap.add_argument('--revision',choices=('r1','r2','r3'),default='r1');a=ap.parse_args()
    stem=a.revision+'_full_'+('smoke' if a.smoke else 'main');root=ROOT/'runs/20261004_m1_tree'/stem
    out=root.parent/(stem+'_analysis');out.mkdir(parents=True,exist_ok=True);decoded=root.parent/(stem+'_decoded')
    if a.stage=='prepare':prepare(root,out,a.smoke,a.revision)
    elif a.stage=='evaluate':assert not a.smoke and a.name;evaluate(root,decoded,a.name)
    else:assert not a.smoke;report(root,decoded,out)


if __name__=='__main__':main()
