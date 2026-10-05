#!/usr/bin/env python3
"""Strict local source replay; canonical evaluator/fixed-r6 orchestration only."""
import argparse
import json
import subprocess
import sys
import numpy as np
from measure import ROOT,SPEC,DATASETS,selected_rows,validate_bundle
from src.mllm_renderer import cpu_renderer
from src.video_inputs import load_asr
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')


def read(path):
    result={}
    for line in path.open():
        r=json.loads(line);key=r['dataset'],r['video_id'];assert key not in result;result[key]=r
    return result


def metrics(path):return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']}


def prepare(root,out,smoke):
    cfg=json.loads((root/'config.json').read_text());assert cfg['GT_read'] is False and cfg['smoke']==smoke and cfg['spec']==SPEC
    rows=selected_rows(smoke);expected={(r['dataset'],r['video_id']):r for r in rows}
    rr={name:read(root/name/'predictions.jsonl') for name in ('base','optimized')}
    assert all(r.keys()==expected.keys() for r in rr.values())
    old=read(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');j=cpu_renderer();asr={ds:load_asr(ds) for ds in DATASETS}
    bundles={};changed=0;repeats=0
    for key,row in expected.items():
        b=json.loads((root/'records'/key[0]/(key[1]+'.json')).read_text())
        validate_bundle(row,b,asr[key[0]].get(key[1],[]),j,smoke)
        assert all(b[name]==rr[name][key] for name in rr)
        base,new=b['base'],b['optimized']
        assert base['extra']['z_video']==old[key]['extra']['z_video'] and base['extra']['windows']==old[key]['extra']['windows']
        assert np.array_equal(base['score_curve'],old[key]['score_curve'])
        changed+=sum(a.get('z_speech')!=z.get('z_speech') for a,z in zip(base['extra']['windows'],new['extra']['windows']))
        repeats+=sum(t.get('speech',{}).get('repeat_new_exact',False) for t in b['traces'])
        bundles[key]=b
    result=dict(coverage=len(rows),GT_read=False,native_allraw_exact=True,global_visual_exact=True,source_mapping_replayed=True,
        changed_speech_windows=changed,cloned_cache_checks=repeats,cost={},mechanism_supported=False)
    for ds in DATASETS:
        bb=[b for key,b in bundles.items() if key[0]==ds];checks=[b['checks'] for b in bb]
        result['cost'][ds]=dict(standalone_seconds={name:sum(b[name]['extra']['standalone_seconds'] for b in bb) for name in rr},
            peak_GiB=max(c['peak_GiB'] for c in checks),actual_language_forwards=sum(c['actual_language_forwards'] for c in checks),
            actual_vision_forwards=sum(c['actual_vision_forwards'] for c in checks),diagnostic_forwards=sum(c['diagnostic_forwards'] for c in checks),
            max_source_key_buffer_bytes=max(c['source_key_buffer_bytes'] for c in checks),
            times={k:sum(c['times'][k] for c in checks) for k in checks[0]['times']})
    assert changed>0
    if smoke:assert repeats==5
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('PREPARE_PASS',flush=True)


def evaluate(root,decoded,name):
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(root/name/'predictions.jsonl'),
        '--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(root/name/'metrics.json')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(root/name),'--datasets',*DATASETS,
        '--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length','--min-windows','2',
        '--bma-grid','6','--arm','m2','--out-root',str(decoded),'--tag',name],cwd=ROOT,check=True)


def report(root,decoded,out):
    mm={name:metrics(decoded/name/'metrics.json') for name in ('base','optimized')}
    current=metrics(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json')
    result=dict(scope='development-selected; fixed r6; not mechanism evidence',metric_sources={name:str((decoded/name/'metrics.json').relative_to(ROOT)) for name in mm},datasets={},mechanism_supported=False)
    for ds in DATASETS:
        assert all(mm['base'][ds][m]==current[ds][m] for m in METRICS),'native six metrics mismatch'
        count=mm['optimized'][ds]['n_videos_defined']
        assert count==(84 if ds=='HateMM' else 99)
        result['datasets'][ds]=dict(final={name:{m:mm[name][ds][m] for m in METRICS} for name in mm},n_eligible=count,
            delta={m:mm['optimized'][ds][m]-current[ds][m] for m in METRICS})
    common=[m for m in METRICS if all(result['datasets'][ds]['delta'][m]>=.01 for ds in DATASETS)]
    losses=all(result['datasets'][ds]['delta'][m]>=(-.01 if m==METRICS[-1] else -.005) for ds in DATASETS for m in METRICS)
    result['gates']=dict(common_gain_metrics=common,no_losses_beyond_noise=losses,performance_pass=bool(common and losses),
        any_qualifying_gain=any(result['datasets'][ds]['delta'][m]>=.01 for ds in DATASETS for m in METRICS))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--name',choices=('base','optimized'));a=ap.parse_args()
    stem='r1_full_'+('smoke' if a.smoke else 'main');root=ROOT/'runs/20261005_m1_rote'/stem
    out=root.parent/(stem+'_analysis');out.mkdir(parents=True,exist_ok=True);decoded=root.parent/(stem+'_decoded')
    if a.stage=='prepare':prepare(root,out,a.smoke)
    elif a.stage=='evaluate':assert not a.smoke and a.name;evaluate(root,decoded,a.name)
    else:assert not a.smoke;report(root,decoded,out)


if __name__=='__main__':main()
