"""Strict unlabelled multigrain input proof then canonical raw/fixed-r6 metrics."""
import argparse
import json
import subprocess
import sys
import time
import numpy as np
from inputs import ROOT,SPEC,DATASETS,selected_rows

METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')


def predictions(path):
    result={}
    for line in path.open():
        r=json.loads(line);key=r['dataset'],r['video_id'];assert key not in result;result[key]=r
    return result


def prepare(root,out,smoke):
    from validate import validate_bundle
    from measure import IMPLEMENTATION
    from src.mllm_renderer import cpu_position_renderer
    from src.video_inputs import load_asr
    cfg=json.loads((root/'config.json').read_text());assert cfg['GT_read'] is False and cfg['spec']==SPEC and cfg['smoke']==smoke and cfg['implementation']==IMPLEMENTATION
    expected=selected_rows(smoke);keys={(r['dataset'],r['video_id']) for r in expected};rr={k:predictions(root/k/'predictions.jsonl') for k in ('base','optimized')}
    assert all(set(rows)==keys for rows in rr.values());old=predictions(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl')
    j=cpu_position_renderer();asr={d:load_asr(d) for d in DATASETS};report={}
    for i,row in enumerate(expected,1):
        key=row['dataset'],row['video_id'];b=json.loads((root/'records'/key[0]/(key[1]+'.json')).read_text());validate_bundle(j,row,asr[key[0]].get(key[1],[]),b,smoke)
        assert all(b[k]==rr[k][key] for k in rr) and b['base']['extra']['windows']==old[key]['extra']['windows'] and b['base']['extra']['z_video']==old[key]['extra']['z_video']
        assert np.array_equal(b['base']['score_curve'],old[key]['score_curve'])
        c=b['checks'];r=report.setdefault(key[0],dict(videos=0,source_frames=0,source_LM=0,source_vision=0,grains={},changed_V=0,probe_calls=0,context_blocks=0,
            actual_LM=0,actual_vision=0,processing_seconds=0.,source_seconds=0.,standalone_seconds=0.,peak_GiB=0.,clones=0,missing_LOCAL=0))
        r['videos']+=1;r['source_frames']+=c['source_frames'];r['source_LM']+=c['source_LM'];r['source_vision']+=c['source_vision'];r['changed_V']+=sum(t['native_visual']!=t['new_visual'] for t in b['traces'])
        for block in b['source_blocks']:r['grains'][block['grain']]=r['grains'].get(block['grain'],0)+1
        r['probe_calls']+=c['probe_calls'];r['context_blocks']+=c['selected_context_blocks'];r['actual_LM']+=c['actual_forwards'];r['actual_vision']+=c['actual_vision']
        r['processing_seconds']+=sum(c['times'].values());r['source_seconds']+=c['source_seconds'];r['standalone_seconds']+=b['optimized']['extra']['standalone_seconds'];r['peak_GiB']=max(r['peak_GiB'],c['peak_GiB'])
        r['clones']+=c['diagnostic_forwards'];r['missing_LOCAL']+=c['missing_LOCAL']
        if i%10==0:print('BOUND',i,len(expected),flush=True)
    assert all(r['changed_V']>0 and r['context_blocks']>0 and r['probe_calls']>0 and all(r['grains'].get(g,0)>0 for g in SPEC['granularities']) for r in report.values()),'completegrain/probe/media inputs must actually enter; no efficacyclaim'
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(dict(PASS=True,GT_read=False,coverage=len(keys),datasets=report,native_allraw_exact=True,
        mechanism_supported=False,CPU_threads=SPEC['cpu_threads'],attempt_paths=[str(p.relative_to(ROOT)) for p in sorted(root.glob('pipeline_attempt_*.json'))],
        cost_scope='all sourcevision/history/fullprefills/paidattention/FFT/storage/probe/reader/decode counted, actualsetup/validation/proof/failedattempts separate; native20/fullASR acquisition extra'),indent=2)+'\n');print('PREPARE_PASS',flush=True)


def evaluate(root,decoded,name):
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(root/name/'predictions.jsonl'),'--gt-dir','data/gt_4fps',
        '--datasets',*DATASETS,'--out',str(root/name/'metrics.json')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(root/name),'--datasets',*DATASETS,
        '--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length','--min-windows','2','--bma-grid','6','--arm','m2',
        '--out-root',str(decoded),'--tag',name],cwd=ROOT,check=True)


def metrics(path):return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']}


def report(decoded,out):
    base=metrics(decoded/'base/metrics.json');new=metrics(decoded/'optimized/metrics.json');current=metrics(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json')
    result=dict(scope='development-selected; originalfixedr6/evaluator/GT/cohort unchanged; performance only',datasets={},mechanism_supported=False,
        metric_sources={k:str((decoded/k/'metrics.json').relative_to(ROOT)) for k in ('base','optimized')})
    for ds in DATASETS:
        assert all(base[ds][k]==current[ds][k] for k in METRICS) and base[ds]['n_videos_defined']==new[ds]['n_videos_defined']==(84 if ds=='HateMM' else 99)
        result['datasets'][ds]=dict(base={k:base[ds][k] for k in METRICS},optimized={k:new[ds][k] for k in METRICS},delta={k:new[ds][k]-current[ds][k] for k in METRICS})
    common=[k for k in METRICS if all(result['datasets'][ds]['delta'][k]>=.01 for ds in DATASETS)]
    losses=all(result['datasets'][ds]['delta'][k]>=(-.01 if k==METRICS[-1] else -.005) for ds in DATASETS for k in METRICS)
    result['gates']=dict(common_gain_metrics=common,no_losses_beyond_noise=losses,performance_pass=bool(common and losses),any_qualifying_gain=any(result['datasets'][ds]['delta'][k]>=.01 for ds in DATASETS for k in METRICS))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--name',choices=('base','optimized'));a=ap.parse_args()
    root=ROOT/'runs/20261007_m1_mukv'/('r1_full_smoke' if a.smoke else 'r1_full_main');out=root.parent/(root.name+'_analysis');out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();done=False
    try:
        if a.stage=='prepare':prepare(root,out,a.smoke)
        elif a.stage=='evaluate':
            assert not a.smoke and a.name and json.loads((out/'alignment.json').read_text())['PASS'] is True;evaluate(root,root.parent/(root.name+'_decoded'),a.name)
        else:assert not a.smoke;report(root.parent/(root.name+'_decoded'),out)
        done=True
    finally:
        n=1;tag=a.stage+('_'+a.name if a.name else '')
        while (out/f'{tag}_attempt_{n:04d}.json').exists():n+=1
        (out/f'{tag}_attempt_{n:04d}.json').write_text(json.dumps(dict(stage=a.stage,completed=done,elapsed_seconds=time.perf_counter()-start),indent=2)+'\n')


if __name__=='__main__':main()
