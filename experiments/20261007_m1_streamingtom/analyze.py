"""Strict current-input binding then sole canonical evaluator/fixed-r6 CLI."""
import argparse
import json
import subprocess
import sys
import time
import numpy as np
from inputs import ROOT,SPEC,DATASETS,selected_rows

METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')


def records(path):
    rows={}
    for line in path.open():
        r=json.loads(line);key=r['dataset'],r['video_id'];assert key not in rows;rows[key]=r
    return rows


def prepare(root,out,smoke):
    from validate import validate_bundle
    from src.mllm_renderer import cpu_position_renderer
    from src.video_inputs import load_asr
    from measure import IMPLEMENTATION
    cfg=json.loads((root/'config.json').read_text());assert cfg['spec']==SPEC and cfg['GT_read'] is False and cfg['smoke']==smoke and cfg['implementation']==IMPLEMENTATION
    rows=selected_rows(smoke);expected={(r['dataset'],r['video_id']) for r in rows}
    rr={name:records(root/name/'predictions.jsonl') for name in ('base','optimized')};assert all(set(r)==expected for r in rr.values())
    old=records(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');j=cpu_position_renderer();asr={ds:load_asr(ds) for ds in DATASETS};result={}
    for ordinal,row in enumerate(rows,1):
        key=row['dataset'],row['video_id'];b=json.loads((root/'records'/key[0]/(key[1]+'.json')).read_text())
        validate_bundle(j,row,asr[key[0]].get(key[1],[]),b,smoke);assert all(b[name]==rr[name][key] for name in rr)
        assert b['base']['extra']['z_video']==old[key]['extra']['z_video'] and b['base']['extra']['windows']==old[key]['extra']['windows']
        assert np.array_equal(b['base']['score_curve'],old[key]['score_curve'])
        c=b['checks'];ds=result.setdefault(key[0],dict(videos=0,source_frames=0,static_tokens=0,dynamic_tokens=0,changed_V=0,remote_reads=0,missing_LOCAL=0,
            actual_forwards=0,actual_vision=0,source_seconds=0.,processing_seconds=0.,peak_GiB=0.,standalone_seconds=0.,clones=0))
        ds['videos']+=1;ds['source_frames']+=c['source_frames'];ds['changed_V']+=sum(t['native_visual']!=t['new_visual'] for t in b['traces'])
        ds['static_tokens']+=sum(r['plan']['static_budget'] for r in b['source_acquisition']['records']);ds['dynamic_tokens']+=sum(r['plan']['dynamic_budget'] for r in b['source_acquisition']['records'])
        ds['remote_reads']+=c['remote_reads'];ds['missing_LOCAL']+=c['missing_LOCAL'];ds['actual_forwards']+=c['actual_forwards'];ds['actual_vision']+=c['actual_vision']
        ds['source_seconds']+=c['source_seconds'];ds['processing_seconds']+=sum(c['times'].values());ds['peak_GiB']=max(ds['peak_GiB'],c['peak_GiB'])
        ds['standalone_seconds']+=b['optimized']['extra']['standalone_seconds'];ds['clones']+=c['diagnostic_forwards']
        if ordinal%10==0:print('BOUND',ordinal,len(rows),flush=True)
    assert all(r['changed_V']>0 and r['remote_reads']>0 and r['static_tokens']>0 and r['dynamic_tokens']>0 for r in result.values()),'both dual paths and actual context intervention must execute; this is not efficacy'
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(dict(PASS=True,GT_read=False,coverage=len(rows),native_allraw_exact=True,datasets=result,
        mechanism_supported=False,cost_scope='all fresh vision/source/quantization/proof/reader work and original source decode charged; saliency nested in source vision, not double counted; native20/fullASR original acquisition additional; setup/validation/I/O/failure wall in attempt paths',
        replay_cpu_threads=__import__('torch').get_num_threads(),generation_cpu_threads=4,
        attempt_paths=[str(p.relative_to(ROOT)) for p in sorted(root.glob('pipeline_attempt_*.json'))]),indent=2)+'\n');print('PREPARE_PASS',flush=True)


def evaluate(root,decoded,name):
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(root/name/'predictions.jsonl'),
        '--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(root/name/'metrics.json')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(root/name),'--datasets',*DATASETS,
        '--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length','--min-windows','2','--bma-grid','6','--arm','m2',
        '--out-root',str(decoded),'--tag',name],cwd=ROOT,check=True)


def metric(path):return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']}


def report(decoded,out):
    base=metric(decoded/'base/metrics.json');new=metric(decoded/'optimized/metrics.json');current=metric(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json')
    result=dict(scope='development-selected; same frozen r6/protocol/evaluator/cohort, descriptive performance only',datasets={},mechanism_supported=False,
        metric_sources={a:str((decoded/a/'metrics.json').relative_to(ROOT)) for a in ('base','optimized')})
    for ds in DATASETS:
        assert all(base[ds][k]==current[ds][k] for k in METRICS)
        assert base[ds]['n_videos_defined']==new[ds]['n_videos_defined']==(84 if ds=='HateMM' else 99)
        result['datasets'][ds]=dict(native={k:base[ds][k] for k in METRICS},optimized={k:new[ds][k] for k in METRICS},delta={k:new[ds][k]-current[ds][k] for k in METRICS})
    common=[k for k in METRICS if all(result['datasets'][ds]['delta'][k]>=.01 for ds in DATASETS)]
    losses=all(result['datasets'][ds]['delta'][k]>=(-.01 if k==METRICS[-1] else -.005) for ds in DATASETS for k in METRICS)
    result['gates']=dict(common_gain_metrics=common,no_losses_beyond_noise=losses,performance_pass=bool(common and losses),
        any_qualifying_gain=any(result['datasets'][ds]['delta'][k]>=.01 for ds in DATASETS for k in METRICS))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);ap.add_argument('--name',choices=('base','optimized'));a=ap.parse_args()
    root=ROOT/'runs/20261007_m1_streamingtom'/('r1_full_smoke' if a.smoke else 'r1_full_main');out=root.parent/(root.name+'_analysis');out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();done=False
    try:
        if a.stage=='prepare':prepare(root,out,a.smoke)
        elif a.stage=='evaluate':
            assert not a.smoke and a.name and json.loads((out/'alignment.json').read_text())['PASS'] is True
            evaluate(root,root.parent/(root.name+'_decoded'),a.name)
        else:assert not a.smoke;report(root.parent/(root.name+'_decoded'),out)
        done=True
    finally:
        n=1;tag=a.stage+('_'+a.name if a.name else '')
        while (out/f'{tag}_attempt_{n:04d}.json').exists():n+=1
        (out/f'{tag}_attempt_{n:04d}.json').write_text(json.dumps(dict(completed=done,elapsed_seconds=time.perf_counter()-start,stage=a.stage),indent=2)+'\n')


if __name__=='__main__':main()
