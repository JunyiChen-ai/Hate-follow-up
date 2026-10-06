"""Strict unlabelled HL0 replay, then canonical metrics and factorial contrasts."""
import argparse
import json
import time
from inputs import ROOT,CACHE,DATASETS,selected_rows,validate_input
from history_local import VERSION,folder,validate
from controls import record
from analyze import records,evaluate,metrics,METRICS
from src.mllm_renderer import cpu_position_renderer
from src.video_inputs import load_asr


def prepare(smoke,out):
    root=folder(smoke);cfg=json.loads((root/'config.json').read_text())
    assert cfg['version']==VERSION and cfg['GT_read'] is False and cfg['smoke']==smoke
    rows=selected_rows(smoke);preds=records(root/'HL0/predictions.jsonl')
    assert preds.keys()=={(r['dataset'],r['video_id']) for r in rows}
    j=cpu_position_renderer();asr={ds:load_asr(ds) for ds in DATASETS};datasets={}
    for i,row in enumerate(rows,1):
        ds,video=row['dataset'],row['video_id'];meta=json.loads((CACHE/ds/video/'metadata.json').read_text());validate_input(meta,row)
        bundle=record(root,row);validate(j,row,asr[ds].get(video,[]),meta,bundle,smoke)
        arm=bundle['controls']['L0'];expected=dict(arm['prediction']);expected['method']='m1_rekv_HL0';expected['code_path']='experiments/20261006_m1_rekv/history_local.py'
        assert preds[(ds,video)]==expected
        info=datasets.setdefault(ds,dict(videos=0,source_frames=0,changed_V=0,actual_forwards=0,actual_vision=0,processing_seconds=0.,standalone_seconds=0.,peak_GiB=0.,clones=0))
        c=bundle['checks'];info['videos']+=1;info['source_frames']+=c['source_frames']
        info['changed_V']+=sum(t['new_visual']!=t['native_visual'] for t in arm['traces'])
        info['actual_forwards']+=bundle['batch_actual_forwards'];info['actual_vision']+=bundle['batch_actual_vision']
        info['processing_seconds']+=sum(c['times'].values())+arm['elapsed_seconds'];info['standalone_seconds']+=expected['extra']['standalone_seconds']
        info['clones']+=c['diagnostic_forwards']+arm['diagnostic_forwards'];info['peak_GiB']=max(info['peak_GiB'],bundle['batch_peak_GiB'])
        if i%25==0:print('HL0_BOUND',i,len(rows),flush=True)
    assert set(datasets)==set(DATASETS) and all(d['changed_V']>0 for d in datasets.values())
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(dict(PASS=True,GT_read=False,coverage=len(rows),
        datasets=datasets,fresh_H0_exact=True,no_source_history=True,no_explicit_remote=True,native_global_context_kept=True,mechanism_supported=False,
        cost_attempt_paths=[str(p.relative_to(ROOT)) for p in sorted(root.glob('pipeline_attempt_*.json'))],
        cost_scope='full fresh H0 source rebuild/native checks + H0 replay + independent HL0 queries and diagnostics charged; original native20/fullASR acquisition additional'),indent=2)+'\n')
    print('HL0_PREPARE_PASS',flush=True)


def report(out):
    root=ROOT/'runs/20261006_m1_rekv'
    paths={'HL0':root/'history_local_main_decoded/HL0/metrics.json',
        'main':root/'r1_full_main_decoded/optimized/metrics.json',
        'L0':root/'controls_main_decoded/L0/metrics.json','H0':root/'controls_main_decoded/H0/metrics.json',
        'native':ROOT/'runs/20260926_twolevel/r6_bma/metrics.json'}
    table={arm:metrics(path) for arm,path in paths.items()};result=dict(scope='development-selected factorial diagnostic; no new success gate or promotion',
        metric_sources={a:str(p.relative_to(ROOT)) for a,p in paths.items()},datasets={},mechanism_supported=False)
    for ds in DATASETS:
        assert all(table[a][ds]['n_videos_defined']==(84 if ds=='HateMM' else 99) for a in table)
        contrasts={name:{k:table[left][ds][k]-table[right][ds][k] for k in METRICS}
            for name,left,right in [('history_without_remote','L0','HL0'),('remote_without_history','H0','HL0'),('joint_deletion','main','HL0'),('HL0_minus_native','HL0','native')]}
        contrasts['interaction']={k:(table['main'][ds][k]-table['L0'][ds][k])-contrasts['remote_without_history'][k] for k in METRICS}
        result['datasets'][ds]=dict(metrics={a:{k:table[a][ds][k] for k in METRICS} for a in table},contrasts=contrasts)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('HL0_ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);a=ap.parse_args()
    root=folder(a.smoke);out=root.parent/(root.name+'_analysis');out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();done=False
    try:
        if a.stage=='prepare':prepare(a.smoke,out)
        elif a.stage=='evaluate':
            assert not a.smoke and json.loads((out/'alignment.json').read_text())['PASS'] is True
            evaluate(root,root.parent/'history_local_main_decoded','HL0')
        else:assert not a.smoke;report(out)
        done=True
    finally:
        n=1
        while (out/f'{a.stage}_attempt_{n:04d}.json').exists():n+=1
        (out/f'{a.stage}_attempt_{n:04d}.json').write_text(json.dumps(dict(completed=done,elapsed_seconds=time.perf_counter()-start,stage=a.stage),indent=2)+'\n')


if __name__=='__main__':main()
