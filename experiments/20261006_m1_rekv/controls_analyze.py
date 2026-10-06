"""No-GT control binding, then only canonical raw/fixed-r6 evaluations."""
import argparse
import json
import subprocess
import sys
import time
from inputs import ROOT,CACHE,DATASETS,selected_rows,validate_input
from controls import VERSION,ARMS,MODES,record,equal_r0,validate_selection,validate_variant
from validate import validate_bundle
from analyze import metrics,METRICS
from src.mllm_renderer import cpu_position_renderer
from src.video_inputs import load_asr


def folders(smoke):
    prefix='controls_'+('smoke' if smoke else 'main')
    return {mode:ROOT/'runs/20261006_m1_rekv'/(prefix+'_'+mode) for mode in MODES}


def prepare(smoke,out):
    j=cpu_position_renderer();rows=selected_rows(smoke);asr={d:load_asr(d) for d in DATASETS}
    roots=folders(smoke);main=ROOT/'runs/20261006_m1_rekv/r1_full_main';expected={(r['dataset'],r['video_id']) for r in rows};report={}
    predictions={}
    for mode,root in roots.items():
        cfg=json.loads((root/'config.json').read_text());assert cfg['version']==VERSION and cfg['mode']==mode and cfg['smoke']==smoke and cfg['GT_read'] is False
        names=ARMS if mode=='selection' else ('T0',) if mode=='time' else ('H0',)
        for name in names:
            rr=[json.loads(line) for line in (root/name/'predictions.jsonl').read_text().splitlines()]
            assert len(rr)==len(expected) and {(r['dataset'],r['video_id']) for r in rr}==expected
            predictions[name]={(r['dataset'],r['video_id']):r for r in rr}
        report[mode]=dict(coverage=0,actual_forwards=0,actual_vision=0,source_seconds=0.,processing_seconds=0.,batch_peak_GiB=0.,clones=0,cost_attempt_paths=[],dataset={})
        for p in sorted(root.glob('pipeline_attempt_*.json')):
            report[mode]['cost_attempt_paths'].append(str(p.relative_to(ROOT)))
    for ordinal,row in enumerate(rows,1):
        key=row['dataset'],row['video_id'];reference=record(main,row);meta=json.loads((CACHE/key[0]/key[1]/'metadata.json').read_text());validate_input(meta,row)
        segments=asr[key[0]].get(key[1],[])
        for mode,root in roots.items():
            bundle=record(root,row);checks=bundle['checks'];info=report[mode];ds=info['dataset'].setdefault(key[0],dict(videos=0,remote_choices=0,full_donor_replacements=0,changed_donors=0,timestamp_changed_blocks=0,timestamp_role_changes=0,timestamp_reads=0))
            if mode=='selection':
                validate_bundle(j,row,segments,meta,bundle,smoke);equal_r0(bundle,reference)
                assert set(bundle['controls'])==set(ARMS)
                for name,arm in bundle['controls'].items():
                    validate_selection(bundle,arm,smoke);assert predictions[name][key]==arm['prediction']
                    info['clones']+=arm['diagnostic_forwards']
                for trace in bundle['controls']['D0']['traces']:
                    for read in trace['exposure']:
                        ds['remote_choices']+=bool(read['reference_remote_ids']);ds['full_donor_replacements']+=read['full_donor_replacement'];ds['changed_donors']+=read['changed_remote']
            else:
                validate_variant(j,row,segments,meta,bundle,smoke,mode,reference)
                name='T0' if mode=='time' else 'H0';assert predictions[name][key]==bundle['optimized']
                if mode=='time':
                    ds['timestamp_changed_blocks']+=bundle['intervention']['cross_window_frames']
                    for exposure in bundle['intervention']['layer_exposures']:
                        ds['timestamp_role_changes']+=exposure['role_changes'];ds['timestamp_reads']+=exposure['reads']
            assert bundle['batch_actual_forwards']==checks['actual_forwards']+sum(a['actual_forwards'] for a in bundle.get('controls',{}).values())
            assert bundle['batch_actual_vision']==checks['actual_vision']
            info['coverage']+=1;ds['videos']+=1;info['actual_forwards']+=bundle['batch_actual_forwards'];info['actual_vision']+=bundle['batch_actual_vision']
            info['source_seconds']+=checks['source_seconds'];info['batch_peak_GiB']=max(info['batch_peak_GiB'],bundle['batch_peak_GiB']);info['clones']+=checks['diagnostic_forwards']
            info['processing_seconds']+=sum(checks['times'].values())+sum(a['elapsed_seconds'] for a in bundle.get('controls',{}).values())
        if ordinal%25==0:print('CONTROLS_BOUND',ordinal,len(rows),flush=True)
    if smoke:
        for mode,info in report.items():assert all(d['videos']>0 for d in info['dataset'].values())
        # Controls must have genuine intervention exposure in both corpora;
        # semantic causality remains untested until complete canonical results.
        for ds in report['selection']['dataset'].values():assert ds['changed_donors']>0 and ds['full_donor_replacements']>0
        for ds in report['time']['dataset'].values():assert ds['timestamp_changed_blocks']>0 and ds['timestamp_role_changes']>0
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(dict(PASS=True,GT_read=False,coverage=len(rows),modes=report,
        source_input_cost_scope='all original source decoding charged in each standalone arm; native20/fullASR original extraction remains additional; current validation/setup/I/O and failures are in attempt logs',mechanism_supported=False),indent=2)+'\n')
    print('CONTROLS_PREPARE_PASS',flush=True)


def evaluate(name,decoded):
    mode='selection' if name in ARMS else 'time' if name=='T0' else 'history';root=folders(False)[mode]
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(root/name/'predictions.jsonl'),
        '--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(root/name/'metrics.json')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(root/name),'--datasets',*DATASETS,
        '--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length','--min-windows','2',
        '--bma-grid','6','--arm','m2','--out-root',str(decoded),'--tag',name],cwd=ROOT,check=True)


def report(decoded,out):
    result=dict(scope='development-selected; exact original protocol/evaluator/r6; control results, not independent confirmation',arms={},deletion_gates={},binding_gates={})
    reference=metrics(ROOT/'runs/20261006_m1_rekv/r1_full_main_decoded/optimized/metrics.json')
    eligible=json.loads((ROOT/'runs/20261006_m1_rekv/r1_full_main_analysis/summary.json').read_text())['gates']['common_gain_metrics']
    assert eligible and all(k in METRICS for k in eligible)
    result['main_common_gain_metrics']=eligible
    for arm in ARMS+('T0','H0'):
        current=metrics(decoded/arm/'metrics.json');values={}
        for ds in DATASETS:
            assert current[ds]['n_videos_defined']==reference[ds]['n_videos_defined']
            values[ds]=dict(metrics={k:current[ds][k] for k in METRICS},main_minus_arm={k:reference[ds][k]-current[ds][k] for k in METRICS})
            if arm=='R0':assert all(current[ds][k]==reference[ds][k] for k in METRICS)
        result['arms'][arm]=dict(metric_source=str((decoded/arm/'metrics.json').relative_to(ROOT)),datasets=values)
    for arm in ('L0','L1','L2','H0'):
        common=[k for k in METRICS if all(result['arms'][arm]['datasets'][ds]['main_minus_arm'][k]>=.01 for ds in DATASETS)]
        result['deletion_gates'][arm]=dict(common_one_point_metrics=common,passed=bool(common),main_gain_concordance_metrics=[k for k in common if k in eligible])
    for arm in ('D0','T0'):
        common=[k for k in METRICS if all(result['arms'][arm]['datasets'][ds]['main_minus_arm'][k]>(.01 if k==METRICS[-1] else .005) for ds in DATASETS)]
        result['binding_gates'][arm]=dict(common_above_noise_metrics=common,passed=bool(common),main_gain_concordance_metrics=[k for k in common if k in eligible])
    result['mechanism_supported']=False
    result['scope_note']='component attribution additionally requires matched comparisons, actual exposure, rawV/S/max ordering and independent review; metrics alone cannot certify mechanism'
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('CONTROLS_ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=('prepare','evaluate','report'));ap.add_argument('--smoke',action='store_true');ap.add_argument('--name',choices=ARMS+('T0','H0'));a=ap.parse_args()
    out=ROOT/'runs/20261006_m1_rekv'/('controls_'+('smoke' if a.smoke else 'main')+'_analysis');out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();done=False
    try:
        if a.stage=='prepare':prepare(a.smoke,out)
        elif a.stage=='evaluate':
            assert not a.smoke and a.name
            assert json.loads((out/'alignment.json').read_text())['PASS'] is True
            evaluate(a.name,out.parent/'controls_main_decoded')
        else:
            assert not a.smoke;report(out.parent/'controls_main_decoded',out)
        done=True
    finally:
        label=a.stage+('_'+a.name if a.name else '');number=1
        while (out/f'{label}_attempt_{number:04d}.json').exists():number+=1
        (out/f'{label}_attempt_{number:04d}.json').write_text(json.dumps(dict(stage=a.stage,name=a.name,completed=done,elapsed_seconds=time.perf_counter()-start),indent=2)+'\n')


if __name__=='__main__':main()
