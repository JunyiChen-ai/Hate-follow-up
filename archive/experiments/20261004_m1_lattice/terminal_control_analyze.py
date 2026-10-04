#!/usr/bin/env python3
"""Source/parity validation precedes unchanged canonical control evaluation."""
import argparse
import json
import numpy as np

from extract import ROOT, CACHE, DATASETS, selected_rows
from terminal_control_reader import ARMS, CONTROL_VERSION
from terminal_control_measure import validate_bundle, equivalent
from analyze import METRICS, metrics, read, evaluate, bootstrap
from src.mllm_renderer import cpu_renderer
from src.video_inputs import load_asr
from src.eval.evaluate import within_video_macro


def prepare(root, out, smoke):
    cfg = json.loads((root/'config.json').read_text())
    assert cfg['GT_read'] is False and cfg['smoke'] == smoke
    assert cfg['control_version'] == CONTROL_VERSION and cfg['arms'] == list(ARMS)
    rows = selected_rows(smoke); expected = {(r['dataset'],r['video_id']) for r in rows}
    names = ('base',) + ARMS
    predictions = {a:read(root/a/'predictions.jsonl') for a in names}
    assert all(set(r) == expected for r in predictions.values())
    main = ROOT/'runs/20261004_m1_lattice'/('r3_full_' + ('smoke' if smoke else 'main'))
    reference = {a:read(main/a/'predictions.jsonl') for a in ('base', 'optimized')}
    asr = {ds:load_asr(ds) for ds in DATASETS}; renderer = cpu_renderer()
    costs = []; coverage = []; changed = {a:0 for a in ARMS if a != 'full'}
    for row in rows:
        key = row['dataset'], row['video_id']
        metadata = json.loads((CACHE/key[0]/(key[1]+'.json')).read_text())
        bundle = json.loads((root/'records'/key[0]/(key[1]+'.json')).read_text())
        validate_bundle(row, bundle, metadata, asr[key[0]].get(key[1],[]), renderer, smoke)
        assert predictions['base'][key] == bundle['base']
        for a in ARMS:
            assert predictions[a][key] == bundle['arms'][a]
        equivalent(predictions['base'][key], reference['base'][key])
        equivalent(predictions['full'][key], reference['optimized'][key])
        for a in changed:
            changed[a] += sum(w != z for w,z in zip(predictions[a][key]['extra']['windows'], predictions['full'][key]['extra']['windows']))
        costs.append(dict(dataset=key[0], video_id=key[1], **bundle['checks']))
        coverage.append(dict(dataset=key[0], video_id=key[1],
            mass=[t['coverage'] for t in bundle['traces']['wrong_mass'] if t['available']],
            source=[dict(destination=t['destination_window'], donor=t['source_window'],
                actual_interval=t['source_actual_interval'], changed=t['coverage']['source_binding_changed'])
                for t in bundle['traces']['wrong_audio_window'] if t['available']]))
    result = dict(coverage=len(rows), GT_read=False, native_and_full_exact=True,
        changed_score_windows=changed, intervention_coverage=coverage, cost_per_video=costs,
        mechanism_supported=False)
    (out/'alignment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cost_per_video','intervention_coverage')},indent=2),flush=True)
    print('CONTROL_PREPARE_DONE',flush=True)


def report(root, decoded, out):
    names = ('base',) + ARMS
    alignment = json.loads((out/'alignment.json').read_text())
    assert alignment['coverage'] == 333 and alignment['native_and_full_exact'] is True
    mm = {a:metrics(decoded/a/'metrics.json') for a in names}
    native = metrics(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json')
    main = metrics(ROOT/'runs/20261004_m1_lattice/r3_full_main_decoded/optimized/metrics.json')
    raw = {a:read(root/a/'predictions.jsonl') for a in names}
    final = {a:read(decoded/a/'predictions.jsonl') for a in names}
    result = dict(scope='development-selected; matched end-state controls, fixed r6',
        metric_sources={a:str((decoded/a/'metrics.json').relative_to(ROOT)) for a in names},
        datasets={}, mechanism_supported=False)
    all_video=[]
    for ds in DATASETS:
        assert all(mm['base'][ds][m] == native[ds][m] and mm['full'][ds][m] == main[ds][m] for m in METRICS)
        with np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True) as gt:
            ys={str(v):np.asarray(gt['y4'][i]) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        paired={a:{kind:[] for kind in ('final','raw_max','raw_visual','raw_speech_shared')} for a in names if a != 'full'}
        for key in [k for k in raw['full'] if k[0]==ds]:
            y=ys[key[1]]; nw=len(raw['full'][key]['extra']['windows'])
            idx=np.clip(((np.arange(len(raw['full'][key]['score_curve']))+.5)/4//8).astype(int),0,nw-1)
            values={}; available={a:np.asarray(['z_speech' in w for w in raw[a][key]['extra']['windows']])[idx] for a in names}
            # Common availability is explicit; native can have speech with missing new audio.
            shared=np.logical_and.reduce(list(available.values()))
            for a in names:
                ww=raw[a][key]['extra']['windows']
                curves=dict(final=np.asarray(final[a][key]['score_curve']),raw_max=np.asarray(raw[a][key]['score_curve']),
                    raw_visual=np.asarray([w['z_visual'] for w in ww])[idx])
                values[a]={kind:within_video_macro({key[1]:y},{key[1]:z})[METRICS[-1]] for kind,z in curves.items()}
                z=np.asarray([w.get('z_speech',0.) for w in ww])[idx];n=min(len(y),len(z));use=shared[:n]
                values[a]['raw_speech_shared']=within_video_macro({key[1]:y[:n][use]},{key[1]:z[:n][use]})[METRICS[-1]]
            if values['full']['final'] is not None:
                all_video.append(dict(dataset=ds,video_id=key[1],values=values))
                for a, kinds in paired.items():
                    for kind in kinds:
                        if values['full'][kind] is not None and values[a][kind] is not None:
                            kinds[kind].append(values['full'][kind]-values[a][kind])
        result['datasets'][ds]=dict(final={a:{m:mm[a][ds][m] for m in METRICS} for a in names},
            n_eligible=len(paired['base']['final']),
            full_minus_arm={a:{m:mm['full'][ds][m]-mm[a][ds][m] for m in METRICS} for a in paired},
            paired_within={a:{kind:bootstrap(v) for kind,v in kinds.items()} for a,kinds in paired.items()})
        assert result['datasets'][ds]['n_eligible'] == (84 if ds == 'HateMM' else 99)
    gains={a:[m for m in METRICS if all(result['datasets'][ds]['full_minus_arm'][a][m]>=.01 for ds in DATASETS)] for a in names if a!='full'}
    common=[m for m in METRICS if all(m in gains[a] for a in ('onebest','flat'))]
    losses=all(result['datasets'][ds]['full_minus_arm']['base'][m]>=(-.01 if m==METRICS[-1] else -.005) for ds in DATASETS for m in METRICS)
    result['gates']=dict(performance_pass=bool(gains['base'] and losses),component_common_gain_metrics=gains,
        structural_common_gain_metrics=common,terminal_common_gain_metrics=gains['allword'],
        mechanism_candidate_common_gain_metrics=[m for m in common if m in gains['allword']],final_independent_mechanism_review_required=True)
    result['intervention_coverage']=alignment['intervention_coverage']
    result['cost']={}
    for ds in DATASETS:
        cc=[c for c in alignment['cost_per_video'] if c['dataset']==ds]
        result['cost'][ds]=dict(physical_read_seconds=sum(c['physical_read_seconds'] for c in cc),
            shared_input_seconds=sum(c['preprocessing_seconds'] for c in cc),
            actual_forwards=sum(c['actual_forwards'] for c in cc),diagnostic_forwards=sum(c['diagnostic_forwards'] for c in cc),
            input_actual_forwards={k:sum(c['input_actual_forwards'][k] for c in cc) for k in ('encoder','decoder')},
            peak_GiB=max(c['peak_GiB'] for c in cc),
            arms={a:{k:sum(c['arms'][a][k] for c in cc) for k in ('speech_seconds','diagnostic_seconds','actual_forwards','diagnostic_forwards','standalone_seconds')} for a in ARMS})
    result['scope_limits']=['Native full transcript/overview remains visible.',
        'Wrong-source control has truthful donor coordinates and an additional destination instruction; not perfectly text matched.',
        'Zero-mass/singleton paths are unchanged in mass control; actual coverage separately recorded.',
        'Gate numbers alone do not establish mechanism; raw ordering, uncertainty, input interventions and independent final review remain required.']
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'per_video.json').write_text(json.dumps(all_video,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='intervention_coverage'},indent=2),flush=True)
    print('CONTROL_ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--name',choices=('base',)+ARMS);args=ap.parse_args()
    root=ROOT/'runs/20261004_m1_lattice'/('r3_controls_'+('smoke' if args.smoke else 'main'))
    out=root.parent/(root.name+'_analysis');out.mkdir(parents=True,exist_ok=True);decoded=root.parent/(root.name+'_decoded')
    if args.stage=='prepare':prepare(root,out,args.smoke)
    elif args.stage=='evaluate':
        assert not args.smoke and args.name;evaluate(root,decoded,args.name)
    else:
        assert not args.smoke;report(root,decoded,out)


if __name__=='__main__':main()
