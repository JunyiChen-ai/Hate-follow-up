#!/usr/bin/env python3
"""Current-input/parity first; canonical evaluation and postscore control analysis."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.mllm_renderer import cpu_renderer
from src.video_inputs import load_asr
from src.eval.evaluate import within_video_macro
from extract import CACHE,DATASETS,selected_rows,validate_cached
from control_inputs import ARMS
from control_extract import CONTROL_CACHE,CONTROL_VERSION,validate_control
from control_measure import validate_bundle,BindingRenderer
from analyze import METRICS,read,metrics,evaluate,bootstrap


def equivalent(a,b):
    for key in ('dataset','video_id','duration','native_rate','score_curve','error','calls'):assert a[key]==b[key]
    for key in ('z_video','stance','prefix_tokens','windows'):assert a['extra'][key]==b['extra'][key]


def prepare(root,out,smoke):
    rows=selected_rows(smoke);expected={(r['dataset'],r['video_id']):r for r in rows}
    config=json.loads((root/'config.json').read_text())
    assert config['GT_read'] is False and config['smoke']==smoke and config['control_version']==CONTROL_VERSION
    assert config['arms']==list(ARMS)
    predictions={name:read(root/name/'predictions.jsonl') for name in ('base',)+ARMS}
    assert all(set(rr)==set(expected) for rr in predictions.values())
    main=ROOT/'runs/20261004_m1_tree'/('r3_full_smoke' if smoke else 'r3_full_main')
    reference={name:read(main/name/'predictions.jsonl') for name in ('base','optimized')}
    asr={ds:load_asr(ds) for ds in DATASETS};renderer=BindingRenderer(cpu_renderer());cost=[];changed={a:0 for a in ARMS if a!='main'}
    binding_coverage=[]
    for key,row in expected.items():
        metadata,features=validate_cached(CACHE/key[0]/key[1],row)
        controls=validate_control(CONTROL_CACHE/key[0]/key[1],row,metadata,features,renderer.tok)
        bundle=json.loads((root/'records'/key[0]/(key[1]+'.json')).read_text())
        validate_bundle(row,bundle,metadata,controls,asr[key[0]].get(key[1],[]),renderer,smoke)
        assert predictions['base'][key]==bundle['base']
        for arm in ARMS:assert predictions[arm][key]==bundle['arms'][arm]
        equivalent(predictions['base'][key],reference['base'][key]);equivalent(predictions['main'][key],reference['optimized'][key])
        for arm in changed:
            changed[arm]+=sum(a!=b for a,b in zip(predictions[arm][key]['extra']['windows'],predictions['main'][key]['extra']['windows']))
        cost.append(dict(dataset=key[0],video_id=key[1],**bundle['checks']))
        parents={n['parent'] for n in metadata['tree']['nodes'] if n['parent'] is not None}
        leaves=[n for n in metadata['tree']['nodes'] if n['id'] not in parents]
        windows=[]
        for i,(original,wrong) in enumerate(zip(controls['arms']['main']['packets'],controls['arms']['wrong_links']['packets'])):
            assert original['pool_members']==wrong['pool_members'] and original['leaf_ids']==wrong['leaf_ids']
            binding_changed=(original['ancestor_ids']!=wrong['ancestor_ids'] or original['context']!=wrong['context'])
            reasons=[]
            if not binding_changed:
                if len(leaves)<=1:reasons.append('one_or_fewer_terminal_leaves')
                if not original['leaf_ids']:reasons.append('no_selected_leaf')
                if not original['ancestor_ids']:reasons.append('no_main_ancestor')
                if original['ancestor_ids']==wrong['ancestor_ids']:reasons.append('same_donor_ancestor_chain')
            windows.append(dict(window=i,binding_changed=binding_changed,unchanged_reasons=reasons,
                main_ancestors=original['ancestor_ids'],donor_ancestors=wrong['ancestor_ids'],
                shared_inventory_changed=False,local_pixels_fixed=True))
        binding_coverage.append(dict(dataset=key[0],video_id=key[1],terminal_leaves=len(leaves),windows=windows))
    summary=dict(coverage=len(rows),GT_read=False,native_and_main_raw_exact=True,changed_windows=changed,
        main_metric_results_not_in_reader=True,mechanism_supported=False,cost_per_video=cost,
        wrong_link_binding_coverage=binding_coverage)
    (out/'alignment.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='cost_per_video'},indent=2),flush=True)
    print('CONTROL_PREPARE_DONE',flush=True)


def report(root,decoded,out):
    names=('base',)+ARMS;mm={a:metrics(decoded/a/'metrics.json') for a in names}
    main_old=metrics(ROOT/'runs/20261004_m1_tree/r3_full_main_decoded/optimized/metrics.json')
    baseline=metrics(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json')
    raw={a:read(root/a/'predictions.jsonl') for a in names};final={a:read(decoded/a/'predictions.jsonl') for a in names}
    alignment=json.loads((out/'alignment.json').read_text())
    assert alignment['native_and_main_raw_exact'] is True and alignment['coverage']==333
    result=dict(scope='development-selected; controls at fixed original topology; unchanged r6',
        metric_sources={a:str((decoded/a/'metrics.json').relative_to(ROOT)) for a in names},datasets={},mechanism_supported=False)
    all_video=[]
    for ds in DATASETS:
        for m in METRICS:
            assert mm['base'][ds][m]==baseline[ds][m] and mm['main'][ds][m]==main_old[ds][m]
        with np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True) as gt:
            ys={str(v):np.asarray(gt['y4'][i]) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        paired={arm:{kind:[] for kind in ('final','raw_max','raw_visual','raw_speech_shared')} for arm in names if arm!='main'}
        for key in [k for k in raw['main'] if k[0]==ds]:
            y=ys[key[1]];main=raw['main'][key];idx=np.clip(((np.arange(len(main['score_curve']))+.5)/4//8).astype(int),0,len(main['extra']['windows'])-1)
            use=np.asarray(['z_speech' in w for w in main['extra']['windows']])[idx];values={}
            for arm in names:
                r=raw[arm][key];ww=r['extra']['windows']
                curves=dict(final=np.asarray(final[arm][key]['score_curve']),raw_max=np.asarray(r['score_curve']),
                    raw_visual=np.asarray([w['z_visual'] for w in ww])[idx])
                values[arm]={kind:within_video_macro({key[1]:y},{key[1]:z})[METRICS[-1]] for kind,z in curves.items()}
                speech=np.asarray([w.get('z_speech',0.) for w in ww])[idx];n=min(len(y),len(speech));shared=use[:n]
                values[arm]['raw_speech_shared']=within_video_macro({key[1]:y[:n][shared]},{key[1]:speech[:n][shared]})[METRICS[-1]]
            if values['main']['final'] is not None:
                all_video.append(dict(dataset=ds,video_id=key[1],values=values))
                for arm,kinds in paired.items():
                    for kind in kinds:
                        if values['main'][kind] is not None and values[arm][kind] is not None:kinds[kind].append(values['main'][kind]-values[arm][kind])
        result['datasets'][ds]=dict(final={a:{m:mm[a][ds][m] for m in METRICS} for a in names},
            main_minus_arm={a:{m:mm['main'][ds][m]-mm[a][ds][m] for m in METRICS} for a in names if a!='main'},
            paired_within={a:{k:bootstrap(v) for k,v in kinds.items()} for a,kinds in paired.items()})
    component_gains={a:[m for m in METRICS if all(result['datasets'][ds]['main_minus_arm'][a][m]>=.01 for ds in DATASETS)] for a in names if a!='main'}
    common_temporal=[m for m in METRICS if all(m in component_gains[a] for a in ('temporal','temporal_fresh_priority'))]
    native_gain=component_gains['base'];losses=all(result['datasets'][ds]['main_minus_arm']['base'][m]>=(-.01 if m==METRICS[-1] else -.005) for ds in DATASETS for m in METRICS)
    result['gates']=dict(performance_pass=bool(native_gain and losses),native_common_gain_metrics=native_gain,
        component_common_gain_metrics=component_gains,complete_acquisition_common_gain_metrics=common_temporal,
        final_mechanism_review_required=True)
    result['cost']={}
    result['wrong_link_binding_coverage']={}
    for ds in DATASETS:
        cc=[r for r in alignment['cost_per_video'] if r['dataset']==ds]
        vv=[r for r in alignment['wrong_link_binding_coverage'] if r['dataset']==ds]
        windows=[w for r in vv for w in r['windows']]
        result['wrong_link_binding_coverage'][ds]=dict(videos=len(vv),windows=len(windows),
            singleton_videos=sum(r['terminal_leaves']<=1 for r in vv),
            actual_binding_changed_windows=sum(w['binding_changed'] for w in windows),
            actual_binding_unchanged_windows=sum(not w['binding_changed'] for w in windows),
            unchanged_reason_counts={reason:sum(reason in w['unchanged_reasons'] for w in windows) for reason in (
                'one_or_fewer_terminal_leaves','no_selected_leaf','no_main_ancestor','same_donor_ancestor_chain')})
        result['cost'][ds]=dict(
            main_input_seconds=sum(c['main_input_preprocessing_seconds'] for c in cc),
            additional_joint_input_seconds=sum(c['additional_control_acquisition']['acquisition_seconds'] for c in cc),
            actual_joint_read_seconds=sum(c['reader_joint_wall_seconds'] for c in cc),
            actual_joint_reader_forwards=sum(c['actual_joint_forwards'] for c in cc),
            peak_reader_GiB=max(c['peak_reader_GiB'] for c in cc),
            arms={a:{k:sum(c['arm_costs'][a][k] for c in cc) for k in (
                'actual_forwards','diagnostic_forwards','cache_copy_seconds','extension_seconds','visual_seconds',
                'speech_seconds','diagnostic_seconds','control_input_acquisition_seconds')} for a in ARMS})
    result['scope_limits']=[
        'Temporal controls inherit actual main topology/member counts; fresh priority changes selection only, not adaptive re-expansion.',
        'wrong_links rewires local ancestor packets; the shared factual inventory still exposes correct global links.',
        'Components without a dual-corpus common metric drop>=.01 must be demoted; breadth is not separately tested.',
        'Raw ordering, paired uncertainty, actual interventions and independent final review determine supported mechanism; gates alone do not prove it.']
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'per_video.json').write_text(json.dumps(all_video,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True);print('CONTROL_ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--name',choices=('base',)+ARMS);a=ap.parse_args()
    root=ROOT/'runs/20261004_m1_tree'/('r3_controls_'+('smoke' if a.smoke else 'main'))
    out=root.parent/(root.name+'_analysis');out.mkdir(parents=True,exist_ok=True);decoded=root.parent/(root.name+'_decoded')
    if a.stage=='prepare':prepare(root,out,a.smoke)
    elif a.stage=='evaluate':assert not a.smoke and a.name;evaluate(root,decoded,a.name)
    else:assert not a.smoke;report(root,decoded,out)


if __name__=='__main__':main()
