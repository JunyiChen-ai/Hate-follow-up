"""Binding checks, sole evaluator + fixed r6, and the declared readings of the placement controls (2026-10-09)."""
import argparse
import json
import subprocess
import sys
import time
import numpy as np
from inputs import ROOT,SPEC,DATASETS,selected_rows,windows_for
from analyze import METRICS,records,metric
from controls_analyze import untimed,curve_ok

EXP=ROOT/'runs/20261007_m1_streamingtom'
ARMS=('local_clean_replay','adjacent_local','prefix_local','adjacent_native')
FLOOR={'frame_ROC_AUC':.005,'frame_PR_AUC':.005,'within_video_macro_ROC_AUC':.01}


def prepare(smoke):
    from placement_controls import IMPLEMENTATION,LOCAL_LABEL,NATIVE_LABEL
    root=EXP/f"placement_controls_{'smoke' if smoke else 'main'}";out=root.parent/(root.name+'_analysis');out.mkdir(exist_ok=True)
    cfg=json.loads((root/'config.json').read_text());assert cfg['spec']==SPEC and cfg['GT_read'] is False and cfg['smoke']==smoke and cfg['implementation']==IMPLEMENTATION
    rows=selected_rows(smoke);expected={(r['dataset'],r['video_id']) for r in rows}
    preds={name:records(root/name/'predictions.jsonl') for name in ('base',*ARMS)};assert all(set(p)==expected for p in preds.values())
    r6=records(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');stored=records(EXP/'local_controls_main/local_clean/predictions.jsonl')
    summary={ds:dict(videos=0,windows=0,covered_local=0,covered_native=0,replay_abs=[],replay_exact=0,standalone_abs=[],standalone_tokens_equal=0,
        standalone_branch_ids_equal=0,standalone_image_counts_equal=0,prefix_local_tokens=[],seconds={a:0. for a in ('standalone_native',*ARMS)},peak_GiB=0.) for ds in DATASETS}
    for ordinal,row in enumerate(rows,1):
        key=row['dataset'],row['video_id'];b=json.loads((root/'records'/key[0]/(key[1]+'.json')).read_text());s=summary[key[0]]
        assert b['implementation']==IMPLEMENTATION and b['spec']==SPEC and b['checks']['GT_read'] is False
        windows=windows_for(row,[]);assert len(b['traces'])==len(windows)
        assert untimed(b['base'])==untimed(preds['base'][key])
        assert b['base']['extra']['z_video']==r6[key]['extra']['z_video'] and b['base']['extra']['windows']==r6[key]['extra']['windows']
        assert np.array_equal(b['base']['score_curve'],r6[key]['score_curve'])
        for a in ARMS:
            p=b['predictions'][a];assert untimed(p)==untimed(preds[a][key]) and curve_ok(p,float(row['duration']))
            assert p['extra']['z_video']==b['base']['extra']['z_video'] and p['extra']['stance']==b['base']['extra']['stance']
            assert [w.get('z_speech') for w in p['extra']['windows']]==[w.get('z_speech') for w in b['base']['extra']['windows']]
        frames=b['frames'];native_times=b['native']['frame_times'];assert 0<len(native_times)<=SPEC['native_frames']
        for t,w,nw,ew in zip(b['traces'],windows,b['base']['extra']['windows'],stored[key]['extra']['windows']):
            assert t['i']==w['i'] and t['bounds']==[w['start'],w['end']] and t['native_visual']==nw['z_visual']
            local=t['LOCAL'];assert local==[i for i,f in enumerate(frames) if w['start']<=f['time']<w['end']]
            assert t['native_inside']==[x for x in native_times if w['start']<=x<w['end']]
            for a in ARMS:
                has=bool(local) if a!='adjacent_native' else bool(t['native_inside'])
                z=b['predictions'][a]['extra']['windows'][w['i']]['z_visual']
                if has:
                    x=t[a];assert x is not None and z==x['z']
                    if a=='local_clean_replay':assert x['labels']==[LOCAL_LABEL.format(time=frames[i]['time']) for i in local] and len(x['image_counts'])==len(local)
                    elif a=='adjacent_local':assert x['labels']==[NATIVE_LABEL.format(time=frames[i]['time']) for i in local] and len(x['image_counts'])==len(local)
                    elif a=='adjacent_native':assert x['labels']==[NATIVE_LABEL.format(time=v) for v in t['native_inside']] and len(x['image_counts'])==len(t['native_inside'])
                    else:
                        assert x['images']==len(native_times)+len(local) and len(x['inserted_positions'])==len(local) and len(x['image_counts'])==x['images']
                        s['prefix_local_tokens'].append(x['tokens'])
                else:assert t[a] is None and z==t['native_visual']
            # Stored E (local_controls_main) per window: equal to this record's copy; replay distance accumulated.
            if local:
                assert t['stored_local_clean']==ew['z_visual'];d=abs(t['local_clean_replay']['z']-ew['z_visual']);assert t['replay_abs']==d
                s['replay_abs'].append(d);s['replay_exact']+=d==0.
            else:assert t.get('stored_local_clean') is None and ew['z_visual']==t['native_visual']
            s['windows']+=1;s['covered_local']+=bool(local);s['covered_native']+=bool(t['native_inside'])
        g=b['diagnostic'];assert g['cached_visual']==b['base']['extra']['windows'][g['i']]['z_visual'] and g['abs_difference']==abs(g['standalone_visual']-g['cached_visual'])
        s['standalone_abs'].append(g['abs_difference']);s['standalone_tokens_equal']+=g['tokens_equal'];s['standalone_branch_ids_equal']+=g['branch_ids_equal'];s['standalone_image_counts_equal']+=g['image_counts_equal']
        ch=b['checks'];s['videos']+=1;s['peak_GiB']=max(s['peak_GiB'],ch['peak_GiB'])
        for a in s['seconds']:s['seconds'][a]+=ch['times'][a]
        if ordinal%25==0:print('BOUND',ordinal,len(rows),flush=True)
    for ds,s in summary.items():
        d=np.asarray(s.pop('replay_abs'));s['replay_max_abs']=float(d.max()) if len(d) else 0.;s['replay_mean_abs']=float(d.mean()) if len(d) else 0.
        d=np.asarray(s.pop('standalone_abs'));s['standalone_native_max_abs']=float(d.max());s['standalone_native_mean_abs']=float(d.mean())
        d=np.asarray(s.pop('prefix_local_tokens'));s['prefix_local_tokens_mean']=float(d.mean()) if len(d) else 0.;s['prefix_local_tokens_max']=int(d.max()) if len(d) else 0
        assert s['standalone_tokens_equal']==s['standalone_branch_ids_equal']==s['standalone_image_counts_equal']==s['videos'],'standalone input must tokenize as the cached conversation'
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(dict(PASS=True,GT_read=False,coverage=len(rows),native_allraw_exact=True,datasets=summary),indent=2)+'\n')
    print('PREPARE_PASS',flush=True)


def evaluate(name):
    root=EXP/'placement_controls_main';decoded=root.parent/(root.name+'_decoded')
    assert json.loads((root.parent/(root.name+'_analysis')/'alignment.json').read_text())['PASS'] is True
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(root/name/'predictions.jsonl'),
        '--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(root/name/'metrics.json')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(root/name),'--datasets',*DATASETS,
        '--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length','--min-windows','2','--bma-grid','6','--arm','m2',
        '--out-root',str(decoded),'--tag',name],cwd=ROOT,check=True)


def report():
    decoded=EXP/'placement_controls_main_decoded'
    m={'r6':metric(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json'),'local_clean':metric(EXP/'local_controls_main_decoded/local_clean/metrics.json')}
    for a in ARMS:m[a]=metric(decoded/a/'metrics.json')
    binding=json.loads((EXP/'placement_controls_main_analysis/alignment.json').read_text())['datasets']
    def diff(x,y):return {ds:{k:m[x][ds][k]-m[y][ds][k] for k in METRICS} for ds in DATASETS}
    steps=dict(replay=diff('local_clean_replay','local_clean'),wording=diff('adjacent_local','local_clean'),position=diff('adjacent_local','prefix_local'),
        more_frames_only=diff('prefix_local','r6'),adjacency_only=diff('adjacent_native','r6'),adjacent_local_vs_r6=diff('adjacent_local','r6'))
    within_noise=lambda d:all(abs(d[ds][k])<FLOOR[k] for ds in DATASETS for k in METRICS)
    def supported(d):
        common=[k for k in METRICS if all(d[ds][k]>=.01 for ds in DATASETS)];no_loss=all(d[ds][k]>-FLOOR[k] for ds in DATASETS for k in METRICS)
        return dict(common_gain_metrics=common,no_loss_beyond_noise=no_loss,supported=bool(common and no_loss))
    neg={ds:{k:-v for k,v in d.items()} for ds,d in steps['position'].items()}
    position='no_effect' if within_noise(steps['position']) else 'adjacent_better' if supported(steps['position'])['supported'] else 'prefix_better' if supported(neg)['supported'] else 'mixed'
    result=dict(scope='development-selected; same frozen r6/protocol/evaluator/cohort; readings declared before the run (README, 2026-10-09)',
        metrics={a:{ds:{k:m[a][ds][k] for k in METRICS} for ds in DATASETS} for a in m},
        sources={'r6':'runs/20260926_twolevel/r6_bma/metrics.json','local_clean':'runs/20261007_m1_streamingtom/local_controls_main_decoded/local_clean/metrics.json',
            **{a:str((decoded/a/'metrics.json').relative_to(ROOT)) for a in ARMS}},steps=steps,
        per_window={ds:{k:binding[ds][k] for k in ('replay_max_abs','replay_mean_abs','replay_exact','covered_local','covered_native','standalone_native_max_abs','prefix_local_tokens_mean','prefix_local_tokens_max','seconds','peak_GiB')} for ds in DATASETS},
        verdicts=dict(replay_within_noise=within_noise(steps['replay']),wording_within_noise=within_noise(steps['wording']),position=position,
            more_frames_only=supported(steps['more_frames_only']),adjacency_only=supported(steps['adjacency_only']),adjacent_local_vs_r6=supported(steps['adjacent_local_vs_r6'])))
    out=EXP/'placement_analysis';out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('PLACEMENT_ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);ap.add_argument('--name');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=EXP/('placement_analysis' if a.stage=='report' else f"placement_controls_{'smoke' if a.smoke else 'main'}_analysis");out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();done=False
    try:
        if a.stage=='prepare':prepare(a.smoke)
        elif a.stage=='evaluate':
            assert not a.smoke and a.name in ARMS;evaluate(a.name)
        else:report()
        done=True
    finally:
        n=1;tag=a.stage+('_'+a.name if a.name else '')
        while (out/f'{tag}_attempt_{n:04d}.json').exists():n+=1
        (out/f'{tag}_attempt_{n:04d}.json').write_text(json.dumps(dict(completed=done,elapsed_seconds=time.perf_counter()-start,stage=a.stage),indent=2)+'\n')


if __name__=='__main__':main()
