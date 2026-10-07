"""Binding checks, sole evaluator + fixed r6, and the declared decomposition of control #1 against r6."""
import argparse
import json
import subprocess
import sys
import time
import numpy as np
from scipy.stats import spearmanr
from inputs import ROOT,SPEC,DATASETS,selected_rows,windows_for
from analyze import METRICS,records,metric
from controls_analyze import untimed,curve_ok

EXP=ROOT/'runs/20261007_m1_streamingtom'
ARMS=('custom_native','local_clean','no_remote_replay')
FLOOR={'frame_ROC_AUC':.005,'frame_PR_AUC':.005,'within_video_macro_ROC_AUC':.01}


def prepare(smoke):
    from local_controls import IMPLEMENTATION
    root=EXP/f"local_controls_{'smoke' if smoke else 'main'}";out=root.parent/(root.name+'_analysis');out.mkdir(exist_ok=True)
    cfg=json.loads((root/'config.json').read_text());assert cfg['spec']==SPEC and cfg['GT_read'] is False and cfg['smoke']==smoke and cfg['implementation']==IMPLEMENTATION
    rows=selected_rows(smoke);expected={(r['dataset'],r['video_id']) for r in rows}
    preds={name:records(root/name/'predictions.jsonl') for name in ('base',*ARMS)};assert all(set(p)==expected for p in preds.values())
    r6=records(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');control1=records(EXP/'controls_dualpath_main/no_remote/predictions.jsonl')
    summary={ds:dict(videos=0,windows=0,covered_windows=0,custom_native_exact=0,custom_native_abs=[],seconds={a:0. for a in ARMS},peak_GiB=0.) for ds in DATASETS}
    pairs={ds:([],[]) for ds in DATASETS}
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
        # Control #1 is reproduced exactly in this job.
        replay=b['predictions']['no_remote_replay'];assert np.array_equal(replay['score_curve'],control1[key]['score_curve']) and replay['extra']['windows']==control1[key]['extra']['windows']
        for t,w,nw in zip(b['traces'],windows,b['base']['extra']['windows']):
            assert t['i']==w['i'] and t['bounds']==[w['start'],w['end']] and t['native_visual']==nw['z_visual'] and t['replay_exact']
            assert b['predictions']['custom_native']['extra']['windows'][w['i']]['z_visual']==t['custom_native']
            s['windows']+=1;s['custom_native_exact']+=t['custom_native']==t['native_visual'];s['custom_native_abs'].append(abs(t['custom_native']-t['native_visual']))
            pairs[key[0]][0].append(t['native_visual']);pairs[key[0]][1].append(t['custom_native'])
            for a in ('local_clean','no_remote_replay'):
                if t['LOCAL']:
                    n=len(t['LOCAL']);x=t[a];assert len(x['image_counts'])==n and x['time_labels']==n and x['role_in_suffix']==(a=='no_remote_replay')
                    assert b['predictions'][a]['extra']['windows'][w['i']]['z_visual']==x['z']
                else:
                    assert t[a] is None and b['predictions'][a]['extra']['windows'][w['i']]['z_visual']==t['native_visual']
            s['covered_windows']+=bool(t['LOCAL'])
        ch=b['checks'];s['videos']+=1;s['peak_GiB']=max(s['peak_GiB'],ch['peak_GiB'])
        for a in ARMS:s['seconds'][a]+=ch['times'][a]
        if ordinal%25==0:print('BOUND',ordinal,len(rows),flush=True)
    for ds,s in summary.items():
        d=np.asarray(s.pop('custom_native_abs'));s['custom_native_max_abs']=float(d.max());s['custom_native_mean_abs']=float(d.mean())
        s['custom_native_spearman']=float(spearmanr(*pairs[ds])[0])
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(dict(PASS=True,GT_read=False,coverage=len(rows),
        native_allraw_exact=True,control1_replay_exact=True,datasets=summary),indent=2)+'\n');print('PREPARE_PASS',flush=True)


def evaluate(name):
    root=EXP/'local_controls_main';decoded=root.parent/(root.name+'_decoded')
    assert json.loads((root.parent/(root.name+'_analysis')/'alignment.json').read_text())['PASS'] is True
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(root/name/'predictions.jsonl'),
        '--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(root/name/'metrics.json')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(root/name),'--datasets',*DATASETS,
        '--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length','--min-windows','2','--bma-grid','6','--arm','m2',
        '--out-root',str(decoded),'--tag',name],cwd=ROOT,check=True)


def report():
    decoded=EXP/'local_controls_main_decoded'
    m={'r6':metric(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json'),'control1':metric(EXP/'controls_dualpath_main_decoded/no_remote/metrics.json')}
    for a in ARMS:m[a]=metric(decoded/a/'metrics.json')
    assert all(m['no_remote_replay'][ds][k]==m['control1'][ds][k] for ds in DATASETS for k in METRICS),'replay must reproduce control #1'
    def diff(x,y):return {ds:{k:m[x][ds][k]-m[y][ds][k] for k in METRICS} for ds in DATASETS}
    steps=dict(code_path=diff('custom_native','r6'),frames=diff('local_clean','custom_native'),role_text=diff('control1','local_clean'),total=diff('control1','r6'))
    within_noise=lambda d:all(abs(d[ds][k])<FLOOR[k] for ds in DATASETS for k in METRICS)
    common=[k for k in METRICS if all(steps['frames'][ds][k]>=.01 for ds in DATASETS)]
    no_loss=all(steps['frames'][ds][k]>-FLOOR[k] for ds in DATASETS for k in METRICS)
    result=dict(scope='development-selected; same frozen r6/protocol/evaluator/cohort; rules declared before the run',
        metrics={a:{ds:{k:m[a][ds][k] for k in METRICS} for ds in DATASETS} for a in m},
        sources={'r6':'runs/20260926_twolevel/r6_bma/metrics.json','control1':'runs/20261007_m1_streamingtom/controls_dualpath_main_decoded/no_remote/metrics.json',
            **{a:str((decoded/a/'metrics.json').relative_to(ROOT)) for a in ARMS}},steps=steps,
        verdicts=dict(code_path_within_noise=within_noise(steps['code_path']),role_text_within_noise=within_noise(steps['role_text']),
            frames_common_gain_metrics=common,frames_no_loss_beyond_noise=no_loss,frames_supported=bool(common and no_loss)))
    out=EXP/'local_controls_analysis';out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('LOCAL_ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);ap.add_argument('--name');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=EXP/('local_controls_analysis' if a.stage=='report' else f"local_controls_{'smoke' if a.smoke else 'main'}_analysis");out.mkdir(parents=True,exist_ok=True)
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
