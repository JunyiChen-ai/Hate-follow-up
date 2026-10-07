"""R1 control binding checks, sole evaluator + fixed r6, and the declared deletion gates."""
import argparse
import json
import subprocess
import sys
import time
import numpy as np
from inputs import ROOT,SPEC,DATASETS,selected_rows,windows_for,frames_for
from analyze import METRICS,records,metric

EXP=ROOT/'runs/20261007_m1_streamingtom'
JOBS=dict(dualpath=('replay','no_remote','nearest'),uniform=('uniform',))
CONTROLS=('no_remote','nearest','uniform')


def untimed(p):
    return {**p,'extra':{k:v for k,v in p['extra'].items() if k!='standalone_seconds'}}


def curve_ok(p,duration):
    ww=p['extra']['windows'];idx=np.clip(((np.arange(int(np.ceil(duration*4)))+.5)/4//SPEC['window_seconds']).astype(int),0,len(ww)-1)
    return np.array_equal(p['score_curve'],np.asarray([w['z'] for w in ww])[idx]) and all(
        w['z']==(max(w['z_visual'],w['z_speech']) if 'z_speech' in w else w['z_visual']) for w in ww)


def prepare(job,smoke):
    from controls import nearest,IMPLEMENTATION
    from src.pre_rotary_memory import select
    import torch
    root=EXP/f"controls_{job}_{'smoke' if smoke else 'main'}";out=root.parent/(root.name+'_analysis');out.mkdir(exist_ok=True)
    cfg=json.loads((root/'config.json').read_text())
    assert cfg['spec']==SPEC and cfg['GT_read'] is False and cfg['smoke']==smoke and cfg['implementation']==IMPLEMENTATION and cfg['arms']==list(JOBS[job])
    rows=selected_rows(smoke);expected={(r['dataset'],r['video_id']) for r in rows}
    preds={name:records(root/name/'predictions.jsonl') for name in ('base',*JOBS[job])};assert all(set(p)==expected for p in preds.values())
    r6=records(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');r1=records(EXP/'r1_full_main/optimized/predictions.jsonl')
    summary={ds:dict(videos=0,source_frames=0,covered_windows=0,missing_LOCAL=0,source_seconds=0.,peak_GiB=0.,actual_forwards=0,
        arm_seconds={a:0. for a in JOBS[job]},comparison_failures=[],changed_vs_replay={a:0 for a in JOBS[job]}) for ds in DATASETS}
    overlap=dict(layers=0,shared=0,identical=0)
    for ordinal,row in enumerate(rows,1):
        key=row['dataset'],row['video_id'];b=json.loads((root/'records'/key[0]/(key[1]+'.json')).read_text());s=summary[key[0]]
        assert b['job']==job and b['implementation']==IMPLEMENTATION and b['spec']==SPEC and b['checks']['GT_read'] is False
        _,frames=frames_for(row);assert b['frames']==frames
        windows=windows_for(row,[]);assert len(b['traces'])==len(windows)
        # Native reads reproduce r6 exactly; every arm keeps the native S and G.
        assert untimed(b['base'])==untimed(preds['base'][key])
        assert b['base']['extra']['z_video']==r6[key]['extra']['z_video'] and b['base']['extra']['windows']==r6[key]['extra']['windows']
        assert np.array_equal(b['base']['score_curve'],r6[key]['score_curve'])
        for a in JOBS[job]:
            p=b['predictions'][a];assert untimed(p)==untimed(preds[a][key]) and curve_ok(p,float(row['duration']))
            assert p['extra']['z_video']==b['base']['extra']['z_video'] and p['extra']['stance']==b['base']['extra']['stance']
            assert [w.get('z_speech') for w in p['extra']['windows']]==[w.get('z_speech') for w in b['base']['extra']['windows']]
        c=b['comparison'];flags=['features_exact','deepstack_exact','saliency_exact']+(['plan_exact','representatives_exact','replay_queries_exact'] if job=='dualpath' else [])
        if not all(c[f] for f in flags):s['comparison_failures'].append(dict(video=key[1],**{f:c[f] for f in flags}))
        if job=='uniform':
            reps=np.load(ROOT/b['vectors']['source_keys'],allow_pickle=False);queries=np.load(ROOT/b['vectors']['question_vectors'],allow_pickle=False)
            for p in b['plans']:
                n=p['original_tokens'];g=min(50,n);assert p['roots']==[(2*k+1)*n//(2*g) for k in range(g)] and p['static_budget']==0 and p['dynamic_budget']==g
        for t,w,nw in zip(b['traces'],windows,b['base']['extra']['windows']):
            assert t['i']==w['i'] and t['bounds']==[w['start'],w['end']] and t['native_visual']==nw['z_visual']
            local=[i for i,f in enumerate(frames) if w['start']<=f['time']<w['end']];assert t['LOCAL']==local
            for a in JOBS[job]:assert b['predictions'][a]['extra']['windows'][w['i']]['z_visual']==(t['arms'][a]['z'] if local else t['native_visual'])
            if not local:
                assert all(v is None for v in t['arms'].values());continue
            if job=='dualpath':
                near=nearest(frames,local,w);assert t['nearest_ids']==near
                assert all(ids==near for ids in t['arms']['nearest']['remote_ids'])
                assert all(ids==[] for ids in t['arms']['no_remote']['remote_ids']) and set(t['arms']['no_remote']['source_tokens'])=={0}
                for ids in t['arms']['replay']['remote_ids']:
                    overlap['layers']+=1;overlap['shared']+=len(set(ids)&set(near));overlap['identical']+=set(ids)==set(near)
                for a in JOBS[job]:s['changed_vs_replay'][a]+=t['arms'][a]['z']!=t['arms']['replay']['z']
            else:
                for l,ids in enumerate(t['arms']['uniform']['remote_ids']):
                    chosen,_=select(torch.from_numpy(queries[t['i'],l]),torch.from_numpy(reps[l]),[f['index'] for f in frames],set(local),SPEC['remote_frames_per_layer'])
                    assert ids==chosen
        if job=='dualpath':
            # In-job R1 replay against the persisted complete R1 run.
            replay=b['predictions']['replay'];assert np.array_equal(replay['score_curve'],r1[key]['score_curve']) and replay['extra']['windows']==r1[key]['extra']['windows']
        ch=b['checks'];s['videos']+=1;s['source_frames']+=ch['source_frames'];s['covered_windows']+=ch['covered_windows'];s['missing_LOCAL']+=ch['missing_LOCAL']
        s['source_seconds']+=ch['source_seconds'];s['peak_GiB']=max(s['peak_GiB'],ch['peak_GiB']);s['actual_forwards']+=ch['actual_forwards']
        for a in JOBS[job]:s['arm_seconds'][a]+=ch['times'][a]
        if ordinal%25==0:print('BOUND',ordinal,len(rows),flush=True)
    # Every rebuilt input must equal R1 exactly; otherwise an arm would differ from R1 in more than its one change.
    exact=not any(s['comparison_failures'] for s in summary.values())
    result=dict(PASS=exact,GT_read=False,job=job,coverage=len(rows),native_allraw_exact=True,datasets=summary,
        replay_equals_r1=job=='dualpath',comparison_all_exact=exact,
        nearest_vs_retrieval=None if job!='dualpath' else dict(overlap,mean_shared_of_4=overlap['shared']/max(1,overlap['layers']),identical_fraction=overlap['identical']/max(1,overlap['layers'])),
        attempt_paths=[str(p.relative_to(ROOT)) for p in sorted(root.glob('pipeline_attempt_*.json'))])
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(result,indent=2)+'\n');print('PREPARE_PASS' if exact else 'PREPARE_FAILED_inexact_rebuild',job,flush=True);assert exact


def evaluate(job,name):
    root=EXP/f'controls_{job}_main';decoded=root.parent/(root.name+'_decoded')
    assert json.loads((root.parent/(root.name+'_analysis')/'alignment.json').read_text())['PASS'] is True
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(root/name/'predictions.jsonl'),
        '--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(root/name/'metrics.json')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(root/name),'--datasets',*DATASETS,
        '--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length','--min-windows','2','--bma-grid','6','--arm','m2',
        '--out-root',str(decoded),'--tag',name],cwd=ROOT,check=True)


def report():
    r1=metric(EXP/'r1_full_main_decoded/optimized/metrics.json');r6=metric(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json')
    arms={}
    for job,names in JOBS.items():
        for name in names:
            path=EXP/f'controls_{job}_main_decoded'/name/'metrics.json'
            if path.exists():arms[name]=(path,metric(path))
    assert 'replay' in arms and all(arms['replay'][1][ds][k]==r1[ds][k] for ds in DATASETS for k in METRICS),'replay must reproduce R1'
    result=dict(scope='development-selected; same frozen r6/protocol/evaluator/cohort; deletion gate declared before the control runs',
        reference=dict(R1='runs/20261007_m1_streamingtom/r1_full_main_decoded/optimized/metrics.json',r6='runs/20260926_twolevel/r6_bma/metrics.json'),arms={})
    for name,(path,m) in arms.items():
        entry=dict(source=str(path.relative_to(ROOT)),datasets={})
        for ds in DATASETS:
            assert m[ds]['n_videos_defined']==(84 if ds=='HateMM' else 99)
            entry['datasets'][ds]=dict(metrics={k:m[ds][k] for k in METRICS},loss_vs_R1={k:r1[ds][k]-m[ds][k] for k in METRICS},delta_vs_r6={k:m[ds][k]-r6[ds][k] for k in METRICS})
        if name in CONTROLS:
            common=[k for k in METRICS if all(entry['datasets'][ds]['loss_vs_R1'][k]>=.01 for ds in DATASETS)]
            entry['gate']=dict(rule='R1 minus control >= .01 on the same primary metric in both corpora',common_loss_metrics=common,component_supported=bool(common))
        result['arms'][name]=entry
    result['complete']=all(c in result['arms'] for c in CONTROLS)
    out=EXP/'controls_analysis';out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('CONTROLS_ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True)
    ap.add_argument('--job',choices=tuple(JOBS));ap.add_argument('--name');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    start=time.perf_counter();done=False
    out=EXP/('controls_analysis' if a.stage=='report' else f"controls_{a.job}_{'smoke' if a.smoke else 'main'}_analysis");out.mkdir(parents=True,exist_ok=True)
    try:
        if a.stage=='prepare':prepare(a.job,a.smoke)
        elif a.stage=='evaluate':
            assert not a.smoke and a.name in JOBS[a.job];evaluate(a.job,a.name)
        else:report()
        done=True
    finally:
        n=1;tag=a.stage+('_'+a.name if a.name else '')
        while (out/f'{tag}_attempt_{n:04d}.json').exists():n+=1
        (out/f'{tag}_attempt_{n:04d}.json').write_text(json.dumps(dict(completed=done,elapsed_seconds=time.perf_counter()-start,stage=a.stage,job=a.job),indent=2)+'\n')


if __name__=='__main__':main()
