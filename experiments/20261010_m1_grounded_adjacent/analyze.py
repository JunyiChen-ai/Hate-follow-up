"""Candidate 41: binding checks, offline controls (random acceptance, inverted rule), sole evaluator + fixed r6, declared report."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from grounded import EXP,SPEC,DATASETS,IMPLEMENTATION,PREDICTIONS,selected_rows,windows_for,prediction

METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')
FLOOR={'frame_ROC_AUC':.005,'frame_PR_AUC':.005,'within_video_macro_ROC_AUC':.01}
STORED=ROOT/'runs/20261007_m1_streamingtom/placement_controls_main'
CONTROLS=('random_0','random_1','inverted')
ARMS=('grounded','accept_all',*CONTROLS)


def records(path):
    rows={}
    for line in path.open():
        r=json.loads(line);key=r['dataset'],r['video_id'];assert key not in rows;rows[key]=r
    return rows


def untimed(p):return {**p,'extra':{k:v for k,v in p['extra'].items() if k!='standalone_seconds'}}


def curve_ok(p,duration):
    ww=p['extra']['windows'];idx=np.clip(((np.arange(int(np.ceil(duration*4)))+.5)/4//SPEC['window_seconds']).astype(int),0,len(ww)-1)
    return np.array_equal(p['score_curve'],np.asarray([w['z'] for w in ww])[idx]) and all(w['z']==(max(w['z_visual'],w['z_speech']) if 'z_speech' in w else w['z_visual']) for w in ww)


def metric(path):return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']}


def prepare(smoke):
    root=EXP/('smoke' if smoke else 'main');out=EXP/(root.name+'_analysis');out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/'config.json').read_text());assert cfg['spec']==SPEC and cfg['GT_read'] is False and cfg['smoke']==smoke and cfg['implementation']==IMPLEMENTATION
    rows=selected_rows(smoke);expected={(r['dataset'],r['video_id']) for r in rows}
    preds={name:records(root/name/'predictions.jsonl') for name in PREDICTIONS};assert all(set(p)==expected for p in preds.values())
    r6=records(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl')
    S={ds:dict(videos=0,windows=0,covered=0,accepted=0,accepted_up=0,accepted_down=0,up=0,down=0,grounded=0,says_none=0,parse_failures=0,cited_inside=0,numbers=0,
        replay_exact=0,replay_abs=[],generated=[],seconds=dict(native=0.,adjacent_probe=0.),peak_GiB=0.) for ds in DATASETS}
    bundles={}
    for ordinal,row in enumerate(rows,1):
        key=row['dataset'],row['video_id'];b=json.loads((root/'records'/key[0]/(key[1]+'.json')).read_text());s=S[key[0]];bundles[key]=b
        assert b['implementation']==IMPLEMENTATION and b['spec']==SPEC and b['checks']['GT_read'] is False
        windows=windows_for(row,[]);assert len(b['traces'])==len(windows)
        base=b['predictions']['base'];assert untimed(base)==untimed(preds['base'][key])
        assert base['extra']['z_video']==r6[key]['extra']['z_video'] and base['extra']['windows']==r6[key]['extra']['windows'] and np.array_equal(base['score_curve'],r6[key]['score_curve'])
        for name in PREDICTIONS:
            p=b['predictions'][name];assert untimed(p)==untimed(preds[name][key]) and curve_ok(p,float(row['duration']))
            assert p['extra']['z_video']==base['extra']['z_video'] and p['extra']['stance']==base['extra']['stance']
            assert [w.get('z_speech') for w in p['extra']['windows']]==[w.get('z_speech') for w in base['extra']['windows']]
        stored=json.loads((STORED/'records'/key[0]/(key[1]+'.json')).read_text());assert len(stored['traces'])==len(windows)
        for t,w,nw,st in zip(b['traces'],windows,base['extra']['windows'],stored['traces']):
            assert t['i']==w['i'] and t['bounds']==[w['start'],w['end']] and t['native_visual']==nw['z_visual'] and st['native_visual']==nw['z_visual']
            assert t['shown_times']==[x for x in b['native']['frame_times'] if w['start']<=x<w['end']]==st['native_inside']
            za=b['predictions']['accept_all']['extra']['windows'][w['i']]['z_visual'];zg=b['predictions']['grounded']['extra']['windows'][w['i']]['z_visual']
            if t['shown_times']:
                v=t['native_visual'];z=t['adjacent'];assert za==z and zg==t['final']
                g=t['grounded'];acc=(z>v and g) or (z<=v and not g);assert t['accepted']==acc and t['final']==(z if acc else v)
                assert g==any(abs(x-u)<=SPEC['match_tolerance'] for x in t['numbers'] for u in t['shown_times'])
                d=abs(z-st['adjacent_native']['z']);s['replay_abs'].append(d);s['replay_exact']+=d==0.
                s['covered']+=1;s['accepted']+=acc;s['grounded']+=g;s['says_none']+=t['says_none'];s['parse_failures']+=t['parse_failure']
                s['up']+=z>v;s['down']+=z<=v;s['accepted_up']+=acc and z>v;s['accepted_down']+=acc and z<=v
                s['numbers']+=bool(t['numbers']);s['cited_inside']+=bool(t['cited']);s['generated'].append(t['generated'])
            else:assert t['adjacent'] is None and t['accepted'] is None and za==zg==t['native_visual']==t['final'] and st['adjacent_native'] is None
            s['windows']+=1
        c=b['checks'];s['videos']+=1;s['peak_GiB']=max(s['peak_GiB'],c['peak_GiB']);s['seconds']['native']+=c['times']['prefix']+c['times']['native_visual']+c['times']['native_speech'];s['seconds']['adjacent_probe']+=c['times']['adjacent_probe']
        if ordinal%25==0:print('BOUND',ordinal,len(rows),flush=True)
    for ds,s in S.items():
        d=np.asarray(s.pop('replay_abs'));s['replay_max_abs']=float(d.max()) if len(d) else 0.;s['replay_mean_abs']=float(d.mean()) if len(d) else 0.
        g=np.asarray(s.pop('generated'));s['generated_mean']=float(g.mean()) if len(g) else 0.;s['generated_max']=int(g.max()) if len(g) else 0
        s['acceptance_rate']=s['accepted']/max(1,s['covered']);s['grounded_rate']=s['grounded']/max(1,s['covered']);s['none_rate']=s['says_none']/max(1,s['covered'])
    # Offline controls from the same reads: random acceptance at the corpus's own rate (seeds 0, 1) and the inverted rule.
    if not smoke:
        ctrl={name:[] for name in CONTROLS}
        for seed in (0,1):
            rng=np.random.default_rng(seed)
            for row in rows:
                key=row['dataset'],row['video_id'];b=bundles[key];rate=S[key[0]]['acceptance_rate'];vals=[]
                for t in b['traces']:
                    if t['adjacent'] is None:vals.append(t['native_visual'])
                    else:vals.append(t['adjacent'] if rng.random()<rate else t['native_visual'])
                ctrl[f'random_{seed}'].append(rebuild(b,row,vals,f'm1_c41_random_{seed}'))
        for row in rows:
            key=row['dataset'],row['video_id'];b=bundles[key]
            vals=[t['native_visual'] if t['adjacent'] is None else (t['native_visual'] if t['accepted'] else t['adjacent']) for t in b['traces']]
            ctrl['inverted'].append(rebuild(b,row,vals,'m1_c41_inverted'))
        for name,rr in ctrl.items():
            folder=root/name;folder.mkdir(exist_ok=True);(folder/'config.json').write_text(json.dumps(dict(control=name,built_from='main records',GT_read=False),indent=2)+'\n')
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(dict(PASS=True,GT_read=False,coverage=len(rows),native_allraw_exact=True,datasets=S),indent=2)+'\n')
    print(json.dumps({ds:{k:S[ds][k] for k in ('covered','acceptance_rate','grounded_rate','none_rate','parse_failures','replay_max_abs','generated_mean')} for ds in DATASETS},indent=1));print('PREPARE_PASS',flush=True)


def rebuild(b,row,vals,name):
    base=b['predictions']['base'];ctx=dict(global_margin=base['extra']['z_video'],stance=base['extra']['stance'],prefix_tokens=base['extra']['prefix_tokens'],
        stance_cache_tokens=base['extra']['stance_cache_tokens'],stance_cache_logical_start=base['extra']['stance_cache_logical_start'])
    speech=[w.get('z_speech') for w in base['extra']['windows']];return prediction(row,ctx,vals,speech,0.,name)


def evaluate(name):
    root=EXP/'main';decoded=EXP/'main_decoded';assert json.loads((EXP/'main_analysis/alignment.json').read_text())['PASS'] is True
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(root/name/'predictions.jsonl'),'--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(root/name/'metrics.json')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(root/name),'--datasets',*DATASETS,'--noleak','--transform','nscore','--key','calib','--duration','bma',
        '--bma-prior','length','--min-windows','2','--bma-grid','6','--arm','m2','--out-root',str(decoded),'--tag',name],cwd=ROOT,check=True)


def report():
    decoded=EXP/'main_decoded';binding=json.loads((EXP/'main_analysis/alignment.json').read_text())['datasets']
    m={'r6':metric(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json'),'adjacent_native':metric(ROOT/'runs/20261007_m1_streamingtom/placement_controls_main_decoded/adjacent_native/metrics.json')}
    for a in ARMS:m[a]=metric(decoded/a/'metrics.json')
    def diff(x,y):return {ds:{k:m[x][ds][k]-m[y][ds][k] for k in METRICS} for ds in DATASETS}
    steps=dict(replay=diff('accept_all','adjacent_native'),grounded_vs_r6=diff('grounded','r6'),grounded_vs_accept_all=diff('grounded','accept_all'),
        grounded_vs_random_0=diff('grounded','random_0'),grounded_vs_random_1=diff('grounded','random_1'),inverted_vs_r6=diff('inverted','r6'),inverted_vs_accept_all=diff('inverted','accept_all'))
    W='within_video_macro_ROC_AUC';within_noise=lambda d:all(abs(d[ds][k])<FLOOR[k] for ds in DATASETS for k in METRICS)
    perf=dict(no_loss=all(steps['grounded_vs_r6'][ds][k]>-FLOOR[k] for ds in DATASETS for k in METRICS),common=[k for k in METRICS if all(steps['grounded_vs_r6'][ds][k]>=.01 for ds in DATASETS)])
    perf['PASS']=bool(perf['no_loss'] and perf['common'])
    d=steps['grounded_vs_accept_all'];both=all(d[ds][W]>=.01 for ds in DATASETS);one=any(d[ds][W]>=.01 for ds in DATASETS);noloss=all(d[ds][k]>-FLOOR[k] for ds in DATASETS for k in METRICS)
    mech='supported' if both else 'partial' if (one and noloss) else 'not supported'
    beats_random=all(steps[f'grounded_vs_random_{s}'][ds][W]>FLOOR[W] for s in (0,1) for ds in DATASETS)
    result=dict(scope='development-selected; declared in README before the run; sole evaluator, fixed r6',metrics={a:{ds:{k:m[a][ds][k] for k in METRICS} for ds in DATASETS} for a in m},
        sources={'r6':'runs/20260926_twolevel/r6_bma/metrics.json','adjacent_native':'runs/20261007_m1_streamingtom/placement_controls_main_decoded/adjacent_native/metrics.json',**{a:str((decoded/a/'metrics.json').relative_to(ROOT)) for a in ARMS}},
        steps=steps,binding={ds:{k:binding[ds][k] for k in ('covered','acceptance_rate','accepted_up','accepted_down','up','down','grounded_rate','none_rate','parse_failures','cited_inside','replay_max_abs','replay_exact','generated_mean','seconds','peak_GiB')} for ds in DATASETS},
        verdicts=dict(replay_within_noise=within_noise(steps['replay']),performance=perf,mechanism_vs_accept_all=mech,beats_random_both_corpora=beats_random))
    out=EXP/'analysis';out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('C41_ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);ap.add_argument('--name');ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=EXP/('analysis' if a.stage=='report' else ('smoke' if a.smoke else 'main')+'_analysis');out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();done=False
    try:
        if a.stage=='prepare':prepare(a.smoke)
        elif a.stage=='evaluate':assert not a.smoke and a.name in ARMS;evaluate(a.name)
        else:report()
        done=True
    finally:
        n=1;tag=a.stage+('_'+a.name if a.name else '')
        while (out/f'{tag}_attempt_{n:04d}.json').exists():n+=1
        (out/f'{tag}_attempt_{n:04d}.json').write_text(json.dumps(dict(completed=done,elapsed_seconds=time.perf_counter()-start,stage=a.stage),indent=2)+'\n')


if __name__=='__main__':main()
