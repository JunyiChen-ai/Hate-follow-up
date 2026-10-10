"""Candidate 41 revision 1: binding checks of the probe job, selections (rules A, B x arms a, l, E) applied offline
from the stored placement reads, six rate-matched random controls per selection, full arms and visual oracles for
reference, sole evaluator + fixed r6 decoder, declared report (README section "Revision 1")."""
import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
EXP=ROOT/'runs/20261010_m1_grounded_adjacent'
STORED=ROOT/'runs/20261007_m1_streamingtom/placement_controls_main'
DATASETS=('HateMM','HateClipSeg')
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')
FLOOR={'frame_ROC_AUC':.005,'frame_PR_AUC':.005,'within_video_macro_ROC_AUC':.01}
RULES=('A','B');ARMS=('a','l','E');SEEDS=range(6);W='within_video_macro_ROC_AUC'
PRIMARY='sel_A_a'


def records(path):
    rows={}
    for line in path.open():
        r=json.loads(line);key=r['dataset'],r['video_id'];assert key not in rows;rows[key]=r
    return rows


def untimed(p):return {**p,'extra':{k:v for k,v in p['extra'].items() if k!='standalone_seconds'}}


def metric(path):return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']}


def hybrid(base,vals,name):
    ww=[]
    for b,v in zip(base['extra']['windows'],vals):ww.append({**b,'z_visual':float(v),'z':float(max(v,b['z_speech'])) if 'z_speech' in b else float(v)})
    idx=np.clip(((np.arange(math.ceil(base['duration']*4))+.5)/4//8).astype(int),0,len(ww)-1)
    return {**base,'method':name,'score_curve':np.asarray([w['z'] for w in ww])[idx].tolist(),'extra':{**base['extra'],'windows':ww}}


def load(smoke):
    from probe_native import SPEC,IMPLEMENTATION,selected_rows,windows_for
    root=EXP/('probe_smoke' if smoke else 'probe_main');cfg=json.loads((root/'config.json').read_text())
    assert cfg['spec']==SPEC and cfg['GT_read'] is False and cfg['smoke']==smoke and cfg['implementation']==IMPLEMENTATION
    rows=selected_rows(smoke);preds=records(root/'base/predictions.jsonl');assert set(preds)=={(r['dataset'],r['video_id']) for r in rows}
    r6=records(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');videos=[];S={ds:dict(videos=0,windows=0,covered=0,correct_a=0,flag_a=0,flag_b=0,none_a=0,none_b=0,inside_b=0,parse_a=0,parse_b=0,
        correct_a_covered=0,correct_a_uncovered=0,bounds_echo=0,generated_a=[],generated_b=[],seconds=dict(native=0.,probes=0.),peak_GiB=0.) for ds in DATASETS}
    for row in rows:
        key=row['dataset'],row['video_id'];b=json.loads((root/'records'/key[0]/(key[1]+'.json')).read_text());s=S[key[0]]
        assert b['implementation']==IMPLEMENTATION and b['spec']==SPEC and b['checks']['GT_read'] is False
        windows=windows_for(row,[]);assert len(b['traces'])==len(windows);base=b['predictions']['base'];assert untimed(base)==untimed(preds[key])
        assert base['extra']['z_video']==r6[key]['extra']['z_video'] and base['extra']['windows']==r6[key]['extra']['windows'] and np.array_equal(base['score_curve'],r6[key]['score_curve'])
        st=json.loads((STORED/'records'/key[0]/(key[1]+'.json')).read_text());assert len(st['traces'])==len(windows) and st['base']['extra']['windows']==base['extra']['windows']
        win=[]
        for t,w,nw,stt in zip(b['traces'],windows,base['extra']['windows'],st['traces']):
            assert t['i']==w['i']==stt['i'] and t['bounds']==[w['start'],w['end']]==stt['bounds'] and t['native_visual']==nw['z_visual']==stt['native_visual']
            assert t['inside_times']==[x for x in b['native']['frame_times'] if w['start']<=x<w['end']]==stt['native_inside']
            truth=sorted(t['inside_times']);correct=(sorted(t['matched_a'])==truth) if truth else (t['none_a'] and not t['matched_a']);assert t['correct_a']==correct and t['flag_a']==(not correct)
            assert t['inside_b']==any(w['start']<=x<w['end'] for x in t['matched_b']) and t['flag_b']==(not t['inside_b'])
            cov=bool(truth);s['windows']+=1;s['covered']+=cov;s['correct_a']+=correct;s['flag_a']+=t['flag_a'];s['flag_b']+=t['flag_b'];s['none_a']+=t['none_a'];s['none_b']+=t['none_b'];s['inside_b']+=t['inside_b']
            s['parse_a']+=t['parse_failure_a'];s['parse_b']+=t['parse_failure_b'];s['correct_a_covered' if cov else 'correct_a_uncovered']+=correct
            s['bounds_echo']+=any(abs(x-w['start'])<=.05+1e-9 or abs(x-w['end'])<=.05+1e-9 for x in t['numbers_a']+t['numbers_b'])
            s['generated_a'].append(t['generated_a']);s['generated_b'].append(t['generated_b'])
            win.append(dict(v=nw['z_visual'],a=stt['adjacent_native']['z'] if stt['adjacent_native'] else None,l=stt['adjacent_local']['z'] if stt['adjacent_local'] else None,E=stt['stored_local_clean'],
                flag_A=bool(t['flag_a']),flag_B=bool(t['flag_b']),covered=cov))
        c=b['checks'];s['videos']+=1;s['peak_GiB']=max(s['peak_GiB'],c['peak_GiB']);s['seconds']['native']+=c['times']['prefix']+c['times']['native_visual']+c['times']['native_speech'];s['seconds']['probes']+=c['times']['probes']
        videos.append(dict(ds=key[0],base=base,win=win))
    for ds,s in S.items():
        for k in ('generated_a','generated_b'):g=np.asarray(s.pop(k));s[k+'_mean']=float(g.mean()) if len(g) else 0.;s[k+'_max']=int(g.max()) if len(g) else 0
        n=max(1,s['windows']);s['correct_a_rate']=s['correct_a']/n;s['flag_a_rate']=s['flag_a']/n;s['flag_b_rate']=s['flag_b']/n;s['none_a_rate']=s['none_a']/n;s['inside_b_rate']=s['inside_b']/n
        s['correct_a_rate_covered']=s['correct_a_covered']/max(1,s['covered']);s['correct_a_rate_uncovered']=s['correct_a_uncovered']/max(1,s['windows']-s['covered'])
    return videos,S


def selections(videos,gt):
    """Returns {name: [prediction per video]} and the per-corpus flag rates over each arm's eligible windows."""
    preds={};rate={}
    for rule in RULES:
        for arm in ARMS:
            for ds in DATASETS:
                el=[w for vid in videos if vid['ds']==ds for w in vid['win'] if w[arm] is not None];rate[(ds,rule,arm)]=sum(w['flag_'+rule] for w in el)/max(1,len(el))
    for vi,vid in enumerate(videos):
        W_=vid['win'];y=gt[vid['ds']][vid['base']['video_id']]
        def put(name,vals):preds.setdefault(name,[]).append(hybrid(vid['base'],vals,'m1_c41r1_'+name))
        for arm in ARMS:
            put(f'full_{arm}',[w[arm] if w[arm] is not None else w['v'] for w in W_])
            lab=[]
            for b in vid['base']['extra']['windows']:
                s0,s1=int(round(b['start']*4)),int(round(b['end']*4));lab.append(int(y[s0:s1].max()) if s1>s0 and s0<len(y) else 0)
            put(f'oracle_{arm}',[w[arm] if (w[arm] is not None and w[arm]!=w['v'] and ((w[arm]>w['v'])==bool(l))) else w['v'] for w,l in zip(W_,lab)])
            for rule in RULES:
                put(f'sel_{rule}_{arm}',[w[arm] if (w[arm] is not None and w['flag_'+rule]) else w['v'] for w in W_])
                for seed in SEEDS:
                    rng=np.random.default_rng([seed,vi]);r=rate[(vid['ds'],rule,arm)]
                    put(f'random_{rule}_{arm}_{seed}',[w[arm] if (w[arm] is not None and rng.random()<r) else w['v'] for w in W_])
    return preds,rate


def evaluate(name,folder,decoded):
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(folder/'predictions.jsonl'),'--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(folder/'metrics.json')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(folder),'--datasets',*DATASETS,'--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length',
        '--min-windows','2','--bma-grid','6','--arm','m2','--out-root',str(decoded),'--tag',name],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    return name


def prepare(smoke):
    videos,S=load(smoke);out=EXP/('probe_smoke_analysis' if smoke else 'probe_analysis');out.mkdir(parents=True,exist_ok=True)
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(dict(PASS=True,GT_read=False,coverage=sum(s['videos'] for s in S.values()),native_exact=True,datasets=S),indent=2)+'\n')
    print(json.dumps({ds:{k:S[ds][k] for k in ('windows','covered','correct_a_rate','correct_a_rate_covered','correct_a_rate_uncovered','flag_a_rate','flag_b_rate','none_a_rate','inside_b_rate','parse_a','parse_b','bounds_echo','generated_a_mean','generated_b_mean','seconds','peak_GiB')} for ds in DATASETS},indent=1))
    print('PROBE_PREPARE_PASS',flush=True)
    if smoke:return
    gt={}
    for ds in DATASETS:
        z=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True);gt[ds]={v:y for v,y in zip(z['video_ids'],z['y4'])}
    preds,rate=selections(videos,gt);sel=EXP/'probe_selection';sel.mkdir(exist_ok=True)
    for name,rr in preds.items():
        folder=sel/name;folder.mkdir(exist_ok=True);(folder/'config.json').write_text(json.dumps(dict(selection=name,built_from='probe_main records + placement_controls_main records',labels_used=name.startswith('oracle')),indent=2)+'\n')
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
    (out/'selection_rates.json').write_text(json.dumps({f'{ds}:{rule}:{arm}':r for (ds,rule,arm),r in rate.items()},indent=2)+'\n');print('SELECTIONS',len(preds),flush=True)


def run_evaluate():
    sel=EXP/'probe_selection';decoded=EXP/'probe_selection_decoded';assert json.loads((EXP/'probe_analysis/alignment.json').read_text())['PASS'] is True
    names=sorted(p.name for p in sel.iterdir() if (p/'predictions.jsonl').exists())
    with ThreadPoolExecutor(8) as ex:
        for name in ex.map(lambda n:evaluate(n,sel/n,decoded),names):print('EVALUATED',name,flush=True)


def report():
    out=EXP/'probe_analysis';decoded=EXP/'probe_selection_decoded';binding=json.loads((out/'alignment.json').read_text())['datasets'];rate=json.loads((out/'selection_rates.json').read_text())
    r6=metric(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json');names=sorted(p.name for p in decoded.iterdir() if (p/'metrics.json').exists());m={n:metric(decoded/n/'metrics.json') for n in names}
    gain={n:{ds:{k:m[n][ds][k]-r6[ds][k] for k in METRICS} for ds in DATASETS} for n in names};verdicts={}
    for rule in RULES:
        for arm in ARMS:
            n=f'sel_{rule}_{arm}';rnd=[f'random_{rule}_{arm}_{s}' for s in SEEDS];full=f'full_{arm}'
            above_random={ds:min(gain[n][ds][W]-gain[r][ds][W] for r in rnd) for ds in DATASETS}
            verdicts[n]=dict(flag_rate={ds:rate[f'{ds}:{rule}:{arm}'] for ds in DATASETS},within_gain_vs_r6={ds:gain[n][ds][W] for ds in DATASETS},within_vs_full={ds:gain[n][ds][W]-gain[full][ds][W] for ds in DATASETS},
                within_above_every_random={ds:above_random[ds] for ds in DATASETS},random_band={ds:[gain[r][ds][W] for r in rnd] for ds in DATASETS},
                pooled_no_loss=all(gain[n][ds][k]>-FLOOR[k] for ds in DATASETS for k in METRICS[:2]),
                supported=bool(all(above_random[ds]>FLOOR[W] for ds in DATASETS) and all(gain[n][ds][k]>-FLOOR[k] for ds in DATASETS for k in METRICS[:2])))
    result=dict(scope='development-selected; declared in README (Revision 1) before the run; sole evaluator, fixed r6; six selections tested, primary '+PRIMARY,
        r6={ds:{k:r6[ds][k] for k in METRICS} for ds in DATASETS},metrics={n:{ds:{k:m[n][ds][k] for k in METRICS} for ds in DATASETS} for n in names},gain_vs_r6=gain,
        binding={ds:{k:binding[ds][k] for k in ('windows','covered','correct_a_rate','correct_a_rate_covered','correct_a_rate_uncovered','flag_a_rate','flag_b_rate','none_a_rate','inside_b_rate','parse_a','parse_b','bounds_echo','generated_a_mean','generated_b_mean','seconds','peak_GiB')} for ds in DATASETS},
        verdicts=verdicts,primary=dict(name=PRIMARY,supported=verdicts[PRIMARY]['supported']),any_supported=[n for n,v in verdicts.items() if v['supported']])
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f"{'selection':12s} flag HMM/HCS   within vs r6        vs full        above every random   supported")
    for n,v in verdicts.items():print(f"{n:12s} {v['flag_rate']['HateMM']:.2f}/{v['flag_rate']['HateClipSeg']:.2f}  "+' / '.join(f"{v['within_gain_vs_r6'][ds]:+.4f}" for ds in DATASETS)+'  '+' / '.join(f"{v['within_vs_full'][ds]:+.4f}" for ds in DATASETS)+'  '+' / '.join(f"{v['within_above_every_random'][ds]:+.4f}" for ds in DATASETS)+f"  {v['supported']}")
    for arm in ARMS:print(f"full_{arm} / oracle_{arm} within vs r6: "+' | '.join(f"{gain[f'full_{arm}'][ds][W]:+.4f} / {gain[f'oracle_{arm}'][ds][W]:+.4f}" for ds in DATASETS))
    print('C41R1_ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parent));out=EXP/('probe_smoke_analysis' if a.smoke else 'probe_analysis');out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();done=False
    try:
        if a.stage=='prepare':prepare(a.smoke)
        elif a.stage=='evaluate':assert not a.smoke;run_evaluate()
        else:report()
        done=True
    finally:
        n=1
        while (out/f'{a.stage}_attempt_{n:04d}.json').exists():n+=1
        (out/f'{a.stage}_attempt_{n:04d}.json').write_text(json.dumps(dict(completed=done,elapsed_seconds=time.perf_counter()-start,stage=a.stage),indent=2)+'\n')


if __name__=='__main__':main()
