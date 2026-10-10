"""Error analysis (development; user question 2026-10-11): if the per-window reads with extra or re-shown frames
differ from the native read mostly by read-to-read noise, no label-free rule can pick the better read per window
(both the pre-read rules of selection_analysis.py and the citation probe of candidate 41 landed in the random
band). The label-free way to use several noisy reads of the same window is to combine them: average them, or
accept a change only when two independent reads agree on its direction. This simulates those rules with the reads
already stored (native v, adjacent_native a [covered windows], adjacent_local l, prefix_local p, local_clean E; all
from runs/20261007_m1_streamingtom/placement_controls_main/records), speech / G / stance unchanged, sole evaluator
and fixed r6 decoder. Size-matched random controls for every agreement rule; the visual-direction oracle for
reference. Diagnostics, not methods; no GT outside the evaluator and the oracle."""
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/20261007_m1_streamingtom/placement_controls_main'
OUT=ROOT/'runs/20261007_m1_streamingtom/combination_analysis'
DATASETS=('HateMM','HateClipSeg')
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')
PY=sys.executable


def metric(path):return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']}


def hybrid(base,vals,name):
    ww=[]
    for b,v in zip(base['extra']['windows'],vals):ww.append({**b,'z_visual':float(v),'z':float(max(v,b['z_speech'])) if 'z_speech' in b else float(v)})
    idx=np.clip(((np.arange(math.ceil(base['duration']*4))+.5)/4//8).astype(int),0,len(ww)-1)
    return {**base,'method':name,'score_curve':np.asarray([w['z'] for w in ww])[idx].tolist(),'extra':{**base['extra'],'windows':ww}}


def load():
    gt={}
    for ds in DATASETS:
        z=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True);gt[ds]={v:y for v,y in zip(z['video_ids'],z['y4'])}
    videos=[]
    for ds in DATASETS:
        for f in sorted((RUN/'records'/ds).glob('*.json')):
            b=json.loads(f.read_text());base=b['base'];y=gt[ds][base['video_id']];win=[]
            for t,w in zip(b['traces'],base['extra']['windows']):
                assert t['i']==w['i'] and t['native_visual']==w['z_visual']
                s0,s1=int(round(w['start']*4)),int(round(w['end']*4));label=int(y[s0:s1].max()) if s1>s0 and s0<len(y) else 0
                win.append(dict(v=w['z_visual'],a=t['adjacent_native']['z'] if t['adjacent_native'] else None,l=t['adjacent_local']['z'] if t['adjacent_local'] else None,p=t['prefix_local']['z'] if t['prefix_local'] else None,E=t['stored_local_clean'],label=label))
            videos.append(dict(ds=ds,base=base,win=win))
    return videos


def sgn(x):return 0 if x==0 else (1 if x>0 else -1)


def rules(vid):
    W=vid['win'];n=len(W);R={}
    def per(f):return [f(w) for w in W]
    def av(*xs):xs=[x for x in xs if x is not None];return float(np.mean(xs)) if xs else None
    R['full_a']=per(lambda w:w['a'] if w['a'] is not None else w['v'])
    R['full_l']=per(lambda w:av(w['l']) if w['l'] is not None else w['v']);R['full_p']=per(lambda w:w['p'] if w['p'] is not None else w['v']);R['full_E']=per(lambda w:w['E'])
    R['avg_va']=per(lambda w:av(w['v'],w['a']));R['avg_vl']=per(lambda w:av(w['v'],w['l']));R['avg_val']=per(lambda w:av(w['v'],w['a'],w['l']))
    R['avg_vlp']=per(lambda w:av(w['v'],w['l'],w['p']));R['avg_all']=per(lambda w:av(w['v'],w['a'],w['l'],w['p'],w['E']));R['avg_extra']=per(lambda w:av(w['a'],w['l'],w['p'],w['E']))
    R['median_all']=per(lambda w:float(np.median([x for x in (w['v'],w['a'],w['l'],w['p'],w['E']) if x is not None])))
    # agreement rules: accept a change only when a second (third) independent read moved the same way from v
    acc={}
    def agree(name,cands,take,value):
        out=[];flags=[]
        for w in W:
            xs=[w[c] for c in cands]
            if any(x is None for x in xs):out.append(w['v']);continue
            ok=len({sgn(x-w['v']) for x in xs})==1 and sgn(xs[0]-w['v'])!=0;flags.append(ok);out.append(value(w) if ok else w['v'])
        R[name]=out;acc[name]=flags
    agree('agree_al',('a','l'),None,lambda w:w['a']);agree('agree_al_mean',('a','l'),None,lambda w:av(w['a'],w['l']))
    agree('agree_alp',('a','l','p'),None,lambda w:w['a']);agree('agree_alp_mean',('a','l','p'),None,lambda w:av(w['a'],w['l'],w['p']))
    agree('agree_lp',('l','p'),None,lambda w:w['l']);agree('agree_lp_mean',('l','p'),None,lambda w:av(w['l'],w['p']))
    agree('agree_lpE_mean',('l','p','E'),None,lambda w:av(w['l'],w['p'],w['E']))
    # neighbour rule: accept a_i when its direction matches the net direction of the adjacent reads next to it
    out=[]
    for i,w in enumerate(W):
        if w['a'] is None:out.append(w['v']);continue
        nb=[W[j]['a']-W[j]['v'] for j in (i-1,i+1) if 0<=j<n and W[j]['a'] is not None]
        out.append(w['a'] if (not nb or sgn(sum(nb))==sgn(w['a']-w['v'])) else w['v'])
    R['neighbour_a']=out
    R['oracle_a']=per(lambda w:w['a'] if w['a'] is not None and w['a']!=w['v'] and ((w['a']>w['v'])==bool(w['label'])) else w['v'])
    R['oracle_l']=per(lambda w:w['l'] if w['l'] is not None and w['l']!=w['v'] and ((w['l']>w['v'])==bool(w['label'])) else w['v'])
    for name in R:R[name]=[w['v'] if (x is None or not np.isfinite(x)) else x for x,w in zip(R[name],W)]  # a missing read keeps the native value
    return R,acc


def main():
    OUT.mkdir(parents=True,exist_ok=True);videos=load()
    # pass 1: rule values and acceptance rates per corpus
    per_video=[rules(v) for v in videos];rate={}
    for ds in DATASETS:
        for name in ('agree_al','agree_alp','agree_lp'):
            fl=[f for vid,(R,acc) in zip(videos,per_video) if vid['ds']==ds for f in acc[name]];rate[(ds,name)]=sum(fl)/max(1,len(fl))
    preds={}
    for vi,(vid,(R,acc)) in enumerate(zip(videos,per_video)):
        for name,vals in R.items():preds.setdefault(name,[]).append(hybrid(vid['base'],vals,f'm1_comb_{name}'))
        W=vid['win']
        for name,cand in (('agree_al','a'),('agree_alp','a'),('agree_lp','l')):
            for seed in (0,1):
                rng=np.random.default_rng([seed,vi])
                vals=[(w[cand] if (w[cand] is not None and rng.random()<rate[(vid['ds'],name)]) else w['v']) for w in W]
                preds.setdefault(f'random_{name}_{seed}',[]).append(hybrid(vid['base'],vals,f'm1_comb_random_{name}_{seed}'))
    for name,rr in preds.items():
        folder=OUT/name;folder.mkdir(exist_ok=True);(folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
        (folder/'config.json').write_text(json.dumps(dict(rule=name,built_from='placement_controls_main records',labels_used=name.startswith('oracle')),indent=2)+'\n')
    def run(name):
        folder=OUT/name
        subprocess.run([PY,'-m','src.eval.evaluate_four_datasets','--predictions',str(folder/'predictions.jsonl'),'--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(folder/'metrics.json')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        subprocess.run([PY,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(folder),'--datasets',*DATASETS,'--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length',
            '--min-windows','2','--bma-grid','6','--arm','m2','--out-root',str(OUT/'decoded'),'--tag',name],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        return name
    with ThreadPoolExecutor(6) as ex:
        for name in ex.map(run,list(preds)):print('EVAL',name,flush=True)
    r6=metric(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json');M={name:metric(OUT/'decoded'/name/'metrics.json') for name in preds}
    res={name:dict(metrics={ds:{k:M[name][ds][k] for k in METRICS} for ds in DATASETS},gain_vs_r6={ds:{k:M[name][ds][k]-r6[ds][k] for k in METRICS} for ds in DATASETS}) for name in preds}
    for name in preds:res[name]['gain_vs_full_a']={ds:{k:M[name][ds][k]-M['full_a'][ds][k] for k in METRICS} for ds in DATASETS}
    summary=dict(scope='development-selected error analysis; labels only inside the evaluator and in oracle_*',acceptance_rates={f'{ds}:{n}':r for (ds,n),r in rate.items()},rules=res,
        files_read=['runs/20261007_m1_streamingtom/placement_controls_main/records','data/gt_4fps/{HateMM,HateClipSeg}.npz','runs/20260926_twolevel/r6_bma/metrics.json'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('acceptance rates',{f'{ds}:{n}':round(r,3) for (ds,n),r in rate.items()})
    print(f"{'rule':24s} HateMM ROC/PR/within vs r6      | HateClipSeg vs r6")
    for name in preds:print(f"{name:24s}",' | '.join(' / '.join(f"{res[name]['gain_vs_r6'][ds][k]:+.4f}" for k in METRICS) for ds in DATASETS))
    print('COMBINATION_DONE',flush=True)


if __name__=='__main__':main()
