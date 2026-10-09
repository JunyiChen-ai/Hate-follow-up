"""Candidate 41 test error analysis (rule 10, development; 2026-10-10).

Question: does the grounding probe's answer carry any information about whether the adjacent read moved a window
toward its label? Per covered window: direction of the adjacent read (up: z > v, down: z <= v), probe class
(none / cited a shown frame / other), window label (positive when any 4 fps GT frame in [start, end) is positive),
and whether the fused margin max(visual, speech) moved toward the label. Then label-free acceptance rules built
from the same reads go through the sole evaluator and the fixed r6 decoder, with size-matched random controls and
the oracle (accept exactly the helpful moves) as the ceiling. Diagnostics, not methods. Files read: main records,
data/gt_4fps (labels only here and in the evaluator).
"""
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'runs/20261010_m1_grounded_adjacent'
OUT=EXP/'error_analysis'
DATASETS=('HateMM','HateClipSeg')
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')
PY=sys.executable


def metric(path):return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']}


def hybrid(base,vals,name):
    ww=[]
    for b,v in zip(base['extra']['windows'],vals):ww.append({**b,'z_visual':v,'z':max(v,b['z_speech']) if 'z_speech' in b else v})
    idx=np.clip(((np.arange(math.ceil(base['duration']*4))+.5)/4//8).astype(int),0,len(ww)-1)
    return {**base,'method':name,'score_curve':np.asarray([w['z'] for w in ww])[idx].tolist(),'extra':{**base['extra'],'windows':ww}}


def load():
    gt={}
    for ds in DATASETS:
        z=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True);gt[ds]={v:y for v,y in zip(z['video_ids'],z['y4'])}
    videos=[]
    for ds in DATASETS:
        for f in sorted((EXP/'main/records'/ds).glob('*.json')):
            b=json.loads(f.read_text());base=b['predictions']['base'];y=gt[ds][base['video_id']];win=[]
            for t,w in zip(b['traces'],base['extra']['windows']):
                assert t['i']==w['i'] and t['native_visual']==w['z_visual']
                a,bb=int(round(w['start']*4)),int(round(w['end']*4));label=int(y[a:bb].max()) if bb>a and a<len(y) else 0
                v=w['z_visual'];s=w.get('z_speech');z=t['adjacent']
                d=dict(v=v,s=s,z=z,label=label,covered=z is not None,accepted=t.get('accepted'),grounded=t.get('grounded'),none=t.get('says_none'),fail=t.get('parse_failure'),numbers=t.get('numbers',[]))
                if z is not None:
                    fused=lambda x:max(x,s) if s is not None else x
                    d['up']=z>v;d['dz']=z-v;d['dfused']=fused(z)-fused(v)
                    d['probe']='fail' if t['parse_failure'] else 'none' if t['says_none'] else 'shown' if t['grounded'] else 'other'
                    d['help']=None if d['dfused']==0 else (d['dfused']>0)==bool(label)
                win.append(d)
            videos.append(dict(ds=ds,base=base,win=win))
    return videos


def table(videos):
    T={}
    for ds in DATASETS:
        rows={}
        for vid in videos:
            if vid['ds']!=ds:continue
            for w in vid['win']:
                if not w['covered']:continue
                k=('up' if w['up'] else 'down',w['probe'],'pos' if w['label'] else 'neg');r=rows.setdefault(k,dict(n=0,fused_changed=0,helpful=0,harmful=0,abs_dz=[]))
                r['n']+=1;r['abs_dz'].append(abs(w['dz']))
                if w['help'] is not None:r['fused_changed']+=1;r['helpful']+=w['help'];r['harmful']+=not w['help']
        out={}
        for k,r in sorted(rows.items()):
            out[' '.join(k)]=dict(n=r['n'],fused_changed=r['fused_changed'],helpful=r['helpful'],harmful=r['harmful'],help_rate=(r['helpful']/r['fused_changed'] if r['fused_changed'] else None),mean_abs_dz=float(np.mean(r['abs_dz'])))
        # label-blind view: does the probe class predict helpfulness within each direction?
        agg={}
        for k,r in rows.items():
            a=agg.setdefault((k[0],k[1]),dict(n=0,fused_changed=0,helpful=0,positives=0))
            a['n']+=r['n'];a['fused_changed']+=r['fused_changed'];a['helpful']+=r['helpful'];a['positives']+=r['n'] if k[2]=='pos' else 0
        T[ds]=dict(cells=out,by_direction_probe={' '.join(k):dict(n=a['n'],positive_share=a['positives']/a['n'],help_rate=(a['helpful']/a['fused_changed'] if a['fused_changed'] else None)) for k,a in sorted(agg.items())})
    return T


def rules(vid,rng_pool):
    W=vid['win'];n=len(W);out={}
    cov=[w['covered'] for w in W]
    def acc(f):return [bool(c and f(w)) for c,w in zip(cov,W)]
    out['accept_all']=acc(lambda w:True)
    out['grounded']=acc(lambda w:w['accepted'])
    out['down_only']=acc(lambda w:not w['up'])
    out['up_only']=acc(lambda w:w['up'])
    out['down_none_only']=acc(lambda w:(not w['up']) and w['none'])
    out['up_shown_only']=acc(lambda w:w['up'] and w['grounded'])
    out['down_all_up_shown']=acc(lambda w:(not w['up']) or w['grounded'])
    down_rate=sum(out['down_only'])/max(1,sum(cov))
    for seed in (0,1):
        rng=rng_pool[seed];out[f'random_down_rate_{seed}']=[bool(c and rng.random()<down_rate) for c in cov]
    out['oracle_help']=acc(lambda w:w['help'] is True)
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True);videos=load();T=table(videos)
    rng_pool={0:np.random.default_rng(0),1:np.random.default_rng(1)}
    preds={};rates={ds:{} for ds in DATASETS}
    for vid in videos:
        R=rules(vid,rng_pool)
        for name,take in R.items():
            vals=[w['z'] if t else w['v'] for w,t in zip(vid['win'],take)]
            preds.setdefault(name,[]).append(hybrid(vid['base'],vals,f'm1_c41_ea_{name}'))
            r=rates[vid['ds']].setdefault(name,dict(accepted=0,covered=0));r['accepted']+=sum(take);r['covered']+=sum(w['covered'] for w in vid['win'])
    for name,rr in preds.items():
        folder=OUT/name;folder.mkdir(exist_ok=True);(folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
        (folder/'config.json').write_text(json.dumps(dict(rule=name,built_from='main records',labels_used=name=='oracle_help'),indent=2)+'\n')
    def run(name):
        folder=OUT/name
        subprocess.run([PY,'-m','src.eval.evaluate_four_datasets','--predictions',str(folder/'predictions.jsonl'),'--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(folder/'metrics.json')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        subprocess.run([PY,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(folder),'--datasets',*DATASETS,'--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length',
            '--min-windows','2','--bma-grid','6','--arm','m2','--out-root',str(OUT/'decoded'),'--tag',name],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
        return name
    with ThreadPoolExecutor(6) as ex:
        for name in ex.map(run,list(preds)):print('EVAL',name,flush=True)
    r6=metric(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json');M={name:metric(OUT/'decoded'/name/'metrics.json') for name in preds}
    res={}
    for name in preds:
        res[name]=dict(acceptance_rate={ds:rates[ds][name]['accepted']/max(1,rates[ds][name]['covered']) for ds in DATASETS},
            metrics={ds:{k:M[name][ds][k] for k in METRICS} for ds in DATASETS},
            gain_vs_r6={ds:{k:M[name][ds][k]-r6[ds][k] for k in METRICS} for ds in DATASETS},
            gain_vs_accept_all={ds:{k:M[name][ds][k]-M['accept_all'][ds][k] for k in METRICS} for ds in DATASETS})
    summary=dict(scope='development-selected test error analysis (rule 10); labels used only in the table, in oracle_help and inside the evaluator',table=T,rules=res,
        files_read=['runs/20261010_m1_grounded_adjacent/main/records','data/gt_4fps/{HateMM,HateClipSeg}.npz','runs/20260926_twolevel/r6_bma/metrics.json'])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for ds in DATASETS:
        print(ds);[print(f"  {k:28s} n={v['n']:5d} pos={v['positive_share']:.2f} help={v['help_rate'] if v['help_rate'] is None else round(v['help_rate'],3)}") for k,v in T[ds]['by_direction_probe'].items()]
    for name in preds:
        print(f"{name:22s}",' | '.join(f"{ds[:3]} acc={res[name]['acceptance_rate'][ds]:.2f} "+' / '.join(f"{res[name]['gain_vs_r6'][ds][k]:+.4f}" for k in METRICS) for ds in DATASETS))
    print('C41_ERROR_ANALYSIS_DONE',flush=True)


if __name__=='__main__':main()
