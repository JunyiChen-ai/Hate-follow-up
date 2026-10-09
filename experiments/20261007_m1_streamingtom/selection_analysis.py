"""Error analysis (development; user question 2026-10-10): can a label-free rule pick the windows where the LOCAL /
adjacent frames help, so that only part of the windows needs the extra read?

For each arm (adjacent_native, E = stored local_clean, prefix_local) and each selection rule, the diagnostic
prediction takes the arm's visual read in the selected windows and r6's native read elsewhere (speech, G, stance
unchanged), then goes through the sole evaluator and the fixed r6 decoder. Rules use only quantities available
before the extra read (native visual margin v, native speech margin s, number of prefix frames inside the window).
`random_q50_*` are size-matched random controls. `oracle_help` selects with the test labels (a window counts as
positive when any 4 fps GT frame in it is positive; selected when the arm moved z toward that label): a ceiling for
selection, never a method. Diagnostics, not methods.
"""
import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

EXP=ROOT/'runs/20261007_m1_streamingtom'
RUN=EXP/'placement_controls_main'
OUT=EXP/'selection_analysis'
DATASETS=('HateMM','HateClipSeg')
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')
ARMS=('adjacent_native','local_clean','prefix_local')


def metric(path):
    return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']}


def hybrid(base,new_z,take,name):
    ww=[]
    for b,z,t in zip(base['extra']['windows'],new_z,take):
        v=z if t else b['z_visual'];ww.append({**b,'z_visual':v,'z':max(v,b['z_speech']) if 'z_speech' in b else v})
    idx=np.clip(((np.arange(math.ceil(base['duration']*4))+.5)/4//8).astype(int),0,len(ww)-1)
    return {**base,'method':name,'score_curve':np.asarray([w['z'] for w in ww])[idx].tolist(),'extra':{**base['extra'],'windows':ww}}


def load():
    gt={}
    for ds in DATASETS:
        z=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True);gt[ds]={v:y for v,y in zip(z['video_ids'],z['y4'])}
    videos=[]
    for ds in DATASETS:
        for f in sorted((RUN/'records'/ds).glob('*.json')):
            b=json.loads(f.read_text());base=b['base'];y=gt[ds][b['base']['video_id']]
            win=[]
            for t,w in zip(b['traces'],base['extra']['windows']):
                assert t['i']==w['i'] and t['native_visual']==w['z_visual']
                a,bb=int(round(w['start']*4)),int(round(w['end']*4));label=int(y[a:bb].max()) if bb>a and a<len(y) else 0
                win.append(dict(v=w['z_visual'],s=w.get('z_speech'),inside=len(t['native_inside']),local=len(t['LOCAL']),label=label,
                    arms={'adjacent_native':t['adjacent_native']['z'] if t['adjacent_native'] else w['z_visual'],
                          'local_clean':t['stored_local_clean'] if t.get('stored_local_clean') is not None else w['z_visual'],
                          'prefix_local':t['prefix_local']['z'] if t['prefix_local'] else w['z_visual']}))
            videos.append(dict(ds=ds,base=base,win=win,G=b['native']['global_margin']))
    return videos


def rules(video,arm,rng):
    W=video['win'];v=np.array([w['v'] for w in W]);absv=np.abs(v);n=len(W)
    half=max(1,n//2);order_unc=np.argsort(absv,kind='stable')[:half];order_top=np.argsort(-v,kind='stable')[:half]
    out={'full':[True]*n}
    for t in (1,2,3):out[f'unc_abs_{t}']=(absv<=t).tolist()
    out['unc_q50']=[i in set(order_unc.tolist()) for i in range(n)];out['top_q50']=[i in set(order_top.tolist()) for i in range(n)]
    out['disagree']=[(w['s'] is not None) and ((w['v']>0)!=(w['s']>0)) for w in W]
    if arm!='adjacent_native':
        out['empty']=[w['inside']==0 for w in W];out['covered']=[w['inside']>=1 for w in W]
    for seed in (0,1):
        r=np.random.default_rng(seed);pick=set(r.choice(n,half,replace=False).tolist());out[f'random_q50_{seed}']=[i in pick for i in range(n)]
    out['oracle_help']=[((w['arms'][arm]-w['v'])>0)==bool(w['label']) and w['arms'][arm]!=w['v'] for w in W]
    return out


def evaluate(folder,name):
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(folder/'predictions.jsonl'),
        '--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(folder/'metrics.json')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(folder),'--datasets',*DATASETS,
        '--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length','--min-windows','2','--bma-grid','6','--arm','m2',
        '--out-root',str(OUT/'decoded'),'--tag',name],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    return name


def main():
    OUT.mkdir(parents=True,exist_ok=True);videos=load();rng=np.random.default_rng(0)
    combos={};frac={}
    for arm in ARMS:
        per_video=[rules(vd,arm,rng) for vd in videos];names=per_video[0].keys()
        for rule in names:
            name=f'{arm}__{rule}';rows=[];sel={ds:[0,0] for ds in DATASETS}
            for vd,rr in zip(videos,per_video):
                take=rr[rule];rows.append(hybrid(vd['base'],[w['arms'][arm] for w in vd['win']],take,'diag_'+name))
                changed=[t and w['arms'][arm]!=w['v'] for t,w in zip(take,vd['win'])];sel[vd['ds']][0]+=sum(changed);sel[vd['ds']][1]+=len(take)
            folder=OUT/name;folder.mkdir(exist_ok=True)
            (folder/'config.json').write_text(json.dumps(dict(diagnostic=name,arm=arm,rule=rule,labels_in_selection=rule=='oracle_help',source=__file__),indent=2)+'\n')
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));combos[name]=folder
            frac[name]={ds:dict(changed_windows=sel[ds][0],windows=sel[ds][1],fraction=sel[ds][0]/sel[ds][1]) for ds in DATASETS}
    with ThreadPoolExecutor(6) as ex:
        for name in ex.map(lambda kv:evaluate(kv[1],kv[0]),combos.items()):print('EVALUATED',name,flush=True)
    r6=metric(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json');m={n:metric(OUT/'decoded'/n/'metrics.json') for n in combos}
    table={}
    for n in combos:
        arm=n.split('__')[0];full=m[f'{arm}__full']
        table[n]=dict(selected=frac[n],metrics={ds:{k:m[n][ds][k] for k in METRICS} for ds in DATASETS},
            gain_vs_r6={ds:{k:m[n][ds][k]-r6[ds][k] for k in METRICS} for ds in DATASETS},
            within_gain_share_of_full={ds:(m[n][ds]['within_video_macro_ROC_AUC']-r6[ds]['within_video_macro_ROC_AUC'])/max(1e-9,full[ds]['within_video_macro_ROC_AUC']-r6[ds]['within_video_macro_ROC_AUC']) for ds in DATASETS})
    (OUT/'summary.json').write_text(json.dumps(dict(scope='development-selected error analysis; label-free rules except oracle_help; test GT inside the sole evaluator and in oracle_help only',
        r6={ds:{k:r6[ds][k] for k in METRICS} for ds in DATASETS},table=table),indent=2)+'\n')
    print('SELECTION_ANALYSIS_DONE',flush=True)


if __name__=='__main__':main()
