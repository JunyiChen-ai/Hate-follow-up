#!/usr/bin/env python3
"""Matched evaluation and paired bootstrap for the ordinal query candidate."""
from pathlib import Path
import json,sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).parent));from prequential_write_gate_pilot import metrics,interp,ecdf
ROOT=Path(__file__).resolve().parents[2];GT=Path('/home/jehc223/Retrieval-hate/data/gt/frame_gt_4fps');D=['HateMM','HateClipSeg','MHC','MHC_zh'];R=ROOT/'results/idea_discovery/prompt_query';METHODS={'fixed':R/'abuse_fixed','absolute':R/'abuse_absolute','ordinal':R/'abuse','shuffle_parity':R/'abuse_shuffle_parity','reverse':R/'abuse_reverse'}
items={m:{d:[] for d in D} for m in METHODS};aucs={m:{d:{} for d in D} for m in METHODS}
for d in D:
 z=np.load(GT/(d+'.npz'),allow_pickle=True);gt={str(v):np.asarray(y,int) for v,s,y in zip(z['video_ids'],z['split'],z['y4']) if str(s)=='test'};common=set(gt)
 for p in METHODS.values():common&={x.stem for x in (p/d).glob('*.npy')}
 for m,p in METHODS.items():
  for v in sorted(common):
   y=gt[v];s=interp(ecdf(np.load(p/d/(v+'.npy'))),len(y));items[m][d].append((y,s))
   if len(np.unique(y))==2:aucs[m][d][v]=float(roc_auc_score(y,s))
def boot(diff,seed=20260826,n=20000):
 x=np.asarray(diff);rng=np.random.default_rng(seed);means=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(n)]);return {'delta':float(x.mean()),'ci95':[float(v) for v in np.quantile(means,[.025,.975])],'wins':int((x>0).sum()),'ties':int((x==0).sum()),'losses':int((x<0).sum()),'n':len(x)}
res={'metrics':{},'paired_vs_fixed':{}}
for m in METHODS:res['metrics'][m]={**{d:metrics(items[m][d]) for d in D},'all':metrics(sum((items[m][d] for d in D),[]))}
for d in D+['all']:
 ds=D if d=='all' else [d];diff=[]
 for x in ds:
  for v,a in aucs['ordinal'][x].items():diff.append(a-aucs['fixed'][x][v])
 res['paired_vs_fixed'][d]=boot(diff) if diff else None
run=json.loads((METHODS['ordinal']/'run.json').read_text());res['coverage']={d:{'adapted':sum(x['n_train']>0 for x in run['videos'] if x['dataset']==d),'total':sum(x['dataset']==d for x in run['videos'])} for d in D};out=R/'ordinal_candidate_eval.json';out.write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
