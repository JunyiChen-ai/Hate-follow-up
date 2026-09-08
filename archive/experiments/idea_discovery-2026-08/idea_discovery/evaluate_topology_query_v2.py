#!/usr/bin/env python3
"""Clean matched evaluation, including held-chunk-only AUC."""
from pathlib import Path
import json,math,sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).parent));from prequential_write_gate_pilot import metrics,interp,ecdf,text_curve
ROOT=Path(__file__).resolve().parents[2];GT=Path('/home/jehc223/Retrieval-hate/data/gt/frame_gt_4fps');R=ROOT/'results/idea_discovery/topology_query_v2';D=['HateMM','HateClipSeg','MHC','MHC_zh'];METHODS={x:R/x for x in ['fixed','absolute','ordinal','topology','shuffle','reverse']};ASR={'HateMM':ROOT/'results/hatemm_localization/per_chunk.jsonl','HateClipSeg':ROOT/'results/reproduction/ours/hateclipseg/per_chunk.jsonl','MHC':ROOT/'results/reproduction/ours/mhclip_en/per_chunk.jsonl','MHC_zh':ROOT/'results/reproduction/ours/mhclip_zh/per_chunk.jsonl'}
res={'metrics':{},'held_chunk_auc':{},'coverage':{}};held={m:{d:{} for d in D} for m in METHODS};items={m:{d:[] for d in D} for m in METHODS};fusion={m:{d:[] for d in D} for m in METHODS};fauc={m:{d:{} for d in D} for m in METHODS};tauc={d:{} for d in D};teacher={d:[] for d in D}
for d in D:
 by={}
 for x in ASR[d].read_text().splitlines():r=json.loads(x);by.setdefault(r['video_id'],[]).append(r)
 z=np.load(GT/(d+'.npz'),allow_pickle=True);gm={str(v):(float(du),np.asarray(y,int),str(sp)) for v,du,y,sp in zip(z['video_ids'],z['duration'],z['y4'],z['split'])};common={v for v,(du,y,sp) in gm.items() if sp=='test' and v in by}
 for p in METHODS.values():common&={x.stem for x in (p/d).glob('*.npy')}
 for v in sorted(common):
  dur,y,_=gm[v];rr=sorted(by[v],key=lambda r:(float(r['span'][0]),float(r['span'][1])));mask=np.zeros(len(y),bool)
  for k,r in enumerate(rr):
   if k%2==0:
    lo=max(0,min(len(y)-1,int(float(r['span'][0])/dur*len(y))));hi=min(len(y),max(lo+1,int(math.ceil(float(r['span'][1])/dur*len(y)))));mask[lo:hi]=1
  tc,_=text_curve(rr,dur,len(y),4);tc=ecdf(tc);teacher[d].append((y,tc))
  if len(np.unique(y))==2:tauc[d][v]=float(roc_auc_score(y,tc))
  for m,p in METHODS.items():
   s=interp(ecdf(np.load(p/d/(v+'.npy'))),len(y));items[m][d].append((y,s));fs=.5*s+.5*tc;fusion[m][d].append((y,fs))
   if len(np.unique(y))==2:fauc[m][d][v]=float(roc_auc_score(y,fs))
   if mask.any() and len(np.unique(y[mask]))==2:held[m][d][v]=float(roc_auc_score(y[mask],s[mask]))
for m in METHODS:
 res['metrics'][m]={**{d:metrics(items[m][d]) for d in D},'all':metrics(sum((items[m][d] for d in D),[]))};res['held_chunk_auc'][m]={**{d:(float(np.mean(list(held[m][d].values()))) if held[m][d] else None) for d in D},'all':float(np.mean([v for d in D for v in held[m][d].values()]))}
res['fusion50']={m:{**{d:metrics(fusion[m][d]) for d in D},'all':metrics(sum((fusion[m][d] for d in D),[]))} for m in METHODS}
def boot(x,n=20000):
 x=np.asarray(x);rng=np.random.default_rng(20260826);z=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(n)]);return {'delta':float(x.mean()),'ci95':[float(v) for v in np.quantile(z,[.025,.975])],'n':len(x)}
res['fusion_topology_paired']={base:boot([fauc['topology'][d][v]-(tauc[d][v] if base=='teacher' else fauc[base][d][v]) for d in D for v in fauc['topology'][d]]) for base in ['teacher','fixed','absolute','ordinal','shuffle']}
res['metrics']['direct_teacher']={**{d:metrics(teacher[d]) for d in D},'all':metrics(sum((teacher[d] for d in D),[]))}
run=json.loads((R/'topology/run.json').read_text());res['coverage']={d:{'adapted':sum(x['n_train']>0 for x in run['videos'] if x['dataset']==d),'total':sum(x['dataset']==d for x in run['videos'])} for d in D};out=R/'eval.json';out.write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
