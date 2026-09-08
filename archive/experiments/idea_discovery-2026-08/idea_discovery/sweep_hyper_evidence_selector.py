#!/usr/bin/env python3
"""Label-free proposal selection using cached multimodal evidence curves.

Hyperparameters are selected on a deterministic hash-development half and
reported once on the disjoint hash-holdout half.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scripts.label_free_adapt.evaluate import interval_f1

DATASETS=("HateMM","HateClipSeg","MHC","MHC_zh")

def load(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def subset(key):return 'dev' if int(hashlib.sha256(f'hyper-evidence-v1\0{key[0]}\0{key[1]}'.encode()).hexdigest(),16)%2==0 else 'holdout'
def ecdf(x):
 x=np.asarray(x,float);order=np.argsort(x,kind='mergesort');r=np.empty(len(x));r[order]=(np.arange(len(x))+.5)/len(x);return r
def region_mean(curve,duration,a,b):
 t=(np.arange(len(curve))+.5)/len(curve)*duration;m=(t>=a)&(t<b);return float(np.mean(curve[m])) if np.any(m) else 0.
def feature(curve,duration,p):
 a,b=float(p['start']),float(p['end']);w=b-a;inside=region_mean(curve,duration,a,b)
 left=region_mean(curve,duration,max(0.,a-w),a);right=region_mean(curve,duration,b,min(duration,b+w))
 return {'inside':inside,'contrast':inside-.5*(left+right),'length':w/duration,'logit':float(p['logit'])}
def metrics(keys,y,selected):
 per={}
 for d in DATASETS:
  kd=[k for k in keys if k[0]==d];per[d]=interval_f1({k:y[k] for k in kd},{k:{'intervals':[selected[k]]} for k in kd})
 return {m:float(np.mean([per[d][m] for d in DATASETS])) for m in ('interval_F1@0.3','interval_F1@0.5','interval_F1@0.7')},per

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--proposals',type=Path,required=True);ap.add_argument('--gt-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 evidence={(r['dataset'],r['video_id'],r['method']):r for r in load(a.evidence)};props={(r['dataset'],r['video_id']):r for r in load(a.proposals)}
 keys=sorted(set((d,v) for d,v,_ in evidence)&set(props));arms=sorted({m for d,v,m in evidence if (d,v) in keys})
 y={}
 for d in DATASETS:
  z=np.load(a.gt_dir/f'{d}.npz',allow_pickle=True)
  for i,v in enumerate(z['video_ids']):
   if (d,str(v)) in keys:y[(d,str(v))]=np.asarray(z['y4'][i],np.int8)
 keys=[k for k in keys if k in y];splits={s:[k for k in keys if subset(k)==s] for s in ('dev','holdout')}
 configs=[]
 for arm in arms:
  for mode in ('inside','contrast'):
   for alpha in (0.,.1,.25,.5,1.,2.,4.):
    for length_penalty in (0.,.1,.25,.5,1.):configs.append((arm,mode,alpha,length_penalty))
 def choose(config,ks):
  arm,mode,alpha,penalty=config;out={}
  for k in ks:
   curve=ecdf(evidence[(k[0],k[1],arm)]['score_curve']);candidates=props[k]['proposals'][:8];fs=[feature(curve,float(props[k]['duration']),p) for p in candidates]
   # Proposal rank is converted to a fixed [1,0] prior; no dataset labels enter.
   scores=[(1-i/max(1,len(fs)-1))+alpha*f[mode]-penalty*f['length'] for i,f in enumerate(fs)]
   j=int(np.argmax(scores));p=candidates[j];out[k]=[p['start'],p['end'],float(scores[j])]
  return out
 dev=[]
 for config in configs:
  selected=choose(config,splits['dev']);macro,_=metrics(splits['dev'],y,selected);dev.append((macro['interval_F1@0.5'],macro['interval_F1@0.3'],config,macro))
 dev.sort(reverse=True,key=lambda x:(x[0],x[1]));best=dev[0][2]
 result={'n':len(keys),'split_sizes':{s:len(v) for s,v in splits.items()},'best_config':{'arm':best[0],'mode':best[1],'alpha':best[2],'length_penalty':best[3]},'dev_top10':[{'config':x[2],'metrics':x[3]} for x in dev[:10]],'evaluation':{}}
 for s,ks in splits.items():
  selected=choose(best,ks);macro,per=metrics(ks,y,selected)
  baseline={k:[props[k]['proposals'][0]['start'],props[k]['proposals'][0]['end'],1.] for k in ks};base_macro,base_per=metrics(ks,y,baseline)
  result['evaluation'][s]={'method':macro,'baseline':base_macro,'delta':{m:macro[m]-base_macro[m] for m in macro},'per_dataset_method':per,'per_dataset_baseline':base_per,'changed':sum(selected[k][:2]!=baseline[k][:2] for k in ks)}
 a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
