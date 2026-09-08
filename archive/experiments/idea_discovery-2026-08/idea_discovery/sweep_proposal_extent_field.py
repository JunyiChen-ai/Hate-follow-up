#!/usr/bin/env python3
"""Hash-split audit of label-free extent recovery from an A10 proposal set."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from scripts.label_free_adapt.evaluate import interval_f1

DATASETS=('HateMM','HateClipSeg','MHC','MHC_zh')
def split(k):return 'dev' if int(hashlib.sha256(f'extent-field-v1\0{k[0]}\0{k[1]}'.encode()).hexdigest(),16)%2==0 else 'holdout'
def components(mask):return np.flatnonzero(np.diff(np.r_[False,mask,False])).reshape(-1,2)
def decode(row,k,threshold,beta,rule):
 ps=row['proposals'][:k];d=float(row['duration']);n=max(1,math.floor(d*4));t=(np.arange(n)+.5)/4
 if rule=='longest':
  p=max(ps,key=lambda x:x['end']-x['start']);return [[p['start'],p['end'],1.]]
 weights=np.exp(-beta*np.arange(k));field=np.zeros(n)
 for w,p in zip(weights,ps):field+=w*((t>=p['start'])&(t<p['end']))
 mask=field>=threshold*weights.sum();bounds=components(mask)
 if not len(bounds):p=ps[0];return [[p['start'],p['end'],1.]]
 center=.5*(ps[0]['start']+ps[0]['end']);containing=[x for x in bounds if x[0]/4<=center<x[1]/4]
 if containing:chosen=max(containing,key=lambda x:x[1]-x[0])
 else:chosen=max(bounds,key=lambda x:float(field[x[0]:x[1]].sum()))
 return [[chosen[0]/4,chosen[1]/4,float(field[chosen[0]:chosen[1]].mean()/weights.sum())]]
def metric(keys,y,pred):
 per={}
 for d in DATASETS:
  kd=[x for x in keys if x[0]==d];per[d]=interval_f1({x:y[x] for x in kd},{x:{'intervals':pred[x]} for x in kd})
 return {m:float(np.mean([per[d][m] for d in DATASETS])) for m in ('interval_F1@0.3','interval_F1@0.5','interval_F1@0.7')},per
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--proposals',type=Path,required=True);ap.add_argument('--gt-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();rows={(r['dataset'],r['video_id']):r for r in map(json.loads,a.proposals.open())};y={}
 for d in DATASETS:
  z=np.load(a.gt_dir/f'{d}.npz',allow_pickle=True)
  for i,v in enumerate(z['video_ids']):
   if (d,str(v)) in rows:y[(d,str(v))]=np.asarray(z['y4'][i],np.int8)
 keys=sorted(set(rows)&set(y));parts={s:[k for k in keys if split(k)==s] for s in ('dev','holdout')}
 configs=[(k,q,b,'field') for k in (2,3,4,6,8) for q in (.05,.1,.2,.3,.4,.5,.6,.7,.8,.9) for b in (0.,.15,.35,.7,1.5)]+[(8,0.,0.,'longest')]
 ranked=[]
 for c in configs:
  pred={k:decode(rows[k],*c) for k in parts['dev']};m,_=metric(parts['dev'],y,pred);ranked.append((m['interval_F1@0.5'],m['interval_F1@0.3'],c,m))
 ranked.sort(reverse=True,key=lambda x:(x[0],x[1]));best=ranked[0][2];result={'n':len(keys),'split_sizes':{s:len(x) for s,x in parts.items()},'best':best,'dev_top10':[{'config':x[2],'metrics':x[3]} for x in ranked[:10]],'evaluation':{}}
 for s,ks in parts.items():
  p={k:decode(rows[k],*best) for k in ks};m,per=metric(ks,y,p);base={k:[[rows[k]['proposals'][0]['start'],rows[k]['proposals'][0]['end'],1.]] for k in ks};bm,bper=metric(ks,y,base);long={k:decode(rows[k],8,0.,0.,'longest') for k in ks};lm,lper=metric(ks,y,long);result['evaluation'][s]={'field':m,'a10':bm,'longest':lm,'delta_vs_a10':{x:m[x]-bm[x] for x in m},'per_dataset_field':per,'per_dataset_a10':bper,'per_dataset_longest':lper}
 a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
