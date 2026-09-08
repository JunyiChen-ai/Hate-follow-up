#!/usr/bin/env python3
"""Frozen text-core cohesion confirmation projector; never reads ground truth."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
def load(path,method):
 out={}
 for l in Path(path).open():
  r=json.loads(l)
  if r.get('method')==method:out[(r['dataset'],r['video_id'])]=r
 return out
def interval(r):return tuple(map(float,r['intervals'][0][:2])) if r and r.get('intervals') else None
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)
def rank01(x):o=np.argsort(x,kind='stable');r=np.empty(len(x));r[o]=np.arange(len(x));return (r+.5)/len(x)
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))
def transport(base,evidence):
 base=np.asarray(base,float);z=np.log(np.clip(base,1e-6,1-1e-6)/np.clip(1-base,1e-6,1));r=rank01(evidence);r-=r.mean();rz=np.median(abs(z-np.median(z)))+1e-6;rr=np.median(abs(r-np.median(r)))+1e-6;changed=z+.01*r*rz/rr;target=base.mean();lo,hi=-20.,20.
 for _ in range(60):
  m=(lo+hi)/2
  if sigmoid(changed+m).mean()<target:lo=m
  else:hi=m
 return sigmoid(changed+(lo+hi)/2)
def cohesion(x,core,duration,n,shift=False):
 x=norm(np.asarray(x,float));
 if shift:x=np.roll(x,max(1,len(x)//2),axis=0)
 lo=int(np.clip(np.floor(core[0]),0,len(x)-1));hi=int(np.clip(np.ceil(core[1]),lo+1,len(x)));p=x[lo:hi].mean(0);p/=max(np.linalg.norm(p),1e-9);c=x@p
 return np.interp(np.linspace(0,len(c)-1,n),np.arange(len(c)),c)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--base-method',required=True);ap.add_argument('--core-method',required=True);ap.add_argument('--text-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 base=load(a.predictions,a.base_method);cores=load(a.predictions,a.core_method);counts={'factual':0,'shifted':0,'fallback':0}
 with a.out.open('w') as h:
  for key,row in sorted(base.items()):
   core=interval(cores.get(key));path=a.text_dir/(key[1]+'.npy');curves={}
   if core and path.exists() and row.get('score_curve'):
    x=np.load(path);n=len(row['score_curve']);curves['factual']=transport(row['score_curve'],cohesion(x,core,float(row['duration']),n));curves['shifted']=transport(row['score_curve'],cohesion(x,core,float(row['duration']),n,True));counts['factual']+=1;counts['shifted']+=1
   else:counts['fallback']+=1
   for arm,method in (('base','sealed_confirmation_base'),('factual','sealed_text_core_cohesion_g0p01'),('shifted','sealed_text_shifted_core_g0p01')):
    out=dict(row);out['method']=method;out['score_curve']=(curves.get(arm,np.asarray(row['score_curve'],float))).tolist();out['raw']={**out.get('raw',{}),'gt_access':False,'confirmation_frozen':True,'text_cohesion_arm':arm,'gain':.01 if arm!='base' else 0.,'video_mean_preserved':True};h.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps({'n':len(base),**counts},indent=2))
if __name__=='__main__':main()
