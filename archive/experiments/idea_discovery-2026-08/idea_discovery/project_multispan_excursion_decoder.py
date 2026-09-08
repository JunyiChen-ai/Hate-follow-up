#!/usr/bin/env python3
"""Label-free multi-span decoders from dense score excursion sets."""
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
def otsu(x,bins=64):
 x=np.asarray(x,float);lo,hi=float(x.min()),float(x.max())
 if hi<=lo:return hi
 hist,edges=np.histogram(x,bins=bins,range=(lo,hi));p=hist/hist.sum();cent=(edges[:-1]+edges[1:])/2;w=np.cumsum(p);mu=np.cumsum(p*cent);total=mu[-1];den=w*(1-w);score=np.where(den>0,(total*w-mu)**2/den,-1);return float(cent[int(np.argmax(score))])
def spans(mask,rate=4.,min_frames=4,merge_gap=2):
 mask=np.asarray(mask,bool).copy();idx=np.flatnonzero(np.diff(np.r_[False,mask,False])).reshape(-1,2);parts=[list(x) for x in idx if x[1]-x[0]>=min_frames]
 merged=[]
 for a,b in parts:
  if merged and a-merged[-1][1]<=merge_gap:merged[-1][1]=b
  else:merged.append([a,b])
 return [[a/rate,b/rate,1.] for a,b in merged]
def hysteresis(x,high,low):
 weak=x>=low;seed=x>=high;out=np.zeros(len(x),bool)
 for a,b in np.flatnonzero(np.diff(np.r_[False,weak,False])).reshape(-1,2):
  if seed[a:b].any():out[a:b]=True
 return out
def persistent_mask(x):
 qs=(.5,.6,.7,.8);masks=np.stack([x>=np.quantile(x,q) for q in qs]);count=np.zeros(len(x),int)
 # A point is retained if its connected event survives at least two filtration levels.
 for m in masks:count+=m
 return count>=2
def sparse_hysteresis_spans(x,rate=4.):
 low,high=np.quantile(x,[.6,.9]);parts=spans(hysteresis(x,high,low),rate=rate,min_frames=4,merge_gap=2)
 ranked=sorted(parts,key=lambda z:max(x[int(z[0]*rate):max(int(z[1]*rate),int(z[0]*rate)+1)]),reverse=True)[:4]
 return sorted(ranked,key=lambda z:z[0])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--base-method',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 base=load(a.base,a.base_method);methods=('multispan_otsu_v1','multispan_hysteresis_v1','multispan_persistence_v1','multispan_sparse_hysteresis_v2');stats={m:{'videos':0,'spans':0} for m in methods}
 with a.out.open('w') as h:
  for key,row in sorted(base.items()):
   x=np.asarray(row['score_curve'],float);fallback=row.get('intervals',[])
   if not fallback or len(x)<4:variants={m:fallback for m in methods};thresholds={}
   else:
    th=otsu(x);med=float(np.median(x));high=max(th,med);low=min(th,med)
    variants={methods[0]:spans(x>=th),methods[1]:spans(hysteresis(x,high,low)),methods[2]:spans(persistent_mask(x)),methods[3]:sparse_hysteresis_spans(x)};thresholds={'otsu':th,'median':med,'sparse_low_quantile':.6,'sparse_high_quantile':.9}
    variants={m:(v or fallback) for m,v in variants.items()}
   for m in methods:
    out=dict(row);out['method']=m;out['intervals']=variants[m];out['raw']={**out.get('raw',{}),'gt_access':False,'decoder':'multi_span_excursion','thresholds':thresholds,'exact_empty_fallback':not bool(fallback)};stats[m]['videos']+=int(len(variants[m])>1);stats[m]['spans']+=len(variants[m]);h.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps(stats,indent=2))
if __name__=='__main__':main()
