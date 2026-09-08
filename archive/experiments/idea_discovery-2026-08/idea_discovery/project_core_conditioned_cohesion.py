#!/usr/bin/env python3
"""Training-free core-conditioned multimodal temporal cohesion pilot."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

DATA_DIR={"HateMM":"hatemm","HateClipSeg":"hateclipseg","MHC":"mhclip_en","MHC_zh":"mhclip_zh"}
FEATURES={"V":"clip_b16_1fps","A":"vggish_1s","T":"bert_sentence_1fps"}

def load(path,method):
 out={}
 for l in Path(path).open():
  r=json.loads(l)
  if r.get('method')==method: out[(r['dataset'],r['video_id'])]=r
 return out
def interval(r): return tuple(map(float,r['intervals'][0][:2])) if r and r.get('intervals') else None
def rank01(x):
 x=np.asarray(x,float); order=np.argsort(x,kind='stable'); ranks=np.empty(len(x),float); ranks[order]=np.arange(len(x))
 return (ranks+.5)/max(1,len(x))
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))
def transport_residual(base_score, cohesion_score, gain=1.0):
 base_score=np.asarray(base_score,float)
 z=np.log(np.clip(base_score,1e-6,1-1e-6)/np.clip(1-base_score,1e-6,1))
 r=np.asarray(cohesion_score,float);r=r-np.mean(r)
 rz=np.median(np.abs(z-np.median(z)))+1e-6; rr=np.median(np.abs(r-np.median(r)))+1e-6
 changed=z+gain*r*(rz/rr)
 # Preserve video propensity exactly in probability space; cohesion is only a
 # within-video transport and must not reorder videos through a Jensen shift.
 target=float(np.mean(base_score)); lo,hi=-20.,20.
 for _ in range(60):
  mid=(lo+hi)/2
  if float(np.mean(sigmoid(changed+mid)))<target: lo=mid
  else: hi=mid
 return sigmoid(changed+(lo+hi)/2)
def resize(x,n):
 if len(x)==n:return np.asarray(x,float)
 pos=np.linspace(0,len(x)-1,n); return np.stack([np.interp(pos,np.arange(len(x)),x[:,j]) for j in range(x.shape[1])],1)
def cohesion(x,lo,hi):
 x=np.asarray(x,float); x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)
 lo=max(0,min(len(x)-1,lo)); hi=max(lo+1,min(len(x),hi)); proto=x[lo:hi].mean(0); proto/=max(np.linalg.norm(proto),1e-9)
 c=x@proto
 return c, float(np.std(c))
def strongest_boundaries(field,tight,broad,duration):
 n=len(field); to_i=lambda t:int(np.clip(round(t/max(duration,1e-9)*(n-1)),0,n-1))
 ts,te=map(to_i,tight); bs,be=map(to_i,broad); span=max(2,be-bs)
 l0=max(0,min(bs,ts)-span); r1=min(n-1,max(be,te)+span)
 # Left seeks the strongest rise into the core; right the strongest fall out.
 dl=np.diff(field[l0:max(ts+1,l0+2)]); dr=np.diff(field[min(te,r1-1):r1+1])
 li=l0+int(np.argmax(dl))+1 if len(dl) else ts
 ri=min(te,r1-1)+int(np.argmin(dr))+1 if len(dr) else te
 start=li/(n-1)*duration if n>1 else 0.; end=ri/(n-1)*duration if n>1 else duration
 return (start,end) if start<end and start<=tight[0] and end>=tight[1] else tight

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--base',type=Path,required=True);ap.add_argument('--base-method',required=True)
 ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--tight-method',default='fact_less_t3al_dualgeo_shorter_v5');ap.add_argument('--broad-method',default='fact_less_t3al_dualgeo_union_v5')
 ap.add_argument('--feature-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 if args.out.exists():raise RuntimeError(f'refusing existing output: {args.out}')
 base=load(args.base,args.base_method);tight=load(args.bank,args.tight_method);broad=load(args.bank,args.broad_method)
 methods=('core_cohesion_visual_v1','core_cohesion_multimodal_median_v1','core_cohesion_multimodal_geomean_v1',
          'core_cohesion_median_transport_v2','core_cohesion_geomean_transport_v2'); counts={m:0 for m in methods}
 gains=(.01,.025,.05,.125,.25,.5)
 methods=methods+tuple(f'core_cohesion_geomean_transport_g{str(g).replace(".","p")}_v3' for g in gains)
 subset_names=('V','A','T','VA','VT','AT')
 methods=methods+tuple(f'core_cohesion_subset_{name}_g0p01_v4' for name in subset_names)
 placebo_names=('shiftA','shiftT','shiftAT')
 methods=methods+tuple(f'core_cohesion_AT_{name}_g0p01_v5' for name in placebo_names)
 methods=methods+('core_cohesion_A_shift_g0p01_v5','core_cohesion_T_shift_g0p01_v5')
 counts={m:0 for m in methods}
 with args.out.open('w') as h:
  for key,row in sorted(base.items()):
   ti,br=interval(tight.get(key)),interval(broad.get(key)); duration=float(row['duration']); n=len(row['score_curve'])
   curves={}; shifted={}; reliability={}
   if ti is not None:
    lo=int(np.floor(ti[0]));hi=max(lo+1,int(np.ceil(ti[1])))
    for mod,folder in FEATURES.items():
     p=args.feature_root/folder/DATA_DIR[key[0]]/(key[1]+'.npy')
     if p.exists():
      x=np.load(p); c,var=cohesion(x,lo,hi)
      if var>1e-5:
       curves[mod]=np.interp(np.linspace(0,len(c)-1,n),np.arange(len(c)),c);reliability[mod]=var
       sx=np.roll(x,max(1,len(x)//2),axis=0); sc,_=cohesion(sx,lo,hi)
       shifted[mod]=np.interp(np.linspace(0,len(sc)-1,n),np.arange(len(sc)),sc)
   b=rank01(np.asarray(row['score_curve'],float))
   variants={}
   if 'V' in curves: variants[methods[0]]=np.sqrt(b*rank01(curves['V']))
   if curves:
    ranks=np.stack([rank01(x) for x in curves.values()])
    variants[methods[1]]=np.sqrt(b*np.median(ranks,axis=0))
    variants[methods[2]]=np.exp((np.log(np.clip(b,1e-6,1))+np.mean(np.log(np.clip(ranks,1e-6,1)),axis=0))/2)
    variants[methods[3]]=transport_residual(np.asarray(row['score_curve'],float),np.median(ranks,axis=0))
    variants[methods[4]]=transport_residual(np.asarray(row['score_curve'],float),np.mean(np.log(np.clip(ranks,1e-6,1)),axis=0))
    for method,gain in zip(methods[5:],gains):
     variants[method]=transport_residual(np.asarray(row['score_curve'],float),np.mean(np.log(np.clip(ranks,1e-6,1)),axis=0),gain)
    for method,name in zip(methods[5+len(gains):],subset_names):
     chosen=[curves[m] for m in name if m in curves]
     if len(chosen)==len(name):
      sr=np.stack([rank01(x) for x in chosen])
      variants[method]=transport_residual(np.asarray(row['score_curve'],float),np.mean(np.log(np.clip(sr,1e-6,1)),axis=0),.01)
    placebo_start=5+len(gains)+len(subset_names)
    placebo_sources=((shifted.get('A'),curves.get('T')),(curves.get('A'),shifted.get('T')),
                     (shifted.get('A'),shifted.get('T')))
    for method,source in zip(methods[placebo_start:],placebo_sources):
     if all(x is not None for x in source):
      sr=np.stack([rank01(x) for x in source])
      variants[method]=transport_residual(np.asarray(row['score_curve'],float),np.mean(np.log(np.clip(sr,1e-6,1)),axis=0),.01)
    for method,mod in zip(methods[placebo_start+len(placebo_names):],('A','T')):
     if mod in shifted:
      variants[method]=transport_residual(np.asarray(row['score_curve'],float),np.log(np.clip(rank01(shifted[mod]),1e-6,1)),.01)
   for method in methods:
    out=dict(row);out['method']=method;field=variants.get(method,np.asarray(row['score_curve'],float));chosen=interval(row)
    if ti is not None and br is not None and method in variants and method in methods[:3]:
     chosen=strongest_boundaries(field,ti,br,duration);counts[method]+=int(chosen!=interval(row))
    out['score_curve']=field.tolist();out['intervals']=[] if chosen is None else [[chosen[0],chosen[1],1.0]]
    out['raw']={**out.get('raw',{}),'gt_access':False,'cohesion_anchor':'tight_event_core','available_modalities':sorted(curves),'reliability':reliability,'fusion':method}
    h.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps({'n':len(base),'changed_intervals':counts},indent=2))
if __name__=='__main__':main()
