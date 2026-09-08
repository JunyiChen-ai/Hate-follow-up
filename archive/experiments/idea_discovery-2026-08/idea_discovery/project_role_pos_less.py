#!/usr/bin/env python3
"""Counterfactual role-binding Pareto-poset projection of LESS.

MLLM outputs never replace the dense field. A temporal inequality is admitted
only when visual and transcript counterfactual margins agree on its direction.
The decoder then applies the minimum-L2 bin-offset correction satisfying those
Pareto-consensus laws. Missing caches fall back exactly to LESS.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

def load(path,method=None):
 out={}
 for r in map(json.loads,Path(path).open()):
  if method is None or r['method']==method:out[(r['dataset'],r['video_id'])]=r
 return out
def project(base,vm,tm,gap):
 k=min(len(vm),len(tm));vm=np.asarray(vm[:k],float);tm=np.asarray(tm[:k],float)
 edges=[(i,j) for i in range(k) for j in range(k) if vm[i]-vm[j]>=gap and tm[i]-tm[j]>=gap]
 if not edges:return base.copy(),0,True,0.
 bounds=np.linspace(0,len(base),k+1).astype(int);membership=np.zeros((len(base),k));means=[]
 for i in range(k):membership[bounds[i]:bounds[i+1],i]=1;means.append(float(np.mean(base[bounds[i]:bounds[i+1]])))
 A=np.zeros((len(edges),k));b=np.zeros(len(edges))
 for q,(i,j) in enumerate(edges):A[q,i]=1;A[q,j]=-1;b[q]=means[i]-means[j]
 res=minimize(lambda x:.5*float(x@x),np.zeros(k),jac=lambda x:x,constraints={'type':'ineq','fun':lambda x:b+A@x,'jac':lambda x:A},method='SLSQP',options={'ftol':1e-10,'maxiter':500})
 score=base+membership@res.x;bm=np.asarray([np.mean(score[bounds[i]:bounds[i+1]]) for i in range(k)]);ok=bool(res.success and all(bm[i]>=bm[j]-1e-6 for i,j in edges));return (score if ok else base.copy()),len(edges),ok,float(np.linalg.norm(res.x))
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--visual',type=Path,required=True);p.add_argument('--transcript',type=Path,required=True);p.add_argument('--gap',type=float,default=.5);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 base=load(a.base,'fact_less_t3al_dualgeo_midpoint_v5');visual=load(a.visual);text=load(a.transcript);audit={}
 with a.out.open('w') as f:
  for key,row in sorted(base.items()):
   p0=np.clip(np.asarray(row['score_curve'],float),1e-5,1-1e-5);logit=np.log(p0/(1-p0));vr=visual.get(key);tr=text.get(key);vm=vr.get('modality_evidence',{}).get('binding_margin',[]) if vr else [];tm=tr.get('modality_evidence',{}).get('binding_margin',[]) if tr else []
   for control,controlled in [('aligned',tm),('shift',np.roll(tm,max(1,len(tm)//3)).tolist() if tm else [])]:
    score,nedges,ok,norm=project(logit,vm,controlled,a.gap) if vm and controlled else (logit.copy(),0,True,0.);post=1/(1+np.exp(-np.clip(score,-30,30)));changed=bool(np.max(np.abs(post-p0))>1e-8);audit[(control,'covered')]=audit.get((control,'covered'),0)+int(bool(vm and controlled));audit[(control,'changed')]=audit.get((control,'changed'),0)+int(changed);audit[(control,'edges')]=audit.get((control,'edges'),0)+nedges
    out=dict(row);out['method']=f'role_pos_less_{control}_v1';out['score_curve']=post.tolist();out['raw']={**row.get('raw',{}),'projection':'counterfactual_role_pareto_poset','role_views':['visual','transcript'],'role_gap':a.gap,'n_edges':nedges,'projection_norm':norm,'feasible':ok,'exact_fallback':not changed,'gt_access':False}
    f.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps({'/'.join(k):v for k,v in audit.items()},indent=2))
if __name__=='__main__':main()
