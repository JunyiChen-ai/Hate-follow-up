#!/usr/bin/env python3
"""Label-free structural-risk selection among frozen transport candidates."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.idea_discovery.project_role_orthogonal_transport import load
from scripts.idea_discovery.project_trace_gcv import gcv_select,reconstruct,resize

def min_gcv(base):
 n=max(2,int(round(float(base['duration'])))+1);curve=np.asarray(base['score_curve'],float);logits=np.log(np.clip(curve,1e-6,1-1e-6)/np.clip(1-curve,1e-6,1));signal=resize(logits,n);signal-=signal.mean();fitted,strength,audit=gcv_select(signal,np.ones(n-1,float));risk=min(audit['gcv'].values()) if audit.get('gcv') else float('inf');variance=float(np.mean(signal*signal))+1e-12
 # Scale-free structural prediction risk so candidates with a compressed score
 # range are not preferred merely because their squared error is smaller.
 return risk/variance,fitted,strength,audit
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate-a',type=Path,required=True);p.add_argument('--method-a',required=True);p.add_argument('--candidate-b',type=Path,required=True);p.add_argument('--method-b',required=True);p.add_argument('--base',type=Path);p.add_argument('--base-method');p.add_argument('--rule',choices=('gcv','agreement'),default='gcv');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 A=load(a.candidate_a,a.method_a);B=load(a.candidate_b,a.method_b);base=load(a.base,a.base_method) if a.base else {};keys=sorted(set(A)&set(B));count={'a':0,'b':0}
 with a.out.open('w') as h:
  for k in keys:
   choices=[]
   for name,row in [('a',A[k]),('b',B[k])]:
    risk,fitted,strength,audit=min_gcv(row);curve=np.asarray(row['score_curve'],float);scores=curve.copy() if strength==0 else reconstruct(curve,fitted);choices.append((risk,name,row,scores,strength,audit))
   if a.rule=='agreement':
    if k not in base: raise KeyError(f'missing base {k}')
    q=np.asarray(base[k]['score_curve'],float);qa=np.asarray(A[k]['score_curve'],float);qb=np.asarray(B[k]['score_curve'],float);semantic=qa-q;bary=qb-qa;agreement=float(np.dot(semantic-semantic.mean(),bary-bary.mean()));name='b' if agreement>0 else 'a';risk,name,row,scores,strength,audit=next(x for x in choices if x[1]==name);selection='positive_semantic_barycentric_inner_product'
   else:
    risk,name,row,scores,strength,audit=min(choices,key=lambda x:(x[0],x[1]));agreement=None;selection='minimum_scale_free_gcv_risk'
   count[name]+=1;out=dict(row);out['method']='gcv_selected_transport_v1';out['score_curve']=scores.tolist();out['raw']={**out.get('raw',{}),'gt_access':False,'selection':selection,'agreement':agreement,'selected':name,'selected_method':a.method_a if name=='a' else a.method_b,'risks':{x[1]:x[0] for x in choices},'gcv':audit,'video_mean_preserved':True,'intervals_preserved':True};h.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps({'n':len(keys),'selected':count},indent=2))
if __name__=='__main__':main()
