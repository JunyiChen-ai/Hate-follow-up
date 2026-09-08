#!/usr/bin/env python3
"""Unweighted logit barycenter of two frozen per-video score fields."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.idea_discovery.project_role_orthogonal_transport import load,sigmoid

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--a',type=Path,required=True);ap.add_argument('--method-a',required=True);ap.add_argument('--b',type=Path,required=True);ap.add_argument('--method-b',required=True);ap.add_argument('--out',type=Path,required=True);x=ap.parse_args()
 if x.out.exists():raise RuntimeError(f'refusing existing output: {x.out}')
 A=load(x.a,x.method_a);B=load(x.b,x.method_b);keys=sorted(set(A)&set(B))
 with x.out.open('w') as h:
  for key in keys:
   row=A[key];pa=np.clip(np.asarray(row['score_curve'],float),1e-6,1-1e-6);pb=np.clip(np.asarray(B[key]['score_curve'],float),1e-6,1-1e-6);za=np.log(pa/(1-pa));zb=np.log(pb/(1-pb));z=(za+zb)/2;target=float((pa.mean()+pb.mean())/2);lo,hi=-30.,30.
   for _ in range(80):
    mid=(lo+hi)/2
    if sigmoid(z+mid).mean()<target:lo=mid
    else:hi=mid
   out=dict(row);out['method']='equal_logit_barycenter_v1';out['score_curve']=sigmoid(z+(lo+hi)/2).tolist();out['raw']={**out.get('raw',{}),'gt_access':False,'module':'symmetric_equal_logit_barycenter','sources':[x.method_a,x.method_b],'learned_weights':0,'dataset_parameters':0,'label_selected_parameters':0,'intervals_preserved':True};h.write(json.dumps(out,separators=(',',':'))+'\n')
if __name__=='__main__':main()
