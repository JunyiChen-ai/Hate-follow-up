#!/usr/bin/env python3
"""Continuous GCV on effective evidence runs rather than duplicated frames."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.idea_discovery.project_mdl_regime_compression import runs
from scripts.idea_discovery.project_role_orthogonal_transport import load
from scripts.idea_discovery.project_trace_continuous_gcv import select
from scripts.idea_discovery.project_trace_gcv import reconstruct

def order_for(n,key):
 seed=int.from_bytes(hashlib.sha256(('run-gcv/'+key).encode()).digest()[:8],'little');return np.random.default_rng(seed).permutation(n)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--base-method',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 rows=load(a.base,a.base_method)
 with a.out.open('w') as h:
  for key,row in sorted(rows.items()):
   scores=np.asarray(row['score_curve'],float);c=np.clip(scores,1e-6,1-1e-6);logits=np.log(c/(1-c));levels,weights,_=runs(logits)
   fit,strength,audit=select(levels-levels.mean());variants=[('run_gcv_v1',fit,strength,audit)]
   order=order_for(len(levels),key[0]+'/'+key[1]);pfit,pstrength,paudit=select(levels[order]-levels[order].mean());restored=np.empty_like(pfit);restored[order]=pfit;variants.append(('run_gcv_permuted_control_v1',restored,pstrength,paudit))
   for method,fitted,lam,record in variants:
    dense=np.repeat(fitted,weights);out=dict(row);out['method']=method;out['score_curve']=reconstruct(scores,dense-dense.mean()).tolist();out['raw']={**out.get('raw',{}),'gt_access':False,'module':'effective_run_continuous_gcv','dataset_parameters':0,'label_selected_parameters':0,'video_mean_preserved':True,'intervals_preserved':True,'run_gcv':{'input_frames':len(scores),'effective_runs':len(levels),'strength':lam,'gcv':record}};h.write(json.dumps(out,separators=(',',':'))+'\n')
if __name__=='__main__':main()
