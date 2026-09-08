#!/usr/bin/env python3
"""Exact energy- and marginal-matched controls for WEBT corrections."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.idea_discovery.project_role_orthogonal_transport import load

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--base-method',required=True);p.add_argument('--text',type=Path,required=True);p.add_argument('--text-method',required=True);p.add_argument('--webt',type=Path,required=True);p.add_argument('--webt-method',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 B=load(a.base,a.base_method);T=load(a.text,a.text_method);W=load(a.webt,a.webt_method);keys=sorted(set(B)&set(T)&set(W));errs=[]
 with a.out.open('w') as h:
  for k in keys:
   q=np.asarray(B[k]['score_curve'],float);dt=np.asarray(T[k]['score_curve'],float)-q;dw=np.asarray(W[k]['score_curve'],float)-q
   l2=np.linalg.norm(dw);tl2=np.linalg.norm(dt);d_l2=dt*(l2/tl2) if tl2>1e-15 else np.zeros_like(dt)
   # Exact marginal control: same multiset of correction values as WEBT,
   # assigned monotonically according to the ordinary T correction ranks.
   order=np.argsort(dt,kind='stable');d_marg=np.empty_like(dw);d_marg[order]=np.sort(dw,kind='stable')
   variants={'webt_energy_base_v1':q,'webt_energy_factual_v1':q+dw,'webt_energy_l2_text_v1':q+d_l2,'webt_energy_marginal_text_v1':q+d_marg}
   for method,scores in variants.items():
    if np.min(scores)<0 or np.max(scores)>1:raise RuntimeError(f'out of range {k} {method}')
    out=dict(B[k]);out['method']=method;out['score_curve']=scores.tolist();out['raw']={**out.get('raw',{}),'gt_access':False,'energy_control':method,'video_mean_preserved':True,'intervals_preserved':True};h.write(json.dumps(out,separators=(',',':'))+'\n')
   errs.append({'mean':abs(d_marg.mean()-dw.mean()),'l1':abs(np.linalg.norm(d_marg,1)-np.linalg.norm(dw,1)),'l2':abs(np.linalg.norm(d_marg)-np.linalg.norm(dw)),'max':abs(np.max(np.abs(d_marg))-np.max(np.abs(dw)))})
 print(json.dumps({'n':len(keys),'max_invariant_error':{x:max(e[x] for e in errs) for x in ('mean','l1','l2','max')}},indent=2))
if __name__=='__main__':main()
