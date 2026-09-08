#!/usr/bin/env python3
"""Per-video self-predictive temporal hazard, with no fitted gain or direction.

The base field itself determines whether evidence drifts early or late via the
closed-form least-squares projection of centered logits on centered normalized
time.  That projection is added once as a self-consistency correction; the
opposite projection is emitted as a direction control.  Video mean probability
is restored exactly.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.idea_discovery.project_role_orthogonal_transport import load,sigmoid

def restore(base,changed):
 target=float(base.mean());lo,hi=-30.,30.
 for _ in range(80):
  mid=(lo+hi)/2
  if float(sigmoid(changed+mid).mean())<target:lo=mid
  else:hi=mid
 return sigmoid(changed+(lo+hi)/2)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--base-method',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 rows=load(a.base,a.base_method)
 with a.out.open('w') as h:
  for key,row in sorted(rows.items()):
   base=np.asarray(row['score_curve'],float);z=np.log(np.clip(base,1e-6,1-1e-6)/np.clip(1-base,1e-6,1));t=np.linspace(-1.,1.,len(z));t-=t.mean();zc=z-z.mean();coef=float(t@zc/max(t@t,1e-12));hazard=coef*t
   for method,sign in [('self_predictive_hazard_v1',1.),('self_predictive_hazard_reverse_v1',-1.)]:
    out=dict(row);out['method']=method;out['score_curve']=restore(base,z+sign*hazard).tolist();out['raw']={**out.get('raw',{}),'gt_access':False,'module':'self_predictive_temporal_hazard','dataset_parameters':0,'closed_form_coefficient':coef,'direction_control':sign<0,'video_mean_preserved':True,'intervals_preserved':True};h.write(json.dumps(out,separators=(',',':'))+'\n')
if __name__=='__main__':main()
