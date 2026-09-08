#!/usr/bin/env python3
"""Per-video extreme-value-debiased event propensity."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.idea_discovery.project_role_orthogonal_transport import load,sigmoid

NORMAL_ABS_MEDIAN=0.6744897501960817
METHODS=('ev_debiased_propensity_v1','ev_raw_max_control_v1','ev_duration_control_v1')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--base-method',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 rows=load(a.base,a.base_method)
 with a.out.open('w') as h:
  for _,row in sorted(rows.items()):
   p=np.clip(np.asarray(row['score_curve'],float),1e-6,1-1e-6);z=np.log(p/(1-p));residual=z-z.mean();sigma=float(np.median(np.abs(z-np.median(z)))/NORMAL_ABS_MEDIAN);null_peak=sigma*np.sqrt(2*np.log(max(2,len(z))));props={METHODS[0]:float(np.max(z)-null_peak),METHODS[1]:float(np.max(z)),METHODS[2]:float(-null_peak)}
   for method,prop in props.items():
    out=dict(row);out['method']=method;out['score_curve']=sigmoid(prop+residual).tolist();out['raw']={**out.get('raw',{}),'gt_access':False,'module':'extreme_value_debiased_video_propensity','dataset_parameters':0,'label_selected_parameters':0,'within_order_preserved':True,'intervals_preserved':True,'propensity':{'robust_sigma':sigma,'null_expected_peak':null_peak,'selected_propensity':prop,'rule':method}};h.write(json.dumps(out,separators=(',',':'))+'\n')
if __name__=='__main__':main()
