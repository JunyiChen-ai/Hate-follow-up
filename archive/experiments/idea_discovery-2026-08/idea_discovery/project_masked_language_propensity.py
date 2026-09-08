#!/usr/bin/env python3
"""Video propensity from cached label-free masked transcript evidence."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from scripts.idea_discovery.project_pos_less import load_chunks
from scripts.idea_discovery.project_role_orthogonal_transport import load,sigmoid

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--base-method',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 rows=load(a.base,a.base_method);chunks=load_chunks()
 with a.out.open('w') as h:
  for key,row in sorted(rows.items()):
   p=np.clip(np.asarray(row['score_curve'],float),1e-6,1-1e-6);z=np.log(p/(1-p));base_prop=float(z.mean());values=[float(r.get('z_masked',r.get('z_isolated',-20))) for r in chunks.get(key,[])];values=[x for x in values if math.isfinite(x)];lang=max(values) if values else base_prop
   dense_peak=float(np.max(z))
   props={'masked_language_replace_v1':lang,
          'masked_language_equal_barycenter_v1':(base_prop+lang)/2,
          'masked_language_peak_triad_v1':(base_prop+lang+dense_peak)/3,
          'masked_language_peak_median_v1':float(np.median([base_prop,lang,dense_peak])),
          'masked_language_peak_pair_v1':(lang+dense_peak)/2,
          'masked_language_min_control_v1':min(values) if values else base_prop}
   for method,prop in props.items():
    out=dict(row);out['method']=method;out['score_curve']=sigmoid(prop+(z-z.mean())).tolist();out['raw']={**out.get('raw',{}),'gt_access':False,'module':'masked_language_video_propensity','dataset_parameters':0,'label_selected_parameters':0,'within_order_preserved':True,'intervals_preserved':True,'language_propensity':{'n_chunks':len(values),'masked_max':lang,'base_propensity':base_prop,'dense_peak':dense_peak,'selected':prop,'rule':method}};h.write(json.dumps(out,separators=(',',':'))+'\n')
if __name__=='__main__':main()
