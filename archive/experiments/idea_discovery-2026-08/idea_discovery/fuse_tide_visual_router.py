#!/usr/bin/env python3
"""Controls for using visual evidence as a router rather than a fused score."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.idea_discovery.run_tide_u32 import bins_to_curve,positive_intervals
from scripts.label_free_adapt.schema import Prediction,append_jsonl
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 g={}
 for r in map(json.loads,a.input.open()):g.setdefault((r['dataset'],r['video_id']),{})[r['method']]=r
 for key,r in g.items():
  j=np.asarray(r['u32_joint']['modality_evidence']['corrected_log_odds']);v=np.asarray(r['u32_visual']['modality_evidence']['corrected_log_odds']);t=np.asarray(r['u32_text']['modality_evidence']['corrected_log_odds'])
  med=np.median(v);dev=np.abs(v-med);gate=dev>=np.median(dev)
  variants={'tide_max_joint_text':np.maximum(j,t),'tide_visual_sign_router':np.where(v>0,j,t),'tide_visual_change_router':np.where(gate,j,t),'tide_positive_visual_residual':t+np.clip(j-t,0,2),'tide_confident_joint_text':np.where(np.abs(j)>=np.abs(t),j,t)}
  d=float(r['u32_joint']['duration'])
  for method,z in variants.items():
   p=1/(1+np.exp(-np.clip(z,-30,30)));append_jsonl(a.out,Prediction(method,key[0],key[1],d,score_curve=bins_to_curve(p,d).tolist(),intervals=positive_intervals(z,d),calls=0,modality_evidence={'routed_log_odds':z.tolist()},raw={'gt_access':False,'router_control':True}))
if __name__=='__main__':main()
