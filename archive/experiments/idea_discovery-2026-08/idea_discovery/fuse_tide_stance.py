#!/usr/bin/env python3
"""Test the complementarity of complete-event and authorial-stance fields."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.idea_discovery.run_tide_u32 import bins_to_curve,positive_intervals
from scripts.label_free_adapt.schema import Prediction,append_jsonl
def load(path):
 out={}
 for r in map(json.loads,path.open()):out.setdefault((r['dataset'],r['video_id']),{})[r['method']]=r
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--factors',type=Path,required=True);ap.add_argument('--controls',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 b=load(a.base);f=load(a.factors);c=load(a.controls)
 for key,arms in b.items():
  joint=np.asarray(arms['u32_joint']['modality_evidence']['corrected_log_odds'],float);text=np.asarray(arms['u32_text']['modality_evidence']['corrected_log_odds'],float)
  stance=np.asarray(f[key]['tide_policy_factors']['modality_evidence']['factor_log_odds']['endorsement'],float);rescue=np.asarray(c[key]['tide_consensus_arb_rescue']['modality_evidence']['fused_log_odds'],float)
  variants={'tide_joint_or_stance':np.maximum(joint,stance),'tide_rescue_or_stance':np.maximum(rescue,stance),'tide_joint_stance_top2':np.sort(np.stack([joint,text,stance]),axis=0)[1:].mean(0),'tide_joint_stance_mean':np.stack([joint,stance]).mean(0)}
  d=float(arms['u32_joint']['duration'])
  for method,z in variants.items():
   p=1/(1+np.exp(-np.clip(z,-30,30)));append_jsonl(a.out,Prediction(method,key[0],key[1],d,score_curve=bins_to_curve(p,d).tolist(),intervals=positive_intervals(z,d),calls=0,modality_evidence={'fused_log_odds':z.tolist()},raw={'gt_access':False,'algebraic_control':True}))
if __name__=='__main__':main()
