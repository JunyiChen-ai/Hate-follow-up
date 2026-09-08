#!/usr/bin/env python3
"""Deterministic controls for whether arbitration may veto or only rescue."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.idea_discovery.run_tide_u32 import bins_to_curve,positive_intervals
from scripts.label_free_adapt.schema import Prediction,append_jsonl

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fields',type=Path,required=True);ap.add_argument('--arbitration',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 base={}
 for line in a.fields.open():
  r=json.loads(line);base.setdefault((r['dataset'],r['video_id']),{})[r['method']]=r
 arb={(r['dataset'],r['video_id']):r for r in map(json.loads,a.arbitration.open())}
 for key,r in arb.items():
  j=np.asarray(base[key]['u32_joint']['modality_evidence']['corrected_log_odds'],float)
  m=np.median(np.stack([base[key][x]['modality_evidence']['corrected_log_odds'] for x in ('u32_joint','u32_visual','u32_text')]),axis=0)
  q=np.asarray(r['modality_evidence']['final_log_odds'],float);idx=np.asarray(r['modality_evidence']['disputed_bins'],int)
  # q equals median outside disputed bins.
  rescue=j.copy();rescue[idx]=np.maximum(j[idx],q[idx])
  consensus_rescue=m.copy();consensus_rescue[idx]=np.maximum(m[idx],q[idx])
  confidence=j.copy();take=np.zeros(32,bool);take[idx]=np.abs(q[idx])>np.abs(j[idx]);confidence[take]=q[take]
  for method,z in {'tide_joint_arb_rescue':rescue,'tide_consensus_arb_rescue':consensus_rescue,'tide_confidence_arb':confidence}.items():
   p=1/(1+np.exp(-np.clip(z,-30,30)));d=float(r['duration'])
   append_jsonl(a.out,Prediction(method,key[0],key[1],d,score_curve=bins_to_curve(p,d).tolist(),intervals=positive_intervals(z,d),calls=0,modality_evidence={'fused_log_odds':z.tolist()},raw={'gt_access':False,'algebraic_control':True}))
if __name__=='__main__':main()
