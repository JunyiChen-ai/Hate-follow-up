#!/usr/bin/env python3
"""Non-destructive set rescue: stance may add, never alter, joint intervals."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.idea_discovery.run_tide_u32 import bins_to_curve,positive_intervals
from scripts.label_free_adapt.schema import Prediction,append_jsonl
def load(path):
 out={}
 for r in map(json.loads,path.open()):out[(r['dataset'],r['video_id'],r['method'])]=r
 return out
def overlap(a,b):return max(a.start,b.start)<min(a.end,b.end)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--factors',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 b=load(a.base);f=load(a.factors)
 keys={(d,v) for d,v,m in b if m=='u32_joint'}
 for d,v in keys:
  br=b[d,v,'u32_joint'];fr=f[d,v,'tide_policy_factors'];dur=float(br['duration'])
  joint=np.asarray(br['modality_evidence']['corrected_log_odds'],float);stance=np.asarray(fr['modality_evidence']['factor_log_odds']['endorsement'],float)
  anchors=positive_intervals(joint,dur);candidates=positive_intervals(stance,dur);added=[x for x in candidates if not any(overlap(x,y) for y in anchors)]
  dense=np.maximum(1/(1+np.exp(-np.clip(joint,-30,30))),1/(1+np.exp(-np.clip(stance,-30,30))))
  append_jsonl(a.out,Prediction('tide_nondestructive_stance_rescue',d,v,dur,score_curve=bins_to_curve(dense,dur).tolist(),intervals=sorted(anchors+added,key=lambda x:x.start),calls=0,modality_evidence={'joint_intervals':len(anchors),'stance_candidates':len(candidates),'stance_added':len(added)},raw={'gt_access':False,'set_rescue':True}))
if __name__=='__main__':main()
