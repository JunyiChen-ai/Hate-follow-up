#!/usr/bin/env python3
"""Project cached TIDE policy factor fields into independently evaluable predictions."""
import argparse,json
from pathlib import Path
import numpy as np
from scripts.idea_discovery.run_tide_u32 import bins_to_curve,positive_intervals
from scripts.label_free_adapt.schema import Prediction,append_jsonl
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 for r in map(json.loads,a.input.open()):
  d=float(r['duration']);fields={k:np.asarray(v,float) for k,v in r['modality_evidence']['factor_log_odds'].items()}
  variants={**{f'tide_factor_{k}':v for k,v in fields.items()},
            'tide_factor_mean_core':np.stack([fields[k] for k in ('target','hostile_act','endorsement')]).mean(0),
            'tide_factor_top2_core':np.sort(np.stack([fields[k] for k in ('target','hostile_act','endorsement')]),axis=0)[1:].mean(0)}
  for method,z in variants.items():
   p=1/(1+np.exp(-np.clip(z,-30,30)));append_jsonl(a.out,Prediction(method,r['dataset'],r['video_id'],d,score_curve=bins_to_curve(p,d).tolist(),intervals=positive_intervals(z,d),calls=0,modality_evidence={'factor_projection':z.tolist()},raw={'gt_access':False}))
if __name__=='__main__':main()
