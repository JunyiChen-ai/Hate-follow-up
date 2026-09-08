#!/usr/bin/env python3
"""Set-union controls between TIDE anchors and frozen baseline intervals."""
import argparse,json
from pathlib import Path
from scripts.label_free_adapt.schema import Interval,Prediction,append_jsonl
def load(path):return {(r['dataset'],r['video_id']):r for r in map(json.loads,path.open())}
def ints(r):return [Interval(float(x[0]),float(x[1]),float(x[2]) if len(x)>2 else 1.) for x in (r or {}).get('intervals',[])]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tide',type=Path,required=True);ap.add_argument('--a10',type=Path,required=True);ap.add_argument('--a12',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 rows=[r for r in map(json.loads,a.tide.open()) if r['method']=='u32_joint'];x10=load(a.a10);x12=load(a.a12)
 for r in rows:
  key=(r['dataset'],r['video_id']);base=ints(r);variants={'tide_joint_plus_a10':base+ints(x10.get(key)),'tide_joint_plus_a12':base+ints(x12.get(key)),'tide_joint_plus_a10_a12':base+ints(x10.get(key))+ints(x12.get(key))}
  for method,iv in variants.items():append_jsonl(a.out,Prediction(method,key[0],key[1],float(r['duration']),score_curve=r['score_curve'],intervals=iv,calls=0,modality_evidence={'set_union':True},raw={'gt_access':False}))
if __name__=='__main__':main()
