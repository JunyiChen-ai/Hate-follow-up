#!/usr/bin/env python3
"""Apply frozen global centered extent calibration to A10 intervals."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from scripts.label_free_adapt.schema import Interval,Prediction,append_jsonl,intervals_to_curve
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--factor',type=float,default=1.4899859333664718);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 for r in map(json.loads,a.input.open()):
  p=r['intervals'][0];s,e,d=float(p[0]),float(p[1]),float(r['duration']);length=min(d,(e-s)*a.factor);c=.5*(s+e);x=c-length/2;y=c+length/2
  if x<0:y-=x;x=0.
  if y>d:x-=y-d;y=d
  iv=Interval(max(0.,x),min(d,y),float(p[2]) if len(p)>2 else 1.);append_jsonl(a.out,Prediction('global_extent_calibration',r['dataset'],r['video_id'],d,score_curve=intervals_to_curve([iv],d),intervals=[iv],calls=0,modality_evidence={'factor':a.factor},raw={'gt_access':False,'source':r['method']}))
if __name__=='__main__':main()
