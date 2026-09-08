#!/usr/bin/env python3
"""Project the frozen top-8 proposal support closure into prediction JSONL."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from scripts.label_free_adapt.schema import Interval,Prediction,append_jsonl,intervals_to_curve

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--proposals',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--top-k',type=int,default=8);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 for row in map(json.loads,a.proposals.open()):
  d=float(row['duration']);ps=row['proposals'][:a.top_k];n=max(1,math.floor(d*4));t=(np.arange(n)+.5)/4;field=np.zeros(n,int)
  for p in ps:field+=(t>=float(p['start']))&(t<float(p['end']))
  active=field>0;bounds=np.flatnonzero(np.diff(np.r_[False,active,False])).reshape(-1,2);center=.5*(ps[0]['start']+ps[0]['end']);inside=[x for x in bounds if x[0]/4<=center<x[1]/4]
  chosen=max(inside if inside else bounds,key=lambda x:x[1]-x[0]);score=float(field[chosen[0]:chosen[1]].mean()/a.top_k);interval=Interval(chosen[0]/4,chosen[1]/4,score)
  pred=Prediction('extent_closure_top8',row['dataset'],row['video_id'],d,score_curve=intervals_to_curve([interval],d),intervals=[interval],calls=0,modality_evidence={'proposal_support_curve':field.tolist(),'anchor_rank':1,'top_k':a.top_k,'closure_component':[int(chosen[0]),int(chosen[1])]},raw={'gt_access':False,'rule':'connected union-support component containing rank-1 center','proposal_source':'A10_VidGroupZeroShot'})
  append_jsonl(a.out,pred)
if __name__=='__main__':main()
