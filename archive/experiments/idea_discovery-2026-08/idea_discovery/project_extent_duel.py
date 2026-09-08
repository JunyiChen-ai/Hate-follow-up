#!/usr/bin/env python3
"""Apply frozen phase-consistent extent duels to CASA predictions (GT blind)."""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_ncl_tribunal import load
from scripts.label_free_adapt.schema import Interval,Prediction,append_jsonl
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--duels',type=Path,required=True);ap.add_argument('--casa',type=Path,required=True);ap.add_argument('--rank1',type=Path,required=True);ap.add_argument('--closure',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 a.out.parent.mkdir(parents=True,exist_ok=True);casa,r1,cl=map(load,(a.casa,a.rank1,a.closure));duels=defaultdict(list)
 for r in map(json.loads,a.duels.open()):duels[(r['dataset'],r['video_id'])].append(r)
 edits=0
 for key,g in casa.items():
  keep=bool(g.get('intervals'));z=duels.get(key,[]);stable_a=len(z)==2 and all(x['winner']=='A' for x in z);src=r1[key] if stable_a else cl[key];edits+=int(keep and stable_a)
  for method,source in [('casa_extent_duel_v1',src),('casa_closure_control',cl[key])]:
   iv=([Interval(float(x[0]),float(x[1]),float(x[2]) if len(x)>2 else 1.) for x in source['intervals']] if keep else []);curve=source['score_curve'] if keep else [0.]*len(source['score_curve'])
   append_jsonl(a.out,Prediction(method,key[0],key[1],float(source['duration']),score_curve=curve,intervals=iv,calls=0,
    modality_evidence={'casa_keep':keep,'stable_rank1_choice':stable_a,'duel_winners':[x['winner'] for x in z]},raw={'gt_access':False,'rule':'rank1 iff both frame phases choose A; else closure'}))
 print(json.dumps({'n':len(casa),'effective_edits':edits,'out':str(a.out)}))
if __name__=='__main__':main()
