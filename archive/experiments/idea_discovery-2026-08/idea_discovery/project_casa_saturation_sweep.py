#!/usr/bin/env python3
"""GT-free geometry sweep: CASA existence with rank1/closure saturation readout."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_ncl_tribunal import load
from scripts.label_free_adapt.schema import Interval,Prediction,append_jsonl
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--casa',type=Path,required=True);ap.add_argument('--rank1',type=Path,required=True);ap.add_argument('--closure',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 a.out.parent.mkdir(parents=True,exist_ok=True);casa,r1,cl=map(load,(a.casa,a.rank1,a.closure))
 for key,g in casa.items():
  b=r1[key];iv=b['intervals'];frac=(max(float(x[1]) for x in iv)-min(float(x[0]) for x in iv))/float(b['duration']);keep=bool(g.get('intervals'))
  for q in range(1,10):
   tau=q/10;src=b if frac>=tau else cl[key]
   intervals=([Interval(float(x[0]),float(x[1]),float(x[2]) if len(x)>2 else 1.) for x in src['intervals']] if keep else [])
   curve=src['score_curve'] if keep else [0.]*len(src['score_curve'])
   append_jsonl(a.out,Prediction(f'casa_sat_t{q}',key[0],key[1],float(src['duration']),score_curve=curve,intervals=intervals,calls=0,
    modality_evidence={'casa_keep':keep,'rank1_fraction':frac,'threshold':tau,'extent_source':'rank1' if frac>=tau else 'closure'},raw={'gt_access':False}))
 print(json.dumps({'n':len(casa),'methods':9,'out':str(a.out)}))
if __name__=='__main__':main()
