#!/usr/bin/env python3
"""GT-free bicameral decoder for proposal text evidence and visual warrants."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_ncl_tribunal import load
from scripts.label_free_adapt.schema import Interval,Prediction,append_jsonl

def emit(out,method,key,source,keep,evidence):
 iv=([Interval(float(x[0]),float(x[1]),float(x[2]) if len(x)>2 else 1.) for x in source['intervals']] if keep else [])
 curve=source['score_curve'] if keep else [0.]*len(source['score_curve'])
 append_jsonl(out,Prediction(method,key[0],key[1],float(source['duration']),score_curve=curve,intervals=iv,calls=0,
  modality_evidence=evidence,raw={'gt_access':False,'base_method':source['method']}))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--text',type=Path,required=True);ap.add_argument('--visual',type=Path,required=True)
 ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 a.out.parent.mkdir(parents=True,exist_ok=True);text,visual,base=load(a.text),load(a.visual),load(a.base)
 for key,row in text.items():
  source=base[key];margins=list(visual.get(key,{}).get('modality_evidence',{}).get('margins',[]))
  counter=(len(margins)==2 and all(float(x)>=0 for x in margins) and any(float(x)>0 for x in margins))
  scores=[float(x['log_odds']) for x in row['observations']];null=float(row['config']['null_log_odds'])
  emit(a.out,'ncl_extent_control',key,source,True,{'control':'base'})
  for name,tau in [('zero',0.),('null',null),('m8',-8.),('m4',-4.)]:
   witness=[i+1 for i,z in enumerate(scores) if z>tau];text_nonempty=bool(witness);keep=text_nonempty or counter
   emit(a.out,f'ncl_bicameral_{name}',key,source,keep,{'proposal_scores':scores,'threshold':tau,
    'text_witness_ranks':witness,'visual_native_margins':margins,'visual_counterwitness':counter,'keep':keep})
 print(json.dumps({'n':len(text),'methods':5,'out':str(a.out)}))
if __name__=='__main__':main()
