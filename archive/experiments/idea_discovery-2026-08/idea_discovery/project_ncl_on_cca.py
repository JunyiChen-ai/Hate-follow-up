#!/usr/bin/env python3
"""GT-free factorial of NCL amendments on the stronger CCA base."""
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
  modality_evidence=evidence,raw={'gt_access':False,'extent_source':source['method']}))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--text',type=Path,required=True);ap.add_argument('--visual',type=Path,required=True)
 ap.add_argument('--extent',type=Path,required=True);ap.add_argument('--cca',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 a.out.parent.mkdir(parents=True,exist_ok=True);text,visual,extent,cca=map(load,(a.text,a.visual,a.extent,a.cca))
 for key,row in text.items():
  source=extent[key];base_keep=bool(cca[key].get('intervals'));scores=[float(x['log_odds']) for x in row['observations']]
  tau=float(row['config']['null_log_odds']);text_keep=any(z>tau for z in scores)
  margins=list(visual.get(key,{}).get('modality_evidence',{}).get('margins',[]))
  vis=(len(margins)==2 and all(float(x)>=0 for x in margins) and any(float(x)>0 for x in margins))
  policies={
   'cca_control':base_keep,
   'ncl_delete_text':base_keep and text_keep,
   'ncl_delete_guarded':base_keep and (text_keep or vis),
   'ncl_rescue_text':base_keep or text_keep,
   'ncl_rescue_joint':base_keep or (text_keep and vis),
   'ncl_symmetric_text':text_keep,
   'ncl_bicameral_strict':(base_keep and (text_keep or vis)) or ((not base_keep) and text_keep and vis),
   'ncl_delete_guarded_rescue_text':(base_keep and (text_keep or vis)) or ((not base_keep) and text_keep),
  }
  evidence={'cca_nonempty':base_keep,'text_nonempty':text_keep,'threshold':tau,'scores':scores,
            'visual_counterwitness':vis,'visual_margins':margins}
  for method,keep in policies.items():emit(a.out,method,key,source,keep,evidence)
 print(json.dumps({'n':len(text),'methods':8,'out':str(a.out)}))
if __name__=='__main__':main()
