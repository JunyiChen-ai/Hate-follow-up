#!/usr/bin/env python3
"""Use generic existence only when M1 declares no valid relation."""
import argparse,json
from pathlib import Path
from scripts.label_free_adapt.schema import Interval,Prediction,append_jsonl

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--main',type=Path,required=True);ap.add_argument('--joint',type=Path,required=True);ap.add_argument('--generic',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists(): raise RuntimeError(f'refusing existing output: {a.out}')
 def load(p): return {(r['dataset'],r['video_id']):r for r in (json.loads(x) for x in p.open())}
 m,j,g=load(a.main),load(a.joint),load(a.generic)
 for key in sorted(set(m)&set(j)&set(g)):
  rows=(m[key],j[key],g[key]);err=next((r.get('error') for r in rows if r.get('error')),None)
  if err: append_jsonl(a.out,Prediction('relation_envelope_fallback',key[0],key[1],rows[0]['duration'],error=err));continue
  relation=j[key]['modality_evidence']['event_relation']; use=relation['stance'] not in {'endorsement','ambiguous'}
  src=g[key] if use else m[key]; ints=[Interval(float(x[0]),float(x[1]),float(x[2]) if len(x)>2 else 1.) for x in src.get('intervals',[])]
  append_jsonl(a.out,Prediction('relation_envelope_fallback',key[0],key[1],float(src['duration']),score_curve=src['score_curve'],intervals=ints,calls=0,modality_evidence={'fallback_used':use,'event_relation':relation}))
if __name__=='__main__':main()
