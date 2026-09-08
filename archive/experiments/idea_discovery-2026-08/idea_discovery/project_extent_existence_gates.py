#!/usr/bin/env python3
"""Apply frozen, label-free event-existence gates to extent-calibrated intervals."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from scripts.label_free_adapt.schema import Prediction,Interval,append_jsonl
def load(p):return {(r['dataset'],r['video_id']):r for r in map(json.loads,p.open())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--extent',type=Path,required=True);ap.add_argument('--a08',type=Path,required=True);ap.add_argument('--a12',type=Path,required=True);ap.add_argument('--text',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 extent=load(a.extent);x8=load(a.a08);x12=load(a.a12);xt=load(a.text) if a.text else {}
 random_veto=set()
 if xt:
  # Same per-dataset veto counts as the frozen tau=-12 route, but selected by hash.
  for dataset in sorted({k[0] for k in extent}):
   disagree=[k for k in extent if k[0]==dataset and bool(x8[k].get('intervals')) != bool(x12[k].get('intervals'))]
   count=sum(float(xt[k]['log_odds']) < -12 for k in disagree)
   ordered=sorted(disagree,key=lambda k:hashlib.sha256(f'vasta-random-v1\0{k[0]}\0{k[1]}'.encode()).hexdigest())
   random_veto.update(ordered[:count])
 for k,row in extent.items():
  gates={'gate_a08':bool(x8[k].get('intervals')),'gate_a12':bool(x12[k].get('intervals'))};gates['gate_and']=gates['gate_a08'] and gates['gate_a12'];gates['gate_or']=gates['gate_a08'] or gates['gate_a12']
  if xt:
   gates['gate_text']=bool(xt[k]['positive']);gates['gate_or3']=gates['gate_or'] or gates['gate_text'];gates['gate_majority3']=sum((gates['gate_a08'],gates['gate_a12'],gates['gate_text']))>=2;gates['gate_a12_or_text']=gates['gate_a12'] or gates['gate_text'];gates['gate_a08_or_text']=gates['gate_a08'] or gates['gate_text']
   for threshold in (2,4,6,8,10,12,14,16,18,20):
    gates[f'gate_visual_or_text_ge{threshold}']=gates['gate_or'] or float(xt[k]['log_odds'])>=threshold
   for threshold in (-20,-16,-12,-8,-4,0):
    gates[f'gate_visual_and_text_ge{threshold}']=gates['gate_or'] and float(xt[k]['log_odds'])>=threshold
   for threshold in (-20,-16,-12,-8,-4,0,4,8,12,16,20):
    # Visual consensus is authoritative; text arbitrates only A08/A12 disagreement.
    both=gates['gate_a08'] and gates['gate_a12'];disagree=gates['gate_a08'] != gates['gate_a12']
    gates[f'gate_consensus_text_ge{threshold}']=both or (disagree and float(xt[k]['log_odds'])>=threshold)
   gates['gate_random_route_m12']=gates['gate_or'] and k not in random_veto
   gates['gate_inverted_text_m12']=gates['gate_or'] and (not both or float(xt[k]['log_odds'])>=-12)
  base=[Interval(float(x[0]),float(x[1]),float(x[2]) if len(x)>2 else 1.) for x in row['intervals']]
  for method,keep in gates.items():
   iv=base if keep else [];curve=row['score_curve'] if keep else [0.]*len(row['score_curve']);append_jsonl(a.out,Prediction('extent_'+method,k[0],k[1],float(row['duration']),score_curve=curve,intervals=iv,calls=0,modality_evidence={'existence_gate':method,'a08_nonempty':gates['gate_a08'],'a12_nonempty':gates['gate_a12'],'text_positive':gates.get('gate_text')},raw={'gt_access':False,'extent_source':row['method']}))
if __name__=='__main__':main()
