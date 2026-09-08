#!/usr/bin/env python3
"""GT-isolated mechanism audit for the HYPER proposal selector."""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
from scripts.label_free_adapt.evaluate import binary_intervals, temporal_iou

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--proposals',type=Path,required=True);ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--gt-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 cohort={(r['dataset'],r['video_id']) for r in map(json.loads,a.manifest.open())}
 props={(r['dataset'],r['video_id']):r for r in map(json.loads,a.proposals.open()) if (r['dataset'],r['video_id']) in cohort}
 preds=[r for r in map(json.loads,a.predictions.open()) if (r['dataset'],r['video_id']) in cohort]
 gt={}
 for d in sorted({x[0] for x in cohort}):
  z=np.load(a.gt_dir/f'{d}.npz',allow_pickle=True)
  for i,v in enumerate(z['video_ids']):
   if (d,str(v)) in cohort:gt[(d,str(v))]=binary_intervals(np.asarray(z['y4'][i],np.int8))
 def q(p,targets):return max((temporal_iou((float(p['start']),float(p['end'])),x) for x in targets),default=0.)
 detail=[]
 for r in preds:
  key=(r['dataset'],r['video_id']); candidates=props[key]['proposals'][:4];qualities=[q(p,gt[key]) for p in candidates];best=max(qualities);oracle={i+1 for i,x in enumerate(qualities) if x==best};rank=int(r['modality_evidence']['selected_rank'])
  detail.append({'dataset':key[0],'video_id':key[1],'arm':r['method'],'selected_rank':rank,'oracle_ranks':sorted(oracle),'selected_iou':qualities[rank-1],'a10_iou':qualities[0],'oracle_iou':best,'correct_oracle_choice':rank in oracle,'strict_improvement':qualities[rank-1]>qualities[0],'strict_degradation':qualities[rank-1]<qualities[0],'fallback':r['modality_evidence']['condorcet_fallback']})
 summary={}
 for arm in sorted({x['arm'] for x in detail}):
  rows=[x for x in detail if x['arm']==arm];summary[arm]={'n':len(rows),'oracle_choices':sum(x['correct_oracle_choice'] for x in rows),'strict_improvements':sum(x['strict_improvement'] for x in rows),'strict_degradations':sum(x['strict_degradation'] for x in rows),'fallback':sum(x['fallback'] for x in rows),'selected_ranks':dict(Counter(x['selected_rank'] for x in rows)),'mean_a10_iou':float(np.mean([x['a10_iou'] for x in rows])),'mean_selected_iou':float(np.mean([x['selected_iou'] for x in rows])),'mean_oracle_iou':float(np.mean([x['oracle_iou'] for x in rows]))}
 out={'summary':summary,'detail':detail};a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
