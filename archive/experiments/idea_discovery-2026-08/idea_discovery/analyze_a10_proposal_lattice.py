#!/usr/bin/env python3
"""Diagnose whether the A10 top-8 oracle has label-free-selectable structure."""
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scripts.label_free_adapt.evaluate import binary_intervals,interval_f1,temporal_iou

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--proposals',type=Path,required=True);ap.add_argument('--gt-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 rows={(r['dataset'],r['video_id']):r for r in map(json.loads,a.proposals.open())};y={}
 for d in sorted({k[0] for k in rows}):
  z=np.load(a.gt_dir/f'{d}.npz',allow_pickle=True)
  for i,v in enumerate(z['video_ids']):
   if (d,str(v)) in rows:y[(d,str(v))]=np.asarray(z['y4'][i],np.int8)
 keys=sorted(set(rows)&set(y))
 def qi(p,t):return max((temporal_iou((p['start'],p['end']),x) for x in t),default=0.)
 detail=[]
 for k in keys:
  ps=rows[k]['proposals'][:8];targets=binary_intervals(y[k]);qs=[qi(p,targets) for p in ps];best=max(qs);winners=[i for i,q in enumerate(qs) if q==best]
  detail.append({'dataset':k[0],'video_id':k[1],'positive':bool(targets),'qualities':qs,'oracle_ranks':[i+1 for i in winners],'strict_nonrank1':best>qs[0],
   'features':[{'rank':i+1,'length_frac':(p['end']-p['start'])/rows[k]['duration'],'start_frac':p['start']/rows[k]['duration'],'end_frac':p['end']/rows[k]['duration'],'logit':p['logit']} for i,p in enumerate(ps)]})
 policies={**{f'rank{i+1}':lambda ps,i=i:i for i in range(8)},'shortest':lambda ps:int(np.argmin([p['end']-p['start'] for p in ps])),'longest':lambda ps:int(np.argmax([p['end']-p['start'] for p in ps])),'middle_length':lambda ps:int(np.argsort([p['end']-p['start'] for p in ps])[len(ps)//2])}
 scores={}
 for name,fn in policies.items():
  sel={};
  for k in keys:
   p=rows[k]['proposals'][fn(rows[k]['proposals'][:8])];sel[k]={'intervals':[[p['start'],p['end'],1.]]}
  per={}
  for d in sorted({k[0] for k in keys}):
   kd=[k for k in keys if k[0]==d];per[d]=interval_f1({k:y[k] for k in kd},{k:sel[k] for k in kd})
  scores[name]={'macro':{m:float(np.mean([x[m] for x in per.values()])) for m in ('interval_F1@0.3','interval_F1@0.5','interval_F1@0.7')},'per_dataset':per}
 out={'n':len(keys),'policy_scores':scores,'oracle_rank_counts':dict(Counter(r for x in detail for r in x['oracle_ranks'])),'strict_nonrank1':sum(x['strict_nonrank1'] for x in detail),'strict_nonrank1_by_dataset':dict(Counter(x['dataset'] for x in detail if x['strict_nonrank1'])),'detail':detail};a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='detail'},indent=2))
if __name__=='__main__':main()
