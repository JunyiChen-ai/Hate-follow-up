#!/usr/bin/env python3
"""Evaluate two methods on exactly the same per-dataset video IDs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.label_free_adapt.evaluate import pooled,within_video_macro,interval_f1

DATASETS=('HateMM','HateClipSeg','MHC','MHC_zh')
def load(path,method):
 rows={(r['dataset'],r['video_id']):r for r in map(json.loads,path.open()) if r.get('method')==method and not r.get('error') and r.get('score_curve')}
 if not rows: raise ValueError(f'no valid rows for method {method!r} in {path}')
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--a',type=Path,required=True);ap.add_argument('--method-a',required=True);ap.add_argument('--b',type=Path,required=True);ap.add_argument('--method-b',required=True);ap.add_argument('--gt-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);x=ap.parse_args()
 A=load(x.a,x.method_a);B=load(x.b,x.method_b);rows=[]
 for ds in DATASETS:
  gt=np.load(x.gt_dir/f'{ds}.npz',allow_pickle=True);y={str(v):np.asarray(gt['y4'][i],np.int8) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'};ids=sorted(set(y)&{k[1] for k in A if k[0]==ds}&{k[1] for k in B if k[0]==ds})
  if not ids: raise ValueError(f'empty common cohort for dataset {ds}')
  yc={v:y[v] for v in ids}
  for method,source in [(x.method_a,A),(x.method_b,B)]:
   pred={v:source[(ds,v)] for v in ids}
   mismatched={v:(len(yc[v]),len(r['score_curve'])) for v,r in pred.items() if len(yc[v])!=len(r['score_curve'])}
   if mismatched: raise ValueError(f'score/GT length mismatch for {method} on {ds}: {list(mismatched.items())[:5]} (n={len(mismatched)})')
   scores={v:np.asarray(r['score_curve'],float) for v,r in pred.items()};rows.append({'method':method,'dataset':ds,**pooled(yc,scores),**within_video_macro(yc,scores),**interval_f1(yc,pred),'n_common_videos':len(ids)})
 metrics=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC','interval_F1@0.3','interval_F1@0.5','interval_F1@0.7');macro=[]
 for method in (x.method_a,x.method_b):
  group=[r for r in rows if r['method']==method];macro.append({'method':method,'dataset':'MACRO','n_datasets':4,**{m:float(np.mean([r[m] for r in group if r.get(m) is not None])) for m in metrics}})
 result={'per_dataset':rows,'macro':macro,'cohort_rule':'per_dataset intersection of GT, method A, method B'};x.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(macro,indent=2,sort_keys=True))
if __name__=='__main__':main()
