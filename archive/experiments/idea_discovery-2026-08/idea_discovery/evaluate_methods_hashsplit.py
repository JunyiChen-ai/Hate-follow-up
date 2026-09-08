#!/usr/bin/env python3
"""Evaluate prediction methods on deterministic per-dataset hash halves."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.label_free_adapt.evaluate import interval_f1
DATASETS=('HateMM','HateClipSeg','MHC','MHC_zh')
def split(k):return 'dev' if int(hashlib.sha256(f'method-hash-v1\0{k[0]}\0{k[1]}'.encode()).hexdigest(),16)%2==0 else 'holdout'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--gt-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();rows={}
 for r in map(json.loads,a.predictions.open()):rows[(r['dataset'],r['video_id'],r['method'])]=r
 methods=sorted({k[2] for k in rows});keys=sorted({k[:2] for k in rows});y={}
 for d in DATASETS:
  z=np.load(a.gt_dir/f'{d}.npz',allow_pickle=True)
  for i,v in enumerate(z['video_ids']):
   if (d,str(v)) in keys:y[(d,str(v))]=np.asarray(z['y4'][i],np.int8)
 result={}
 for method in methods:
  result[method]={}
  for s in ('dev','holdout'):
   per={}
   for d in DATASETS:
    kd=[k for k in keys if k[0]==d and split(k)==s];per[d]=interval_f1({k:y[k] for k in kd},{k:rows[(k[0],k[1],method)] for k in kd})
   result[method][s]={'macro':{m:float(np.mean([per[d][m] for d in DATASETS])) for m in ('interval_F1@0.3','interval_F1@0.5','interval_F1@0.7')},'per_dataset':per}
 a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');best=sorted(((v['dev']['macro']['interval_F1@0.5'],m) for m,v in result.items()),reverse=True)[:10];print(json.dumps([{'method':m,'dev':result[m]['dev']['macro'],'holdout':result[m]['holdout']['macro']} for _,m in best],indent=2))
if __name__=='__main__':main()
