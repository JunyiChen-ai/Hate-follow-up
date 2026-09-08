#!/usr/bin/env python3
"""Dataset-balanced paired bootstrap for per-video ROC-AUC deltas."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

DATASETS=('HateMM','HateClipSeg','MHC','MHC_zh')
def load(path,method):return {(r['dataset'],r['video_id']):r for r in map(json.loads,Path(path).open()) if r['method']==method}
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--candidate-method',required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--baseline-method',required=True);p.add_argument('--gt-dir',type=Path,required=True);p.add_argument('--changed-only',action='store_true');p.add_argument('--samples',type=int,default=20000);p.add_argument('--out',type=Path,required=True);a=p.parse_args();c=load(a.candidate,a.candidate_method);b=load(a.baseline,a.baseline_method);groups={}
 for d in DATASETS:
  z=np.load(a.gt_dir/f'{d}.npz',allow_pickle=True);gt={str(v):np.asarray(z['y4'][i],int) for i,v in enumerate(z['video_ids']) if str(z['split'][i])=='test'};vals=[]
  for v in sorted(gt):
   k=(d,v)
   if k not in c or k not in b or len(np.unique(gt[v]))<2:continue
   if a.changed_only and c[k].get('raw',{}).get('exact_fallback',True):continue
   n=min(len(gt[v]),len(c[k]['score_curve']),len(b[k]['score_curve']));y=gt[v][:n];vals.append(float(roc_auc_score(y,c[k]['score_curve'][:n])-roc_auc_score(y,b[k]['score_curve'][:n])))
  groups[d]=np.asarray(vals,float)
 point=float(np.mean([x.mean() for x in groups.values() if len(x)]));rng=np.random.default_rng(20260828);draw=[]
 for _ in range(a.samples):draw.append(float(np.mean([rng.choice(x,len(x),replace=True).mean() for x in groups.values() if len(x)])))
 out={'changed_only':a.changed_only,'n_by_dataset':{d:len(x) for d,x in groups.items()},'difference':point,'ci95':np.quantile(draw,[.025,.975]).tolist(),'p_difference_le_zero':float(np.mean(np.asarray(draw)<=0)),'samples':a.samples};a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
