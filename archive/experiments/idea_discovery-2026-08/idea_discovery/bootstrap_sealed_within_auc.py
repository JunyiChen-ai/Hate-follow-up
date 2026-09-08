#!/usr/bin/env python3
"""Paired video bootstrap for a single sealed cohort's within-video AUC."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
def load(path,method):
 out={}
 for l in Path(path).open():
  r=json.loads(l)
  if r.get('method')==method:out[r['video_id']]=r
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--predictions',type=Path);ap.add_argument('--candidate-predictions',type=Path);ap.add_argument('--baseline-predictions',type=Path);ap.add_argument('--candidate-method',required=True);ap.add_argument('--baseline-method',required=True);ap.add_argument('--gt',type=Path,required=True);ap.add_argument('--samples',type=int,default=20000);ap.add_argument('--seed',type=int,default=20260828);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 cp=a.candidate_predictions or a.predictions;bp=a.baseline_predictions or a.predictions
 if cp is None or bp is None: raise ValueError('provide --predictions or both prediction paths')
 c=load(cp,a.candidate_method);b=load(bp,a.baseline_method);z=np.load(a.gt,allow_pickle=True);y={str(v):np.asarray(z['y4'][i],np.int8) for i,v in enumerate(z['video_ids'])};diff=[]
 for v in sorted(set(y)&set(c)&set(b)):
  n=min(len(y[v]),len(c[v]['score_curve']),len(b[v]['score_curve']));yy=y[v][:n]
  if n and yy.min()!=yy.max():diff.append(roc_auc_score(yy,np.asarray(c[v]['score_curve'][:n]))-roc_auc_score(yy,np.asarray(b[v]['score_curve'][:n])))
 diff=np.asarray(diff);rng=np.random.default_rng(a.seed);boot=np.asarray([np.mean(diff[rng.integers(0,len(diff),len(diff))]) for _ in range(a.samples)]);result={'candidate':a.candidate_method,'baseline':a.baseline_method,'n_defined':len(diff),'difference':float(diff.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'p_difference_le_zero':float(np.mean(boot<=0)),'positive_videos':int(np.sum(diff>0)),'negative_videos':int(np.sum(diff<0)),'ties':int(np.sum(diff==0)),'samples':a.samples,'seed':a.seed};a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
