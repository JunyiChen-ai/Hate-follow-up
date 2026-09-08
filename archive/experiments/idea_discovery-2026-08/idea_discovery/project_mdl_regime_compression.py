#!/usr/bin/env python3
"""Per-video MDL compression of piecewise-constant evidence regimes."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scripts.idea_discovery.project_role_orthogonal_transport import load
from scripts.idea_discovery.project_trace_gcv import reconstruct

NORMAL_ABS_MEDIAN=0.6744897501960817
def runs(x):
 cuts=np.r_[0,np.flatnonzero(np.diff(x)!=0)+1,len(x)];return x[cuts[:-1]],np.diff(cuts),cuts
def robust_noise(levels):
 if len(levels)>=3:
  d=np.diff(levels,n=2);sigma=np.median(np.abs(d-np.median(d)))/(NORMAL_ABS_MEDIAN*np.sqrt(6))
 elif len(levels)>=2:
  d=np.diff(levels);sigma=np.median(np.abs(d-np.median(d)))/(NORMAL_ABS_MEDIAN*np.sqrt(2))
 else:sigma=0.
 return float(sigma*sigma)
def segment(levels,weights,total):
 m=len(levels);noise=robust_noise(levels)
 if m<2 or noise<=np.finfo(float).eps:return levels.copy(),{'fallback':True,'noise_variance':noise}
 penalty=noise*np.log(total);w=np.asarray(weights,float);x=np.asarray(levels,float);cw=np.r_[0,np.cumsum(w)];cs=np.r_[0,np.cumsum(w*x)];cq=np.r_[0,np.cumsum(w*x*x)]
 def cost(i,j):
  ww=cw[j]-cw[i];ss=cs[j]-cs[i];return (cq[j]-cq[i])-ss*ss/ww
 dp=np.full(m+1,np.inf);prev=np.full(m+1,-1,int);dp[0]=-penalty
 for j in range(1,m+1):
  vals=np.asarray([dp[i]+cost(i,j)+penalty for i in range(j)]);i=int(np.argmin(vals));dp[j]=vals[i];prev[j]=i
 bounds=[];j=m
 while j>0:i=int(prev[j]);bounds.append((i,j));j=i
 fitted=np.empty(m)
 for i,j in reversed(bounds):fitted[i:j]=(cs[j]-cs[i])/(cw[j]-cw[i])
 return fitted,{'fallback':False,'noise_variance':noise,'mdl_penalty':penalty,'input_runs':m,'output_regimes':len(bounds),'criterion':'known_noise_gaussian_BIC'}
def perm(levels,key):
 seed=int.from_bytes(hashlib.sha256(('mdl-control/'+key).encode()).digest()[:8],'little');rng=np.random.default_rng(seed);order=rng.permutation(len(levels));return order
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--base-method',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 rows=load(a.base,a.base_method)
 with a.out.open('w') as h:
  for key,row in sorted(rows.items()):
   scores=np.asarray(row['score_curve'],float);c=np.clip(scores,1e-6,1-1e-6);logits=np.log(c/(1-c));levels,weights,cuts=runs(logits)
   variants=[];fit,audit=segment(levels,weights,len(scores));variants.append(('mdl_regime_compression_v1',fit,audit))
   order=perm(levels,key[0]+'/'+key[1]);pfit,paudit=segment(levels[order],weights[order],len(scores));restored=np.empty_like(pfit);restored[order]=pfit;variants.append(('mdl_regime_permuted_control_v1',restored,paudit))
   for method,fitted,record in variants:
    dense=np.repeat(fitted,weights);out=dict(row);out['method']=method;out['score_curve']=reconstruct(scores,dense-dense.mean()).tolist();out['raw']={**out.get('raw',{}),'gt_access':False,'module':'per_video_mdl_temporal_regime_compression','dataset_parameters':0,'label_selected_parameters':0,'video_mean_preserved':True,'intervals_preserved':True,'mdl':record};h.write(json.dumps(out,separators=(',',':'))+'\n')
if __name__=='__main__':main()
