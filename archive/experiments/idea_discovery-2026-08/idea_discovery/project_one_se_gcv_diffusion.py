#!/usr/bin/env python3
"""Per-video one-standard-error GCV temporal diffusion."""
from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from scipy.optimize import brentq,minimize_scalar
from scipy.fft import dct,idct
from scripts.idea_discovery.project_role_orthogonal_transport import load
from scripts.idea_discovery.project_trace_gcv import laplacian_diagonals,reconstruct,smooth


def select(signal):
 n=len(signal);eps=np.finfo(float).eps
 if n<4:return signal.copy(),0.,{"fallback":True}
 edges=np.ones(n-1);diag,off=laplacian_diagonals(edges)
 eigen=eigvalsh_tridiagonal(diag,off,check_finite=False);positive=eigen[eigen>eps]
 coefficients=dct(signal,type=2,norm='ortho')
 lower=np.log(np.sqrt(eps)/positive.max());upper=np.log(1/(np.sqrt(eps)*positive.min()))
 def stats(x):
  lam=float(np.exp(x));fit=idct(coefficients/(1+lam*eigen),type=2,norm='ortho');res=signal-fit
  trace=float(np.sum(1/(1+lam*eigen)));den=max(eps,1-trace/n)
  errors=(res/den)**2;return float(np.mean(errors)),float(np.std(errors,ddof=1)/np.sqrt(n)),fit
 optimum=minimize_scalar(lambda x:stats(x)[0],bounds=(lower,upper),method='bounded',options={'xatol':np.sqrt(eps)})
 risk,se,_=stats(optimum.x);target=risk+se
 def root(x):return stats(x)[0]-target
 # The right-hand crossing is the most regularized model statistically
 # indistinguishable from the empirical minimum.
 if root(upper)<=0:selected_log=upper;crossing='upper_bound'
 else:selected_log=float(brentq(root,optimum.x,upper,xtol=np.sqrt(eps)));crossing='one_se_right_root'
 strength=float(np.exp(selected_log));fit=idct(coefficients/(1+strength*eigen),type=2,norm='ortho')
 return fit,strength,{"fallback":False,"gcv_min_strength":float(np.exp(optimum.x)),
  "gcv_min":risk,"per_video_standard_error":se,"one_se_target":target,
  "strength":strength,"crossing":crossing,"bounds_source":"video_spectrum_machine_precision"}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--base-method',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 rows=load(a.base,a.base_method)
 with a.out.open('w') as h:
  for _,row in sorted(rows.items()):
   scores=np.asarray(row['score_curve'],float);c=np.clip(scores,1e-6,1-1e-6);logits=np.log(c/(1-c));fit,strength,audit=select(logits-logits.mean())
   out=dict(row);out['method']='one_se_gcv_diffusion_v1';out['score_curve']=reconstruct(scores,fit).tolist();out['raw']={**out.get('raw',{}),'gt_access':False,'module':'per_video_one_se_predictive_diffusion','dataset_parameters':0,'label_selected_parameters':0,'video_mean_preserved':True,'intervals_preserved':True,'one_se_gcv':audit};h.write(json.dumps(out,separators=(',',':'))+'\n')
if __name__=='__main__':main()
