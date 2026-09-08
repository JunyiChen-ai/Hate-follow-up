#!/usr/bin/env python3
"""Per-video, per-scale SURE-selected Haar shrinkage."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.idea_discovery.project_role_orthogonal_transport import load
from scripts.idea_discovery.project_trace_gcv import reconstruct
from scripts.idea_discovery.project_universal_haar import NORMAL_ABS_MEDIAN,forward,inverse

def sure_threshold(detail,sigma):
 n=len(detail)
 if n==0 or sigma<=np.finfo(float).eps:return 0.
 y=np.abs(detail/sigma);candidates=np.unique(np.r_[0.,y]);cap=np.sqrt(2*np.log(max(2,n)));candidates=candidates[candidates<=cap];candidates=np.unique(np.r_[candidates,cap])
 risks=[]
 for t in candidates:
  risks.append(n-2*np.sum(y<=t)+np.sum(np.minimum(y*y,t*t)))
 return float(candidates[int(np.argmin(risks))]*sigma)
def denoise(signal):
 n=len(signal)
 if n<4:return signal.copy(),{'fallback':True}
 size=1<<(n-1).bit_length();pad=np.pad(signal,(0,size-n),mode='reflect') if size>n else signal.copy();approx,details=forward(pad)
 sigma=float(np.median(np.abs(details[0]-np.median(details[0])))/NORMAL_ABS_MEDIAN);thresholds=[sure_threshold(d,sigma) for d in details];shrunk=[np.sign(d)*np.maximum(np.abs(d)-t,0) for d,t in zip(details,thresholds)];fit=inverse(approx,shrunk)[:n];fit-=fit.mean()-signal.mean()
 return fit,{'fallback':False,'noise_sigma':sigma,'scale_thresholds':thresholds,'padded_length':size,'wavelet':'haar','threshold_rule':'per_scale_Stein_unbiased_risk_capped_by_universal'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--base-method',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 rows=load(a.base,a.base_method)
 with a.out.open('w') as h:
  for _,row in sorted(rows.items()):
   scores=np.asarray(row['score_curve'],float);c=np.clip(scores,1e-6,1-1e-6);logits=np.log(c/(1-c));fit,audit=denoise(logits-logits.mean());out=dict(row);out['method']='sureshrink_haar_v1';out['score_curve']=reconstruct(scores,fit).tolist();out['raw']={**out.get('raw',{}),'gt_access':False,'module':'per_video_multiscale_sure_shrinkage','dataset_parameters':0,'label_selected_parameters':0,'video_mean_preserved':True,'intervals_preserved':True,'wavelet_audit':audit};h.write(json.dumps(out,separators=(',',':'))+'\n')
if __name__=='__main__':main()
