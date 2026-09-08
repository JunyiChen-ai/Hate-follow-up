#!/usr/bin/env python3
"""Per-video universal-threshold Haar shrinkage for piecewise events."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from scripts.idea_discovery.project_role_orthogonal_transport import load
from scripts.idea_discovery.project_trace_gcv import reconstruct

NORMAL_ABS_MEDIAN=0.6744897501960817

def forward(x):
 levels=[];current=np.asarray(x,float)
 while len(current)>=2:
  approx=(current[0::2]+current[1::2])/np.sqrt(2);detail=(current[0::2]-current[1::2])/np.sqrt(2)
  levels.append(detail);current=approx
 return current,levels
def inverse(approx,levels):
 current=approx
 for detail in reversed(levels):
  x=np.empty(2*len(current));x[0::2]=(current+detail)/np.sqrt(2);x[1::2]=(current-detail)/np.sqrt(2);current=x
 return current
def denoise(signal):
 n=len(signal)
 if n<4:return signal.copy(),{'fallback':True}
 size=1<<(n-1).bit_length();pad=np.pad(signal,(0,size-n),mode='reflect') if size>n else signal.copy()
 approx,details=forward(pad);sigma=float(np.median(np.abs(details[0]-np.median(details[0])))/NORMAL_ABS_MEDIAN)
 threshold=float(sigma*np.sqrt(2*np.log(n)))
 shrunk=[np.sign(d)*np.maximum(np.abs(d)-threshold,0) for d in details]
 fitted=inverse(approx,shrunk)[:n];fitted-=fitted.mean()-signal.mean()
 return fitted,{'fallback':False,'noise_sigma':sigma,'universal_threshold':threshold,'padded_length':size,'wavelet':'haar','threshold_rule':'Donoho_Johnstone_universal_soft'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--base-method',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 rows=load(a.base,a.base_method)
 with a.out.open('w') as h:
  for _,row in sorted(rows.items()):
   scores=np.asarray(row['score_curve'],float);c=np.clip(scores,1e-6,1-1e-6);logits=np.log(c/(1-c));signal=logits-logits.mean();fit,audit=denoise(signal)
   out=dict(row);out['method']='universal_haar_shrinkage_v1';out['score_curve']=reconstruct(scores,fit).tolist();out['raw']={**out.get('raw',{}),'gt_access':False,'module':'per_video_universal_wavelet_shrinkage','dataset_parameters':0,'label_selected_parameters':0,'video_mean_preserved':True,'intervals_preserved':True,'wavelet_audit':audit};h.write(json.dumps(out,separators=(',',':'))+'\n')
if __name__=='__main__':main()
