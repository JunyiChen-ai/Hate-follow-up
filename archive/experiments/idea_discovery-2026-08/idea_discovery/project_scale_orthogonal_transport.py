#!/usr/bin/env python3
"""Orthogonally separate video propensity from within-video temporal ranking."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
def load(p,m):return {(r['dataset'],r['video_id']):r for r in map(json.loads,Path(p).open()) if r['method']==m}
def logit(x):x=np.clip(np.asarray(x,float),1e-5,1-1e-5);return np.log(x/(1-x))
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--hard',type=Path,required=True);p.add_argument('--soft',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 b=load(a.base,'fact_less_t3al_dualgeo_midpoint_v5');h=load(a.hard,'pos_less_aligned_v1');s=load(a.soft,'transport_pos_less_aligned_v1');audit={'n':0,'changed':0}
 with a.out.open('w') as f:
  for k,r in sorted(b.items()):
   z0=logit(r['score_curve']);dh=logit(h[k]['score_curve'])-z0;ds=logit(s[k]['score_curve'])-z0;agree=dh*ds>1e-16;dg=np.sign(dh)*np.sqrt(np.abs(dh*ds))*agree
   # Hard evidence owns the centered temporal residual. Multi-support
   # consensus owns only the video-level propensity (a constant shift).
   d=(dh-float(np.mean(dh)))+float(np.mean(dg));z=z0+d;post=1/(1+np.exp(-np.clip(z,-30,30)));changed=bool(np.max(np.abs(post-1/(1+np.exp(-np.clip(z0,-30,30)))))>1e-8);audit['n']+=1;audit['changed']+=int(changed)
   out=dict(r);out['method']='scale_orthogonal_transport_v1';out['score_curve']=post.tolist();out['raw']={**r.get('raw',{}),'gt_access':False,'temporal_residual':'centered_hard_pos_projection','video_propensity':'mean_hard_soft_geometric_consensus','orthogonal_by_construction':True,'exact_fallback':not changed}
   f.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
