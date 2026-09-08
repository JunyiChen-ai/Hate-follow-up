#!/usr/bin/env python3
"""Parameter-free consensus of hard-span and soft-transport PoS corrections."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
def load(path,method):
 return {(r['dataset'],r['video_id']):r for r in map(json.loads,Path(path).open()) if r['method']==method}
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--hard',type=Path,required=True);p.add_argument('--soft',type=Path,required=True);p.add_argument('--hard-method',default='pos_less_aligned_v1');p.add_argument('--soft-method',default='transport_pos_less_aligned_v1');p.add_argument('--name',default='aligned');p.add_argument('--rule',choices=('geometric','minimum','average'),required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 base=load(a.base,'fact_less_t3al_dualgeo_midpoint_v5');hard=load(a.hard,a.hard_method);soft=load(a.soft,a.soft_method);audit={'n':0,'changed':0,'agreement_frames':0,'conflict_frames':0}
 with a.out.open('w') as f:
  for key,b in sorted(base.items()):
   p0=np.clip(np.asarray(b['score_curve'],float),1e-5,1-1e-5);z0=np.log(p0/(1-p0));h=hard.get(key);s=soft.get(key)
   if h is None or s is None:post=p0.copy();agree=np.zeros(len(p0),bool);conflict=agree.copy()
   else:
    hp=np.clip(np.asarray(h['score_curve'],float),1e-5,1-1e-5);sp=np.clip(np.asarray(s['score_curve'],float),1e-5,1-1e-5);dh=np.log(hp/(1-hp))-z0;ds=np.log(sp/(1-sp))-z0;agree=dh*ds>1e-16;conflict=dh*ds< -1e-16
    if a.rule=='geometric':d=np.sign(dh)*np.sqrt(np.abs(dh*ds))*agree
    elif a.rule=='minimum':d=np.sign(dh)*np.minimum(np.abs(dh),np.abs(ds))*agree
    else:d=.5*(dh+ds)
    post=1/(1+np.exp(-np.clip(z0+d,-30,30)))
   changed=bool(np.max(np.abs(post-p0))>1e-8);audit['n']+=1;audit['changed']+=int(changed);audit['agreement_frames']+=int(agree.sum());audit['conflict_frames']+=int(conflict.sum())
   out=dict(b);out['method']=f'multiscale_transport_consensus_{a.name}_{a.rule}_v1';out['score_curve']=post.tolist();out['raw']={**b.get('raw',{}),'gt_access':False,'hard_view':a.hard_method,'soft_view':a.soft_method,'consensus_rule':a.rule,'control_name':a.name,'noncompensatory':True,'exact_fallback':not changed}
   f.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
