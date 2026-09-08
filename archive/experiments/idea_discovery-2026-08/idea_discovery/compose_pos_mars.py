#!/usr/bin/env python3
"""Compose empirical PoS dense ranking with a frozen MARS boundary decision."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--pos',type=Path,required=True);p.add_argument('--pos-method',default='pos_less_appellate_aligned_v1');p.add_argument('--mars',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 m={(r['dataset'],r['video_id']):r for r in map(json.loads,a.mars.open())};n=changed=0
 with a.out.open('w') as f:
  for r in map(json.loads,a.pos.open()):
   if r['method']!=a.pos_method:continue
   n+=1;k=(r['dataset'],r['video_id']);mr=m.get(k);out=dict(r);out['method']='pos_mars_composed_v1'
   if mr and (mr.get('raw',{}).get('mars',{}).get('changed_interval')):
    out['intervals']=mr['intervals'];changed+=1
   out['raw']={**r.get('raw',{}),'composition':{'dense_authority':'empirical_pos_less','boundary_authority':'frozen_mars','mars_changed':bool(mr and mr.get('raw',{}).get('mars',{}).get('changed_interval')),'gt_access':False}}
   f.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps({'n':n,'changed':changed}))
if __name__=='__main__':main()
