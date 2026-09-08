#!/usr/bin/env python3
"""GT-isolated ceiling for fixed-total-length left/right extent allocation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.label_free_adapt.evaluate import binary_intervals,interval_f1,temporal_iou

ALLOC=(0.,.25,.5,.75,1.)
def allocate(row,a,factor):
 p=row['proposals'][0];s,e,d=float(p['start']),float(p['end']),float(row['duration']);delta=(e-s)*(factor-1);x=max(0.,s-a*delta);y=min(d,e+(1-a)*delta)
 # Preserve total length by shifting at video boundaries whenever possible.
 target=min(d,(e-s)*factor)
 if y-x<target:
  if x<=1e-9:y=min(d,x+target)
  elif y>=d-1e-9:x=max(0.,y-target)
 return [x,y,1.]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--proposals',type=Path,required=True);ap.add_argument('--gt-dir',type=Path,required=True);ap.add_argument('--factor',type=float,default=1.4899859333664718);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();rows={(r['dataset'],r['video_id']):r for r in map(json.loads,a.proposals.open())};y={}
 for d in sorted({k[0] for k in rows}):
  z=np.load(a.gt_dir/f'{d}.npz',allow_pickle=True)
  for i,v in enumerate(z['video_ids']):
   if (d,str(v)) in rows:y[(d,str(v))]=np.asarray(z['y4'][i],np.int8)
 keys=sorted(set(rows)&set(y));base={};geometry={};oracle={};detail=[]
 for k in keys:
  targets=binary_intervals(y[k]);options=[allocate(rows[k],x,a.factor) for x in ALLOC]
  q=[max((temporal_iou(tuple(p[:2]),g) for g in targets),default=0.) for p in options];j=max(range(len(q)),key=lambda i:(q[i],ALLOC[i]==.5));base[k]={'intervals':[options[2]]};oracle[k]={'intervals':[options[j]]}
  top=rows[k]['proposals'][0];bank=rows[k]['proposals'][:8];left=max(0.,float(top['start'])-min(float(p['start']) for p in bank));right=max(0.,max(float(p['end']) for p in bank)-float(top['end']));ga=.5 if left+right<=1e-8 else left/(left+right);gidx=min(range(len(ALLOC)),key=lambda i:abs(ALLOC[i]-ga));geometry[k]={'intervals':[options[gidx]]}
  detail.append({'dataset':k[0],'video_id':k[1],'oracle_allocation':ALLOC[j],'geometry_allocation':ALLOC[gidx],'symmetric_iou':q[2],'geometry_iou':q[gidx],'oracle_iou':q[j],'strict_gain':q[j]>q[2]})
 result={'factor':a.factor,'allocations':ALLOC,'per_dataset':{},'detail':detail}
 for d in sorted({k[0] for k in keys}):
  kd=[k for k in keys if k[0]==d];result['per_dataset'][d]={'n':len(kd),'symmetric':interval_f1({k:y[k] for k in kd},{k:base[k] for k in kd}),'geometry':interval_f1({k:y[k] for k in kd},{k:geometry[k] for k in kd}),'oracle':interval_f1({k:y[k] for k in kd},{k:oracle[k] for k in kd}),'strict_gain_videos':sum(x['strict_gain'] for x in detail if x['dataset']==d)}
 result['macro']={}
 for m in ('interval_F1@0.3','interval_F1@0.5','interval_F1@0.7'):
  b=float(np.mean([x['symmetric'][m] for x in result['per_dataset'].values()]));g=float(np.mean([x['geometry'][m] for x in result['per_dataset'].values()]));o=float(np.mean([x['oracle'][m] for x in result['per_dataset'].values()]));result['macro'][m]={'symmetric':b,'geometry':g,'geometry_delta':g-b,'oracle':o,'delta':o-b}
 a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'macro':result['macro'],'strict_gain_by_dataset':{d:x['strict_gain_videos'] for d,x in result['per_dataset'].items()}},indent=2))
if __name__=='__main__':main()
