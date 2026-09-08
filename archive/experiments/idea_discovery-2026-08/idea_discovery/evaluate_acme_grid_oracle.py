#!/usr/bin/env python3
"""Exact-grid GT-isolated oracle for ACME-CP qualification."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.label_free_adapt.evaluate import binary_intervals,interval_f1,temporal_iou
FACTOR=1.4899859333664718;OFFSETS=(-2,-1,0,1,2)
def base(row):
 p=row['proposals'][0];s,e,d=float(p['start']),float(p['end']),float(row['duration']);target=min(d,(e-s)*FACTOR);c=.5*(s+e);a=c-target/2;b=c+target/2
 if a<0:b-=a;a=0.
 if b>d:a-=b-d;b=d
 return max(0.,a),min(d,b),s,e
def options(row):
 a,b,s,e=base(row);w=(b-a)/16;out=[]
 for js in OFFSETS:
  for je in OFFSETS:
   x=max(0.,min(float(row['duration']),a+js*w));y=max(0.,min(float(row['duration']),b+je*w))
   if x<y and x<=s and y>=e:out.append((x,y,js,je))
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--proposals',type=Path,required=True);ap.add_argument('--gt-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();rows={(r['dataset'],r['video_id']):r for r in map(json.loads,a.proposals.open())};y={}
 for d in sorted({k[0] for k in rows}):
  z=np.load(a.gt_dir/f'{d}.npz',allow_pickle=True)
  for i,v in enumerate(z['video_ids']):
   if (d,str(v)) in rows:y[(d,str(v))]=np.asarray(z['y4'][i],np.int8)
 keys=sorted(set(rows)&set(y));bp={};op={};detail=[]
 for k in keys:
  targets=binary_intervals(y[k]);opts=options(rows[k]);q=[max((temporal_iou(x[:2],g) for g in targets),default=0.) for x in opts];zero=next(i for i,x in enumerate(opts) if x[2:]==(0,0));best=max(range(len(q)),key=lambda i:(q[i],i==zero));bp[k]={'intervals':[[*opts[zero][:2],1.]]};op[k]={'intervals':[[*opts[best][:2],1.]]};detail.append({'dataset':k[0],'video_id':k[1],'base_iou':q[zero],'oracle_iou':q[best],'js':opts[best][2],'je':opts[best][3],'strict_gain':q[best]>q[zero]})
 result={'factor':FACTOR,'offsets':OFFSETS,'per_dataset':{},'detail':detail}
 for d in sorted({k[0] for k in keys}):
  kd=[k for k in keys if k[0]==d];result['per_dataset'][d]={'n':len(kd),'base':interval_f1({k:y[k] for k in kd},{k:bp[k] for k in kd}),'oracle':interval_f1({k:y[k] for k in kd},{k:op[k] for k in kd}),'strict_gain':sum(x['strict_gain'] for x in detail if x['dataset']==d),'start_only':sum(x['strict_gain'] and x['js']!=0 and x['je']==0 for x in detail if x['dataset']==d),'end_only':sum(x['strict_gain'] and x['js']==0 and x['je']!=0 for x in detail if x['dataset']==d)}
 result['macro']={}
 for m in ('interval_F1@0.3','interval_F1@0.5','interval_F1@0.7'):
  b=float(np.mean([x['base'][m] for x in result['per_dataset'].values()]));o=float(np.mean([x['oracle'][m] for x in result['per_dataset'].values()]));result['macro'][m]={'base':b,'oracle':o,'delta':o-b}
 a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'macro':result['macro'],'datasets':{d:{'delta@.5':x['oracle']['interval_F1@0.5']-x['base']['interval_F1@0.5'],'delta@.7':x['oracle']['interval_F1@0.7']-x['base']['interval_F1@0.7'],'strict_gain':x['strict_gain']} for d,x in result['per_dataset'].items()}},indent=2))
if __name__=='__main__':main()
