#!/usr/bin/env python3
"""Hash-split multimodal change-point snapping of extent-corrected A10 boundaries."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter1d
from scripts.label_free_adapt.evaluate import interval_f1
from scripts.idea_discovery.multimodal_boundary_pilot import ASR,FEAT,GT,audio_salience,media,norm

DATASETS=('HateMM','HateClipSeg','MHC','MHC_zh')
def hs(k):return 'dev' if int(hashlib.sha256(f'mm-snap-v1\0{k[0]}\0{k[1]}'.encode()).hexdigest(),16)%2==0 else 'holdout'
def interp(x,n):return np.interp((np.arange(n)+.5)/n,(np.arange(len(x))+.5)/len(x),x)
def closure(row,k=8):
 ps=sorted((float(p['start']),float(p['end'])) for p in row['proposals'][:k]);groups=[]
 for a,b in ps:
  if groups and a<=groups[-1][1]:groups[-1][1]=max(groups[-1][1],b)
  else:groups.append([a,b])
 c=.5*(row['proposals'][0]['start']+row['proposals'][0]['end']);return next((x for x in groups if x[0]<=c<=x[1]),groups[0])
def snap(interval,sal,duration,radius,quantile):
 out=[];cut=np.quantile(sal,quantile)
 for side,boundary in enumerate(interval):
  center=int(round(boundary/duration*len(sal)));r=max(1,int(radius/duration*len(sal)));lo=max(1,center-r);hi=min(len(sal)-1,center+r+1)
  if lo>=hi:out.append(boundary);continue
  local=lo+int(np.argmax(sal[lo:hi]));value=(local+.5)/len(sal)*duration
  out.append(value if sal[local]>=cut else boundary)
 if out[0]>=out[1]:return interval
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--proposals',type=Path,required=True);ap.add_argument('--gt-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();rows={(r['dataset'],r['video_id']):r for r in map(json.loads,a.proposals.open())};asr={}
 for d,p in ASR.items():
  q={}
  for line in p.read_text().splitlines():
   r=json.loads(line);q.setdefault(r['video_id'],[]).append(r)
  asr[d]=q
 y={};salience={};usable=[]
 for d in DATASETS:
  z=np.load(a.gt_dir/f'{d}.npz',allow_pickle=True);gm={str(v):np.asarray(z['y4'][i],np.int8) for i,v in enumerate(z['video_ids'])}
  for k,row in [(k,r) for k,r in rows.items() if k[0]==d]:
   v=k[1];feature=FEAT/d/'coca_vitL14_4fps'/f'{v}.npy'
   if v not in gm or v not in asr[d] or not feature.exists():continue
   duration=float(row['duration']);n=max(8,int(duration*4));X=np.load(feature);X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8);visual=interp(norm(np.r_[0,1-np.sum(X[1:]*X[:-1],axis=1)]),n);audio=norm(audio_salience(media(d,v),n));text=np.zeros(n);last=None
   for chunk in asr[d][v]:
    i=min(n-1,max(0,int(float(chunk['span'][0])/duration*n)));value=float(chunk.get('z_masked',chunk.get('z_isolated',-20)));text[i]=0 if last is None else abs(value-last);last=value
   text=norm(text);salience[k]=(visual,audio,text);y[k]=gm[v];usable.append(k)
 configs=[]
 for fusion in ('mean','second','async'):
  for lag in (1,2,4):
   for radius in (2,5,10,20):
    for quantile in (.5,.7,.8,.9,.95):configs.append((fusion,lag,radius,quantile))
 parts={s:[k for k in usable if hs(k)==s] for s in ('dev','holdout')}
 def predictions(config,keys):
  fusion,lag,radius,q=config;out={}
  for k in keys:
   streams=salience[k]
   if fusion=='mean':s=np.mean(streams,axis=0)
   elif fusion=='second':s=np.sort(np.stack(streams),axis=0)[-2]
   else:
    size=2*lag*4+1;scores=[]
    for i,x in enumerate(streams):
     support=np.maximum.reduce([maximum_filter1d(z,size=size,mode='nearest') for j,z in enumerate(streams) if j!=i]);scores.append(2*x*support/(x+support+1e-8))
    s=np.maximum.reduce(scores)
   interval=snap(closure(rows[k]),s,float(rows[k]['duration']),radius,q);out[k]=[interval[0],interval[1],1.]
  return out
 def metric(keys,p):
  per={}
  for d in DATASETS:
   kd=[k for k in keys if k[0]==d];per[d]=interval_f1({k:y[k] for k in kd},{k:{'intervals':[p[k]]} for k in kd})
  return {m:float(np.mean([per[d][m] for d in DATASETS])) for m in ('interval_F1@0.3','interval_F1@0.5','interval_F1@0.7')},per
 ranked=[]
 for c in configs:
  m,_=metric(parts['dev'],predictions(c,parts['dev']));ranked.append((m['interval_F1@0.7'],m['interval_F1@0.5'],c,m))
 ranked.sort(reverse=True,key=lambda x:(x[0],x[1]));best=ranked[0][2];result={'n':len(usable),'split_sizes':{s:len(x) for s,x in parts.items()},'best':best,'dev_top10':[{'config':x[2],'metrics':x[3]} for x in ranked[:10]],'evaluation':{}}
 for s,keys in parts.items():
  p=predictions(best,keys);m,per=metric(keys,p);base={k:[*closure(rows[k]),1.] for k in keys};bm,bper=metric(keys,base);result['evaluation'][s]={'snapped':m,'closure':bm,'delta':{x:m[x]-bm[x] for x in m},'per_dataset_snapped':per,'per_dataset_closure':bper,'changed':sum(p[k][:2]!=base[k][:2] for k in keys)}
 a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
