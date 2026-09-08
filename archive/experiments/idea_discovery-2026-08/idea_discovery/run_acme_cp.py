#!/usr/bin/env python3
"""ACME-CP: anchor-conditioned multimodal extent change points."""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.multimodal_boundary_pilot import ASR
from scripts.label_free_adapt.schema import Interval,Prediction,append_jsonl,intervals_to_curve
FACTOR=1.4899859333664718;OFFSETS=(-2,-1,0,1,2);METHODS=('acme_cp','acme_visual','acme_text','acme_mean','acme_no_agreement','acme_base')
def ecdf(x):return (rankdata(np.asarray(x,float),method='average')-1)/max(len(x)-1,1)
def base(row):
 p=row['proposals'][0];s,e,d=float(p['start']),float(p['end']),float(row['duration']);target=min(d,(e-s)*FACTOR);c=.5*(s+e);a=c-target/2;b=c+target/2
 if a<0:b-=a;a=0.
 if b>d:a-=b-d;b=d
 return max(0.,a),min(d,b),s,e
def visual_field(row,n=128):
 cells=[[] for _ in range(n)]
 for p in row['proposals'][:8]:
  a=max(0,int(math.floor(float(p['start'])/row['duration']*n)));b=min(n,int(math.ceil(float(p['end'])/row['duration']*n)));value=float(p['logit'])-math.log(max(1,b-a))
  for j in range(a,b):cells[j].append(value)
 finite=[logsumexp(x) for x in cells if x];floor=min(finite)-1 if finite else -20.;return ecdf([logsumexp(x) if x else floor for x in cells])
def text_field(row,chunks,n=128):
 values=np.full(n,np.nan);counts=np.zeros(n)
 for c in chunks:
  a,b=map(float,c['span']);i=max(0,int(math.floor(a/row['duration']*n)));j=min(n,int(math.ceil(b/row['duration']*n)));z=float(c.get('z_masked',c.get('z_isolated',-20.)))
  if j>i:
   old=np.nan_to_num(values[i:j],nan=0.);old+=z;values[i:j]=old;counts[i:j]+=1
 valid=counts>0
 if not np.any(valid):return None
 values[valid]/=counts[valid];values[~valid]=float(np.min(values[valid])-1);return ecdf(values)
def mean_region(field,duration,a,b):
 n=len(field);i=max(0,int(math.floor(a/duration*n)));j=min(n,int(math.ceil(b/duration*n)));return float(np.mean(field[i:j])) if j>i else -1.
def candidates(field,duration,boundary,w,side,original):
 out=[]
 for off in OFFSETS:
  b=max(0.,min(duration,boundary+off*w));valid=b<=original if side=='start' else b>=original
  score=(mean_region(field,duration,b,b+w)-mean_region(field,duration,b-w,b) if side=='start' else mean_region(field,duration,b-w,b)-mean_region(field,duration,b,b+w))
  out.append((b,score,valid))
 return out
def decide(v,t,duration,boundary,w,side,original,mode):
 cv=candidates(v,duration,boundary,w,side,original);ct=candidates(t,duration,boundary,w,side,original) if t is not None else None;valid=np.array([x[2] for x in cv]);vs=np.array([x[1] if x[2] else -1e9 for x in cv]);ts=np.array([x[1] if x[2] else -1e9 for x in ct]) if ct else None;rv=ecdf(vs);rt=ecdf(ts) if ts is not None else None
 if mode=='base' or (t is None and mode not in ('visual',)):idx=2;gate=False
 elif mode=='visual':idx=int(np.argmax(vs));gate=vs[idx]>vs[2]
 elif mode=='text':idx=int(np.argmax(ts));gate=ts[idx]>ts[2]
 elif mode=='mean':idx=int(np.argmax((rv+rt)/2));gate=vs[idx]>vs[2] and ts[idx]>ts[2]
 elif mode=='no_agreement':idx=int(np.argmax(np.minimum(rv,rt)));gate=vs[idx]>vs[2] and ts[idx]>ts[2]
 else:
  iv,it=int(np.argmax(vs)),int(np.argmax(ts));idx=int(np.argmax(np.minimum(rv,rt)));gate=abs(iv-it)<=1 and vs[idx]>vs[2] and ts[idx]>ts[2]
 if not gate:idx=2
 return cv[idx][0],{'offset':int(OFFSETS[idx]),'gate':bool(gate),'visual_scores':vs.tolist(),'text_scores':None if ts is None else ts.tolist(),'visual_argmax':int(np.argmax(vs)),'text_argmax':None if ts is None else int(np.argmax(ts))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--proposals',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 rows={(r['dataset'],r['video_id']):r for r in map(json.loads,a.proposals.open())};chunks={}
 for d,p in ASR.items():
  q={}
  for line in p.read_text().splitlines():
   r=json.loads(line);q.setdefault(r['video_id'],[]).append(r)
  chunks[d]=q
 for key,row in rows.items():
  if key[1] not in chunks.get(key[0],{}):continue
  a0,b0,s,e=base(row);w=(b0-a0)/16;v=visual_field(row);t=text_field(row,chunks[key[0]][key[1]])
  for method in METHODS:
   mode=method.removeprefix('acme_');x,ms=decide(v,t,row['duration'],a0,w,'start',s,mode);y,me=decide(v,t,row['duration'],b0,w,'end',e,mode)
   if x>=y or x>s or y<e:x,y=a0,b0;ms['gate']=me['gate']=False
   iv=Interval(x,y,1.);append_jsonl(a.out,Prediction(method,key[0],key[1],float(row['duration']),score_curve=intervals_to_curve([iv],float(row['duration'])),intervals=[iv],calls=0,modality_evidence={'start':ms,'end':me,'text_available':t is not None,'window_seconds':w},raw={'gt_access':False,'factor':FACTOR,'visual_field':'proposal_mass_lse_length_normalized','text_field':'timestamp_policy_logodds_ecdf'}))
 print(json.dumps({'videos':len({(r['dataset'],r['video_id']) for r in map(json.loads,a.out.open())}),'methods':METHODS}))
if __name__=='__main__':main()
