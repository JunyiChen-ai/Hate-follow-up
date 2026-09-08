#!/usr/bin/env python3
"""P4: label-free distributional boundary localization from P2 evidence."""
from __future__ import annotations
import argparse,hashlib,json,sys,zlib
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_counterfactual_evidence import rows,transcript_rows,poset_curve,ecdf
from scripts.idea_discovery.run_visual_temporal_canvas import dense_bins,valid_chunks,bin_means
from scripts.label_free_adapt.schema import Prediction,append_jsonl,curve_to_intervals

COHORT=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_8_clean.jsonl';P2=ROOT/'results/idea_discovery/paradigm_adapt/p2_stage_a_8.jsonl'
T3AL=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_t3al_clean';TOPO=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_topology_clean'

def boundary_posterior(base,scores,seed,samples=512,concentration=12.,threshold=.65):
 b=bin_means(ecdf(base),16);s=np.asarray(scores,float)/100;mean=np.clip(.5*b+.5*s,.01,.99)
 rng=np.random.default_rng(seed);draw=rng.beta(1+concentration*mean,1+concentration*(1-mean),size=(samples,16))
 # A sampled event is a temporally coherent connected component, not independent frame votes.
 active=draw>=threshold
 for k in range(samples):
  x=active[k]
  # bridge one-bin holes, then reject singleton components as boundary noise
  x[1:-1]|=x[:-2]&x[2:]
  padded=np.r_[False,x,False];starts=np.flatnonzero(~padded[:-1]&padded[1:]);ends=np.flatnonzero(padded[:-1]&~padded[1:])
  for a,b1 in zip(starts,ends):
   if b1-a<2:x[a:b1]=False
  active[k]=x
 return active.mean(0),mean

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def canonical_curve(curve,n):
 x=np.asarray(curve,float)
 if len(x)==n:return x
 return np.interp(np.arange(n)/4,np.arange(len(x))/4,x)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cohort',type=Path,default=COHORT);ap.add_argument('--p2',type=Path,default=P2)
 ap.add_argument('--t3al-curves',type=Path,default=T3AL);ap.add_argument('--topology-curves',type=Path,default=TOPO);ap.add_argument('--samples',type=int,default=512);ap.add_argument('--seed',type=int,default=20260826);a=ap.parse_args()
 cohort=rows(a.cohort);p2_rows=rows(a.p2);p2={(r['dataset'],r['video_id'],r['method']):r for r in p2_rows};chunks=transcript_rows();methods=('t3al_boundary_dist','poset_boundary_dist')
 p2_ids={r.get('raw',{}).get('config_id') for r in p2_rows}
 if len(p2_ids)!=1 or None in p2_ids:raise RuntimeError(f'P2 input must have one config, got {p2_ids}')
 config={'version':'p4_boundary_dist_v1','cohort':str(a.cohort.resolve()),'p2':str(a.p2.resolve()),'t3al':str(a.t3al_curves.resolve()),'poset':str(a.topology_curves.resolve()),
  'samples':a.samples,'seed':a.seed,'code_sha256':digest(__file__),'utility_sha256':digest(ROOT/'scripts/idea_discovery/run_visual_temporal_canvas.py'),
  'cohort_sha256':digest(a.cohort),'p2_sha256':digest(a.p2),'p2_config_id':next(iter(p2_ids)),'t3al_run_sha256':digest(a.t3al_curves/'run.json'),'poset_run_sha256':digest(a.topology_curves/'run.json')}
 cid=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16];prior=rows(a.out) if a.out.exists() else []
 if any(r.get('raw',{}).get('config_id')!=cid for r in prior):raise RuntimeError('output config mismatch; use a new --out')
 done={(r['dataset'],r['video_id'],r['method']) for r in prior};expected={(r['dataset'],r['video_id'],m) for r in cohort for m in methods}
 for row in cohort:
  d,v,dur=row['dataset'],row['video_id'],float(row['duration']);clean,_=valid_chunks(chunks.get((d,v),[]),dur)
  n=max(1,int(np.floor(dur*4)));t=canonical_curve(np.load(a.t3al_curves/d/f'{v}.npy'),n);q=canonical_curve(np.load(a.topology_curves/d/f'{v}.npy'),n);p=poset_curve(q,clean,dur)
  for name,base,p2method in [('t3al',t,'t3al_canvas'),('poset',p,'poset_canvas')]:
   method=f'{name}_boundary_dist'
   if (d,v,method) in done:continue
   src=p2.get((d,v,p2method));scores=src.get('modality_evidence',{}).get('bin_scores',[]) if src else []
   if len(scores)!=16:raise RuntimeError(f'missing/invalid P2 scores: {d}/{v}/{p2method}')
   vseed=(a.seed^zlib.crc32(f'{d}/{v}/{name}'.encode()))&0xffffffff;post,mean=boundary_posterior(base,scores,vseed,a.samples)
   ids=np.clip((np.arange(n)/4/dur*16).astype(int),0,15);posterior=post[ids]
   curve=.75*ecdf(base)+.25*posterior;intervals=curve_to_intervals(posterior,dur,.5)
   if len(posterior)!=n or len(curve)!=n:raise RuntimeError(f'non-canonical output grid: {d}/{v}/{method}')
   pred=Prediction(method,d,v,dur,score_curve=curve.tolist(),intervals=intervals,calls=int(src.get('calls',1)),seed=a.seed,
    modality_evidence={'bin_observation_mean':mean.tolist(),'bin_active_posterior':post.tolist(),'samples':a.samples},raw={'config_id':cid,'config':config})
   append_jsonl(a.out,pred);print(json.dumps({'dataset':d,'video_id':v,'method':method,'intervals':len(intervals)}),flush=True)
 counts=Counter((r['dataset'],r['video_id'],r['method']) for r in rows(a.out));observed=set(counts)
 if expected-observed or observed-expected or any(counts[k]!=1 for k in expected):raise RuntimeError('coverage/duplicate failure')
 return 0
if __name__=='__main__':raise SystemExit(main())
