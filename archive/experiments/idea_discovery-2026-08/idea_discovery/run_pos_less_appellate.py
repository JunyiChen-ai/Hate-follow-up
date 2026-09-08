#!/usr/bin/env python3
"""Frozen dataset-agnostic PoS-LESS appellate decoder.

This is the clean method entry point.  It contains no dataset routing and no
absolute significance threshold.  Orbit statistics are used only to compare
frozen proposals within the same video.  Midpoint remains the status quo; only
the native semantic core (T3AL) has boundary-amendment jurisdiction.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

from project_pos_less_tribunal import chunks_by_video,text_curve,mask,contrast,certificate,rank

RULES=("cca","t3al","midpoint","union")

def load(path,method=None):
 out={}
 for r in map(json.loads,Path(path).open()):
  if method is None or r['method']==method:out[(r['dataset'],r['video_id'])]=r
 return out

def winner(candidates):
 vr=rank([x['visual'] for x in candidates],reverse=True)
 pr=rank([x['orbit_p'] for x in candidates])
 sr=rank([x['surplus'] for x in candidates],reverse=True)
 objective=vr+pr+sr;best=np.flatnonzero(objective==objective.min()).tolist()
 midpoint=next((i for i,x in enumerate(candidates) if x['rule']=='midpoint'),0)
 return midpoint if midpoint in best else best[0]

def main():
 p=argparse.ArgumentParser();p.add_argument('--posterior',type=Path,required=True);p.add_argument('--geometry-bank',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 aligned=load(a.posterior,'pos_less_aligned_v1');shifted=load(a.posterior,'pos_less_circular_shift_v1');bank={(r['method'],r['dataset'],r['video_id']):r for r in map(json.loads,a.geometry_bank.open())};chunks=chunks_by_video();audit={}
 with a.out.open('w') as f:
  for control,posts in [('aligned',aligned),('shift',shifted)]:
   for key,row in sorted(posts.items()):
    dataset,video_id=key;curve=np.asarray(row['score_curve'],float);duration=float(row['duration']);n=len(curve);language=text_curve(chunks.get(key,[]),n);has_text=key in chunks
    seed=int.from_bytes(hashlib.sha256(f'appellate/{dataset}/{video_id}'.encode()).digest()[:8],'little');offset=max(1,seed%max(2,n-1))
    if control=='shift':language=np.roll(language,offset)
    offsets=sorted({max(1,int(round(n*q/16))) for q in range(1,16)});candidates=[];seen=set()
    for rule in RULES:
     candidate=bank[(f'fact_less_t3al_dualgeo_{rule}_v5',dataset,video_id)];identity=tuple(tuple(map(float,x[:2])) for x in candidate['intervals'])
     if identity in seen:continue
     seen.add(identity);region=mask(candidate['intervals'],duration,n);v=contrast(curve,region);_,pv,s=certificate(language,region,offsets) if has_text else (0.,1.,0.)
     candidates.append({'rule':rule,'row':candidate,'visual':v,'orbit_p':pv,'surplus':s})
    chosen=candidates[winner(candidates)] if has_text else next((x for x in candidates if x['rule']=='midpoint'),candidates[0])
    # Non-compensatory jurisdiction: competitors may block an appeal, but only
    # T3AL can amend the status quo and only with positive aligned surplus.
    appeal=chosen['rule']=='t3al' and chosen['surplus']>=0
    midpoint=bank[('fact_less_t3al_dualgeo_midpoint_v5',dataset,video_id)]
    out=dict(row);out['method']=f'pos_less_appellate_{control}_v1';out['intervals']=chosen['row']['intervals'] if appeal else midpoint['intervals']
    audit[(control,'appeal')]=audit.get((control,'appeal'),0)+int(appeal)
    out['raw']={**row.get('raw',{}),'decoder':'orbit_relative_rank_sum_appellate','status_quo':'midpoint','boundary_jurisdiction':'t3al_only','winner':chosen['rule'],'appeal_granted':appeal,'orbit_p_descriptive_only':chosen['orbit_p'],'alignment_surplus':chosen['surplus'],'dataset_identity_access':False,'gt_access':False}
    f.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps({'/'.join(k):v for k,v in audit.items()},indent=2))
if __name__=='__main__':main()
