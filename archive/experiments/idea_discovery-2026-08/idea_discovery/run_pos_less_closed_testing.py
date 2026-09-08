#!/usr/bin/env python3
"""Closed-testing boundary amendments for label-free PoS-LESS.

Each frozen proposal tests the null that timestamp-language evidence is not
preferentially concentrated inside that proposal. Exact nonzero circular
rotations form the within-video randomization reference. Holm step-down controls
the family-wise false-amendment risk; no rejection means exact midpoint fallback.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from project_pos_less_tribunal import chunks_by_video,text_curve,mask,contrast

RULES=("cca","t3al","midpoint","union")
def load(path,method=None):
 out={}
 for r in map(json.loads,Path(path).open()):
  if method is None or r['method']==method:out[(r['dataset'],r['video_id'])]=r
 return out
def exact_p(language,region):
    observed=contrast(language,region)
    nin=int(region.sum());nout=len(region)-nin
    if nin==0 or nout==0 or len(language)<2:return 1.,observed
    # Circular cross-correlation gives the inside sum at every rotation in
    # O(T log T), making the exhaustive reference practical for long videos.
    inside=np.fft.ifft(np.conj(np.fft.fft(language))*np.fft.fft(region.astype(float))).real
    all_scores=inside/nin-(float(np.sum(language))-inside)/nout
    null=all_scores[1:]
    return float((1+np.sum(null>=observed))/(1+len(null))),observed
def holm(pvalues,alpha):
 order=np.argsort(pvalues,kind='stable');rejected=[];m=len(pvalues)
 for rank,index in enumerate(order):
  if pvalues[index] <= alpha/(m-rank):rejected.append(int(index))
  else:break
 return rejected
def main():
 p=argparse.ArgumentParser();p.add_argument('--posterior',type=Path,required=True);p.add_argument('--geometry-bank',type=Path,required=True);p.add_argument('--alpha',type=float,default=.1);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 aligned=load(a.posterior,'pos_less_aligned_v1');shifted=load(a.posterior,'pos_less_circular_shift_v1');bank={(r['method'],r['dataset'],r['video_id']):r for r in map(json.loads,a.geometry_bank.open())};chunks=chunks_by_video();audit={}
 with a.out.open('w') as f:
  for control,posts in [('aligned',aligned),('shift',shifted)]:
   for key,row in sorted(posts.items()):
    dataset,video_id=key;curve=np.asarray(row['score_curve'],float);n=len(curve);duration=float(row['duration']);language=text_curve(chunks.get(key,[]),n);has_text=key in chunks
    if control=='shift' and has_text:
     seed=int.from_bytes(hashlib.sha256(f'closed/{dataset}/{video_id}'.encode()).digest()[:8],'little');language=np.roll(language,max(1,seed%max(2,n-1)))
    candidates=[];seen=set()
    for rule in RULES:
     proposal=bank[(f'fact_less_t3al_dualgeo_{rule}_v5',dataset,video_id)];identity=tuple(tuple(map(float,x[:2])) for x in proposal['intervals'])
     if identity in seen:continue
     seen.add(identity);region=mask(proposal['intervals'],duration,n);pv,te=exact_p(language,region) if has_text else (1.,0.);candidates.append({'rule':rule,'row':proposal,'p':pv,'text':te,'visual':contrast(curve,region)})
    rejected=holm([x['p'] for x in candidates],a.alpha) if has_text else []
    eligible=[i for i in rejected if candidates[i]['rule']=='t3al' and candidates[i]['visual']>0]
    appeal=bool(eligible);chosen=candidates[eligible[0]] if appeal else next((x for x in candidates if x['rule']=='midpoint'),candidates[0]);mid=bank[('fact_less_t3al_dualgeo_midpoint_v5',dataset,video_id)]
    audit[(control,'rejected_videos')]=audit.get((control,'rejected_videos'),0)+int(bool(rejected));audit[(control,'appeals')]=audit.get((control,'appeals'),0)+int(appeal)
    out=dict(row);out['method']=f'pos_less_closed_{control}_a{int(a.alpha*100):02d}_v1';out['intervals']=chosen['row']['intervals'] if appeal else mid['intervals'];out['raw']={**row.get('raw',{}),'decoder':'holm_closed_testing','fwer_alpha':a.alpha,'randomization_reference':'all_nonzero_circular_rotations','proposal_pvalues':{x['rule']:x['p'] for x in candidates},'rejected_rules':[candidates[i]['rule'] for i in rejected],'appeal_granted':appeal,'status_quo':'midpoint','dataset_identity_access':False,'gt_access':False}
    f.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps({'/'.join(k):v for k,v in audit.items()},indent=2))
if __name__=='__main__':main()
