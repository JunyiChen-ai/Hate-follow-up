#!/usr/bin/env python3
"""Select a visual adaptation trajectory using disjoint speech ordinal checks."""
from __future__ import annotations
import argparse,json,math,zlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
ASR={'HateMM':ROOT/'results/hatemm_localization/per_chunk.jsonl','HateClipSeg':ROOT/'results/reproduction/ours/hateclipseg/per_chunk.jsonl','MHC':ROOT/'results/reproduction/ours/mhclip_en/per_chunk.jsonl','MHC_zh':ROOT/'results/reproduction/ours/mhclip_zh/per_chunk.jsonl'}
GT=Path('/home/jehc223/Retrieval-hate/data/gt/frame_gt_4fps')
CANDS={'fixed':ROOT/'results/idea_discovery/cpo_tta/fixed','t3al':ROOT/'results/idea_discovery/t3al_real/repaired_full2','ordinal':ROOT/'results/idea_discovery/prompt_query/abuse'}
def load(p):
 d={}
 for x in p.read_text().splitlines():r=json.loads(x);d.setdefault(r['video_id'],[]).append(r)
 return d
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--fallback',choices=CANDS,default='t3al');ap.add_argument('--min-improvement',type=float,default=0.);ap.add_argument('--control',choices=['none','shuffle','reverse'],default='none');ap.add_argument('--no-ordinal',action='store_true');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=[]
 for ds,p in ASR.items():
  by=load(p);z=np.load(GT/(ds+'.npz'),allow_pickle=True);gm={str(v):(float(d),str(s)) for v,d,s in zip(z['video_ids'],z['duration'],z['split'])};od=out/ds;od.mkdir(exist_ok=True)
  for vid,rr in sorted(by.items()):
   if vid not in gm or gm[vid][1]!='test':continue
   paths={k:v/ds/(vid+'.npy') for k,v in CANDS.items()}
   if not all(x.exists() for x in paths.values()):continue
   curves={k:np.load(x) for k,x in paths.items()};dur=gm[vid][0];held=[i for i in range(len(rr)) if i%2==0];pairs=[]
   order=sorted(held,key=lambda i:float(rr[i].get('z_masked',rr[i].get('z_isolated',-20))))
   if a.control=='shuffle':order=list(np.random.default_rng(20260825^zlib.crc32(vid.encode())).permutation(order))
   if a.control=='reverse':order=order[::-1]
   k=max(1,len(order)//3)
   if len(order)>=3:pairs=[(i,j) for i in order[-k:] for j in order[:k]]
   losses={}
   for name,c in curves.items():
    cs=[]
    for r in rr:
     lo=max(0,min(len(c)-1,int(float(r['span'][0])/dur*len(c))));hi=min(len(c),max(lo+1,int(math.ceil(float(r['span'][1])/dur*len(c)))));cs.append(float(np.mean(c[lo:hi])))
    losses[name]=float(np.mean([np.logaddexp(0,-10*(cs[i]-cs[j])) for i,j in pairs])) if pairs else float('inf')
   eligible=['fixed','t3al'] if a.no_ordinal else list(CANDS);best=min(eligible,key=lambda x:losses[x]);chosen=best if losses[best]+a.min_improvement<losses[a.fallback] else a.fallback
   np.save(od/(vid+'.npy'),curves[chosen].astype(np.float32));rec={'dataset':ds,'video_id':vid,'n_held_pairs':len(pairs),'losses':losses,'chosen':chosen};meta.append(rec)
 (out/'run.json').write_text(json.dumps({'config':vars(a),'videos':meta},indent=2)+'\n');print(json.dumps({'n':len(meta),'choices':{k:sum(x['chosen']==k for x in meta) for k in CANDS}},indent=2))
if __name__=='__main__':main()
