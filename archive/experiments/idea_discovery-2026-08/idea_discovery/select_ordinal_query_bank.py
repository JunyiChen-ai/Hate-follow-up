#!/usr/bin/env python3
"""Select a visual query trajectory using disjoint transcript ordinal pairs."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];GT=Path('/home/jehc223/Retrieval-hate/data/gt/frame_gt_4fps')
ASR={'HateMM':ROOT/'results/hatemm_localization/per_chunk.jsonl','HateClipSeg':ROOT/'results/reproduction/ours/hateclipseg/per_chunk.jsonl','MHC':ROOT/'results/reproduction/ours/mhclip_en/per_chunk.jsonl','MHC_zh':ROOT/'results/reproduction/ours/mhclip_zh/per_chunk.jsonl'}
CANDS={'baseprompt':ROOT/'results/idea_discovery/cpo_tta/chunk50_all','mainq':ROOT/'results/idea_discovery/prompt_query/mainq','hostile':ROOT/'results/idea_discovery/prompt_query/hostile','visual':ROOT/'results/idea_discovery/prompt_query/visual','abuse':ROOT/'results/idea_discovery/prompt_query/abuse'}
def load(p):
 d={}
 for x in p.read_text().splitlines():r=json.loads(x);d.setdefault(r['video_id'],[]).append(r)
 return d
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--fallback',choices=CANDS,default='abuse');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=[]
 for ds,p in ASR.items():
  by=load(p);z=np.load(GT/(ds+'.npz'),allow_pickle=True);gm={str(v):(float(d),str(s)) for v,d,s in zip(z['video_ids'],z['duration'],z['split'])};od=out/ds;od.mkdir(exist_ok=True)
  for vid,rr in sorted(by.items()):
   paths={k:v/ds/(vid+'.npy') for k,v in CANDS.items()}
   if vid not in gm or gm[vid][1]!='test' or not all(x.exists() for x in paths.values()):continue
   curves={k:np.load(x) for k,x in paths.items()};dur=gm[vid][0];b=np.array([1/(1+np.exp(-np.clip(float(r.get('z_masked',r.get('z_isolated',-20)))/4,-30,30))) for r in rr]);held={i for i in range(len(rr)) if i%2==0};pairs=[(i,j) for i in held for j in held if b[i]-b[j]>=.4];loss={}
   for name,c in curves.items():
    cs=[]
    for r in rr:
     lo=max(0,min(len(c)-1,int(float(r['span'][0])/dur*len(c))));hi=min(len(c),max(lo+1,int(math.ceil(float(r['span'][1])/dur*len(c)))));cs.append(float(c[lo:hi].mean()))
    loss[name]=float(np.mean([np.logaddexp(0,-10*(cs[i]-cs[j])) for i,j in pairs])) if pairs else float('inf')
   chosen=min(loss,key=loss.get) if pairs else a.fallback;np.save(od/(vid+'.npy'),curves[chosen].astype(np.float32));meta.append({'dataset':ds,'video_id':vid,'n_pairs':len(pairs),'chosen':chosen,'loss':loss})
 (out/'run.json').write_text(json.dumps({'config':vars(a),'videos':meta},indent=2)+'\n');print(json.dumps({'n':len(meta),'choices':{k:sum(x['chosen']==k for x in meta) for k in CANDS}},indent=2))
if __name__=='__main__':main()
