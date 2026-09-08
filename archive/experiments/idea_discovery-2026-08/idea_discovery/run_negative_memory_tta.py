#!/usr/bin/env python3
"""Cross-video hard-negative memory atop the per-video ordinal visual scorer."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,torch,open_clip
ROOT=Path(__file__).resolve().parents[2];RETR=Path('/home/jehc223/Retrieval-hate');GT=RETR/'data/gt/frame_gt_4fps';FEAT=RETR/'data/CLIP_Embedding';BASE=ROOT/'results/idea_discovery/prompt_query/abuse'
ASR={'HateMM':ROOT/'results/hatemm_localization/per_chunk.jsonl','HateClipSeg':ROOT/'results/reproduction/ours/hateclipseg/per_chunk.jsonl','MHC':ROOT/'results/reproduction/ours/mhclip_en/per_chunk.jsonl','MHC_zh':ROOT/'results/reproduction/ours/mhclip_zh/per_chunk.jsonl'}
def load(p):
 d={}
 for x in p.read_text().splitlines():r=json.loads(x);d.setdefault(r['video_id'],[]).append(r)
 return d
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--lambda-neg',type=float,default=.05);ap.add_argument('--memory-size',type=int,default=64);ap.add_argument('--order-seed',type=int,default=-1);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);device='cuda';model,_,_=open_clip.create_model_and_transforms('coca_ViT-L-14',pretrained='mscoco_finetuned_laion2B-s13B-b90k');P=model.to(device).eval().visual.proj.detach();meta=[]
 for ds,p in ASR.items():
  by=load(p);z=np.load(GT/(ds+'.npz'),allow_pickle=True);gm={str(v):(float(d),str(s)) for v,d,s in zip(z['video_ids'],z['duration'],z['split'])};ids=sorted(v for v in by if v in gm and gm[v][1]=='test' and (BASE/ds/(v+'.npy')).exists())
  if a.order_seed>=0:np.random.default_rng(a.order_seed).shuffle(ids)
  bank=[];od=out/ds;od.mkdir(exist_ok=True)
  for vid in ids:
   F=torch.tensor(np.load(FEAT/ds/'coca_vitL14_4fps'/(vid+'.npy')),device=device)@P;F=F/F.norm(dim=1,keepdim=True).clamp_min(1e-8);base=np.load(BASE/ds/(vid+'.npy'));pen=np.zeros(len(F),np.float32)
   if bank:pen=(F@torch.stack(bank).T).max(1).values.cpu().numpy()
   score=base-a.lambda_neg*pen;np.save(od/(vid+'.npy'),score.astype(np.float32));dur=gm[vid][0];cand=[]
   for r in by[vid]:
    lo=max(0,min(len(F)-1,int(float(r['span'][0])/dur*len(F))));hi=min(len(F),max(lo+1,int(np.ceil(float(r['span'][1])/dur*len(F)))));zv=float(r.get('z_masked',r.get('z_isolated',-20)));cand.append((zv,float(base[lo:hi].mean()),F[lo:hi].mean(0)))
   if cand:
    med=np.median([x[1] for x in cand]);writes=[x[2]/x[2].norm().clamp_min(1e-8) for x in cand if x[0]<=-4 and x[1]>=med];bank=(bank+writes)[-a.memory_size:]
   meta.append({'dataset':ds,'video_id':vid,'memory_len':len(bank),'writes':len(writes) if cand else 0})
 (out/'run.json').write_text(json.dumps({'config':vars(a),'videos':meta},indent=2)+'\n');print(json.dumps({'n':len(meta),'writes':sum(x['writes'] for x in meta)},indent=2))
if __name__=='__main__':main()
