#!/usr/bin/env python3
"""Cross-modal write-certified episodic query memory for label-free localization."""
from __future__ import annotations
import argparse,json,math,zlib
from pathlib import Path
import numpy as np,torch,open_clip
ROOT=Path(__file__).resolve().parents[2];RETR=Path('/home/jehc223/Retrieval-hate');GT=RETR/'data/gt/frame_gt_4fps';FEAT=RETR/'data/CLIP_Embedding'
ASR={'HateMM':ROOT/'results/hatemm_localization/per_chunk.jsonl','HateClipSeg':ROOT/'results/reproduction/ours/hateclipseg/per_chunk.jsonl','MHC':ROOT/'results/reproduction/ours/mhclip_en/per_chunk.jsonl','MHC_zh':ROOT/'results/reproduction/ours/mhclip_zh/per_chunk.jsonl'};PROMPT='abusive or hateful behavior toward a person or group'
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x]
def pairloss(M,q,pairs):
 s=M@q;return torch.stack([torch.nn.functional.softplus(-10*(s[i]-s[j])) for i,j in pairs]).mean()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--beta',type=float,default=.5);ap.add_argument('--memory-size',type=int,default=16);ap.add_argument('--write-margin',type=float,default=0.);ap.add_argument('--write',choices=['certified','all','none'],default='certified');ap.add_argument('--order-seed',type=int,default=-1);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);device='cuda';model,_,_=open_clip.create_model_and_transforms('coca_ViT-L-14',pretrained='mscoco_finetuned_laion2B-s13B-b90k');model=model.to(device).eval();tok=open_clip.get_tokenizer('coca_ViT-L-14');P=model.visual.proj.detach();q0=model.encode_text(tok(['a video of action '+PROMPT]).to(device)).detach()[0];q0=q0/q0.norm();meta=[]
 for ds,apath in ASR.items():
  by={}
  for r in rows(apath):by.setdefault(r['video_id'],[]).append(r)
  z=np.load(GT/(ds+'.npz'),allow_pickle=True);gm={str(v):(float(d),str(s)) for v,d,s in zip(z['video_ids'],z['duration'],z['split'])};ids=sorted(v for v in by if v in gm and gm[v][1]=='test' and (FEAT/ds/'coca_vitL14_4fps'/(v+'.npy')).exists())
  if a.order_seed>=0:np.random.default_rng(a.order_seed).shuffle(ids)
  memory=[];od=out/ds;od.mkdir(exist_ok=True)
  for vid in ids:
   rr=by[vid];duration=gm[vid][0];X=torch.tensor(np.load(FEAT/ds/'coca_vitL14_4fps'/(vid+'.npy')),device=device)@P;X=X/X.norm(dim=1,keepdim=True).clamp_min(1e-8);M=[];zs=[]
   for r in rr:
    lo=max(0,min(len(X)-1,int(float(r['span'][0])/duration*len(X))));hi=min(len(X),max(lo+1,int(np.ceil(float(r['span'][1])/duration*len(X)))));M.append(X[lo:hi].mean(0));zs.append(float(r.get('z_masked',r.get('z_isolated',-20))))
   M=torch.stack(M);b=1/(1+np.exp(-np.clip(np.asarray(zs)/4,-30,30)));tr=[i for i in range(len(rr)) if i%2];va=[i for i in range(len(rr)) if not i%2]
   def pairs(nodes):
    ns=set(nodes);return [(i,j) for i in ns for j in ns if b[i]-b[j]>=.4]
   train,held=pairs(tr),pairs(va);mem=torch.stack(memory).mean(0) if memory else torch.zeros_like(q0);qb=q0+a.beta*mem;qb=qb/qb.norm();q=qb.clone().requires_grad_(True)
   if train:
    opt=torch.optim.Adam([q],lr=.03)
    for _ in range(30):
     qn=q/q.norm();loss=pairloss(M,qn,train)+.5*(1-qn@qb);opt.zero_grad();loss.backward();opt.step()
   qn=(q/q.norm()).detach();improve=0.
   if held:improve=float(((pairloss(M,qb,held)-pairloss(M,qn,held))/(pairloss(M,qb,held)+1e-8)).cpu())
   write=(a.write=='all' and bool(train)) or (a.write=='certified' and bool(train) and bool(held) and improve>a.write_margin)
   if write:
    memory.append((qn-qb).detach());memory=memory[-a.memory_size:]
   np.save(od/(vid+'.npy'),(X@qn).cpu().numpy().astype(np.float32));rec={'dataset':ds,'video_id':vid,'train_pairs':len(train),'held_pairs':len(held),'held_improvement':improve,'write':write,'memory_len':len(memory)};meta.append(rec)
 (out/'run.json').write_text(json.dumps({'config':vars(a),'prompt':PROMPT,'videos':meta},indent=2)+'\n');print(json.dumps({'writes':sum(x['write'] for x in meta),'n':len(meta)},indent=2))
if __name__=='__main__':main()
