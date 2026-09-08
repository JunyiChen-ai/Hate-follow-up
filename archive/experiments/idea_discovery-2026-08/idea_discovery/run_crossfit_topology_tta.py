#!/usr/bin/env python3
"""Two-fold cross-fitted coarse-to-dense ordinal query adaptation."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,torch,open_clip
ROOT=Path(__file__).resolve().parents[2];RETR=Path('/home/jehc223/Retrieval-hate');GT=RETR/'data/gt/frame_gt_4fps';FEAT=RETR/'data/CLIP_Embedding';PROMPT='abusive or hateful behavior toward a person or group'
ASR={'HateMM':ROOT/'results/hatemm_localization/per_chunk.jsonl','HateClipSeg':ROOT/'results/reproduction/ours/hateclipseg/per_chunk.jsonl','MHC':ROOT/'results/reproduction/ours/mhclip_en/per_chunk.jsonl','MHC_zh':ROOT/'results/reproduction/ours/mhclip_zh/per_chunk.jsonl'}
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--topology-weight',type=float,default=.1);ap.add_argument('--pair-margin',type=float,default=.4);ap.add_argument('--prompt',default=PROMPT);ap.add_argument('--control',choices=['none','shuffle_parity','reverse'],default='none');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);device='cuda';model,_,_=open_clip.create_model_and_transforms('coca_ViT-L-14',pretrained='mscoco_finetuned_laion2B-s13B-b90k');model=model.to(device).eval();P=model.visual.proj.detach();tok=open_clip.get_tokenizer('coca_ViT-L-14');q0=model.encode_text(tok(['a video of action '+a.prompt]).to(device)).detach()[0];q0=q0/q0.norm();meta=[]
 for ds,p in ASR.items():
  by={}
  for r in rows(p):by.setdefault(r['video_id'],[]).append(r)
  z=np.load(GT/(ds+'.npz'),allow_pickle=True);gm={str(v):(float(d),str(s)) for v,d,s in zip(z['video_ids'],z['duration'],z['split'])};od=out/ds;od.mkdir(exist_ok=True)
  for vid,rr in sorted(by.items()):
   fp=FEAT/ds/'coca_vitL14_4fps'/(vid+'.npy')
   if vid not in gm or gm[vid][1]!='test' or not fp.exists():continue
   rr=sorted(rr,key=lambda r:(float(r['span'][0]),float(r['span'][1])));dur=gm[vid][0];X=torch.tensor(np.load(fp),device=device)@P;X=X/X.norm(dim=1,keepdim=True).clamp_min(1e-8);M=[];spans=[];b=[]
   for r in rr:
    lo=max(0,min(len(X)-1,int(float(r['span'][0])/dur*len(X))));hi=min(len(X),max(lo+1,int(np.ceil(float(r['span'][1])/dur*len(X)))));M.append(X[lo:hi].mean(0));spans.append((lo,hi));zv=float(r.get('z_masked',r.get('z_isolated',-20)));b.append(1/(1+math.exp(-np.clip(zv/4,-30,30))))
   M=torch.stack(M);b=torch.tensor(b,device=device)
   if a.control=='shuffle_parity' and len(b)>2:
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{ds}/{vid}'.encode()).digest()[:8],'little'));perm=np.arange(len(b))
    for parity in (0,1):
     idx=np.arange(parity,len(b),2);perm[idx]=rng.permutation(idx)
    b=b[torch.tensor(perm,device=device)]
   if a.control=='reverse':b=1-b
   edge=((X[1:]*X[:-1]).sum(1)+1)/2;qs=[];npairs=[]
   for parity in (0,1):
    nodes=[i for i in range(len(b)) if i%2==parity];pairs=[(i,j) for i in nodes for j in nodes if float(b[i]-b[j])>=a.pair_margin];q=q0.clone().requires_grad_(True)
    if pairs:
     opt=torch.optim.Adam([q],lr=.03)
     for _ in range(30):
      qn=q/q.norm();s=M@qn;loss=torch.stack([torch.nn.functional.softplus(-10*(s[i]-s[j])) for i,j in pairs]).mean()+.5*(1-qn@q0)
      if a.topology_weight and len(X)>1:
       fs=X@qn;loss=loss+a.topology_weight*(edge*(fs[1:]-fs[:-1]).square()).mean()/(fs.var(unbiased=False)+1e-6)
      opt.zero_grad();loss.backward();opt.step()
    qs.append((q/q.norm()).detach());npairs.append(len(pairs))
   accum=torch.zeros(len(X),device=device);count=torch.zeros(len(X),device=device)
   for i,(lo,hi) in enumerate(spans):accum[lo:hi]+=X[lo:hi]@qs[1-i%2];count[lo:hi]+=1
   fallback=X@((qs[0]+qs[1])/2);curve=torch.where(count>0,accum/count.clamp_min(1),fallback);np.save(od/(vid+'.npy'),curve.cpu().numpy().astype(np.float32));meta.append({'dataset':ds,'video_id':vid,'pairs_even':npairs[0],'pairs_odd':npairs[1]})
 (out/'run.json').write_text(json.dumps({'config':vars(a),'prompt':a.prompt,'videos':meta},indent=2)+'\n')
if __name__=='__main__':main()
