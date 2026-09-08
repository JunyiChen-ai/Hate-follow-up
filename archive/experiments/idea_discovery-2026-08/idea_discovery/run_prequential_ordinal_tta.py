#!/usr/bin/env python3
"""Cross-modal Prequential Ordinal TTA (CPO-TTA) exploratory implementation.

Timestamped speech provides only within-video ordinal constraints.  A CoCa/T3AL
query is adapted on a deterministic training subset of those constraints.  The
update is committed only if it transfers to held-out constraints by a frozen
relative-loss margin.  Scores are always visual-only after adaptation.
"""
from __future__ import annotations
import argparse,json,math,hashlib
from pathlib import Path
import numpy as np,torch,open_clip

ROOT=Path(__file__).resolve().parents[2]; RETR=Path('/home/jehc223/Retrieval-hate')
GT=RETR/'data/gt/frame_gt_4fps'; FEAT=RETR/'data/CLIP_Embedding';
ASR={'HateMM':ROOT/'results/hatemm_localization/per_chunk.jsonl','HateClipSeg':ROOT/'results/reproduction/ours/hateclipseg/per_chunk.jsonl','MHC':ROOT/'results/reproduction/ours/mhclip_en/per_chunk.jsonl','MHC_zh':ROOT/'results/reproduction/ours/mhclip_zh/per_chunk.jsonl'}
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--prompt',default='hateful content');ap.add_argument('--pair-margin',type=float,default=.4);ap.add_argument('--min-train-pairs',type=int,default=1);ap.add_argument('--accept-margin',type=float,default=.2);ap.add_argument('--lr',type=float,default=.03);ap.add_argument('--steps',type=int,default=30);ap.add_argument('--anchor',type=float,default=.5);ap.add_argument('--visual-weight',type=float,default=0.);ap.add_argument('--topology-weight',type=float,default=0.);ap.add_argument('--objective',choices=['ordinal','absolute'],default='ordinal');ap.add_argument('--control',choices=['none','shuffle_within','shuffle_parity','reverse'],default='none');ap.add_argument('--split',choices=['pair','chunk'],default='pair');a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);device='cuda';model,_,_=open_clip.create_model_and_transforms('coca_ViT-L-14',pretrained='mscoco_finetuned_laion2B-s13B-b90k');model=model.to(device).eval();tok=open_clip.get_tokenizer('coca_ViT-L-14');P=model.visual.proj.detach();q0=model.encode_text(tok(['a video of action '+a.prompt]).to(device)).detach()[0];q0=q0/q0.norm();meta=[]
 for ds,apath in ASR.items():
  by={}
  for r in rows(apath):by.setdefault(r['video_id'],[]).append(r)
  z=np.load(GT/(ds+'.npz'),allow_pickle=True);gm={str(v):(float(d),str(s)) for v,d,s in zip(z['video_ids'],z['duration'],z['split'])};od=out/ds;od.mkdir(exist_ok=True)
  for vid,rr in sorted(by.items()):
   rr=sorted(rr,key=lambda r:(float(r['span'][0]),float(r['span'][1])))
   fp=FEAT/ds/'coca_vitL14_4fps'/(vid+'.npy')
   if vid not in gm or gm[vid][1]!='test' or not fp.exists():continue
   duration=gm[vid][0];X=torch.tensor(np.load(fp),device=device)@P;X=X/X.norm(dim=1,keepdim=True).clamp_min(1e-8);M=[];b=[]
   for r in rr:
    lo=max(0,min(len(X)-1,int(float(r['span'][0])/duration*len(X))));hi=min(len(X),max(lo+1,int(np.ceil(float(r['span'][1])/duration*len(X)))));M.append(X[lo:hi].mean(0));zv=float(r.get('z_masked',r.get('z_isolated',-20)));b.append(1/(1+math.exp(-np.clip(zv/4,-30,30))))
   M=torch.stack(M);b=torch.tensor(b,device=device)
   if a.control=='shuffle_within' and len(b)>1:
    seed=int.from_bytes(hashlib.sha256(f'{ds}/{vid}'.encode()).digest()[:8],'little')
    b=b[torch.tensor(np.random.default_rng(seed).permutation(len(b)),device=device)]
   if a.control=='shuffle_parity' and len(b)>2:
    seed=int.from_bytes(hashlib.sha256(f'parity/{ds}/{vid}'.encode()).digest()[:8],'little');rng=np.random.default_rng(seed);perm=np.arange(len(b))
    for parity in (0,1):
     idx=np.arange(parity,len(b),2);perm[idx]=rng.permutation(idx)
    b=b[torch.tensor(perm,device=device)]
   if a.control=='reverse': b=1-b
   pairs=[(i,j) for i in range(len(b)) for j in range(len(b)) if float(b[i]-b[j])>=a.pair_margin]
   if a.split=='pair':train=[x for k,x in enumerate(pairs) if k%3];held=[x for k,x in enumerate(pairs) if k%3==0]
   else:
    train_nodes={i for i in range(len(b)) if i%2};held_nodes=set(range(len(b)))-train_nodes
    train=[x for x in pairs if x[0] in train_nodes and x[1] in train_nodes];held=[x for x in pairs if x[0] in held_nodes and x[1] in held_nodes]
   q=q0.clone().requires_grad_(True);base_score=X@q0;nk=max(1,len(X)//5);vp=torch.topk(base_score,nk).indices;vn=torch.topk(-base_score,nk).indices;edge=((X[1:]*X[:-1]).sum(1)+1)/2
   train_nodes_abs=sorted({i for x in train for i in x})
   if (len(train)>=a.min_train_pairs if a.objective=='ordinal' else len(train_nodes_abs)>=a.min_train_pairs):
    opt=torch.optim.Adam([q],lr=a.lr)
    for _ in range(a.steps):
     qn=q/q.norm().clamp_min(1e-8);s=M@qn
     if a.objective=='ordinal':task=torch.stack([torch.nn.functional.softplus(-10*(s[i]-s[j])) for i,j in train]).mean()
     else:task=torch.nn.functional.mse_loss(torch.sigmoid(10*s[train_nodes_abs]),b[train_nodes_abs])
     loss=task+a.anchor*(1-qn@q0)
     if a.visual_weight:loss=loss+a.visual_weight*torch.nn.functional.softplus(-10*((X[vp]@qn).mean()-(X[vn]@qn).mean()))
     if a.topology_weight and len(X)>1:
      fs=X@qn;tv=(edge*(fs[1:]-fs[:-1]).square()).mean()/(fs.var(unbiased=False)+1e-6);loss=loss+a.topology_weight*tv
     opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([q],1);opt.step()
   qn=q/q.norm().clamp_min(1e-8);improvement=-1.
   if held:
    def loss(qq):
     s=M@qq;return torch.stack([torch.nn.functional.softplus(-10*(s[i]-s[j])) for i,j in held]).mean()
    l0=loss(q0);l1=loss(qn);improvement=float(((l0-l1)/(l0+1e-8)).detach())
   accepted=improvement>a.accept_margin;final_q=qn if accepted else q0;curve=(X@final_q).detach().float().cpu().numpy();np.save(od/(vid+'.npy'),curve.astype(np.float32));rec={'dataset':ds,'video_id':vid,'n_chunks':len(rr),'n_pairs':len(pairs),'n_train':len(train),'n_heldout':len(held),'heldout_relative_improvement':improvement,'accepted':accepted};meta.append(rec);print(json.dumps(rec),flush=True)
 (out/'run.json').write_text(json.dumps({'method':'CPO-TTA','config':vars(a),'videos':meta},indent=2)+'\n')
if __name__=='__main__':main()
