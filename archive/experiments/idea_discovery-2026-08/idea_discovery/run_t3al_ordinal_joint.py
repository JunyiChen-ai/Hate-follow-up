#!/usr/bin/env python3
"""Exploratory T3AL + rank-only timestamped speech ordinal loss."""
from __future__ import annotations
import argparse,importlib.util,json,math,sys,types,zlib
from pathlib import Path
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[2];RETR=Path('/home/jehc223/Retrieval-hate')
sys.path.insert(0,str(ROOT/'third_party/T3AL'))
spec=importlib.util.spec_from_file_location('campaign_t3al',RETR/'scripts/repro_campaign/run_t3al.py');campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)
from run_t3al_prequential_gate import gated_get_indices
ASR={'HateMM':ROOT/'results/hatemm_localization/per_chunk.jsonl','HateClipSeg':ROOT/'results/reproduction/ours/hateclipseg/per_chunk.jsonl','MHC':ROOT/'results/reproduction/ours/mhclip_en/per_chunk.jsonl','MHC_zh':ROOT/'results/reproduction/ours/mhclip_zh/per_chunk.jsonl'}
def load(p):
 d={}
 for x in p.read_text().splitlines():
  r=json.loads(x);d.setdefault(r['video_id'],[]).append(r)
 return d
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',required=True);ap.add_argument('--out',required=True);ap.add_argument('--weight',type=float,default=1.);ap.add_argument('--pair-margin',type=float,default=.3);ap.add_argument('--prompt',default='abusive or hateful behavior toward a person or group');ap.add_argument('--control',choices=['none','shuffle','reverse'],default='none');ap.add_argument('--combine',choices=['joint','pcgrad'],default='joint');ap.add_argument('--seed',type=int,default=20260825);a=ap.parse_args()
 cohort=[json.loads(x) for x in Path(a.cohort).read_text().splitlines()];wanted={(x['dataset'],x['video_id']) for x in cohort};amaps={d:load(p) for d,p in ASR.items()};device='cuda';net=campaign.build_net(campaign.PRESETS['D_anet'],device);net.eval();net.refine_with_captions=False;net.pristine={k:v.detach().clone() for k,v in net.model.state_dict().items()};net.gate_mode='repaired';net.gate_pool=.2;net.get_indices=types.MethodType(gated_get_indices,net)
 base_forward=net.tta_loss.forward;tokenizer=__import__('open_clip').get_tokenizer('coca_ViT-L-14');tokens=tokenizer(['a video of action '+a.prompt]).to(device)
 def ordinal_loss(self,pred,target):
  base=base_forward(pred,target)
  if pred.ndim!=1 or net.ordinal_raw is None or not net.ordinal_pairs:return base
  f=net.ordinal_raw@net.model.visual.proj;f=f/f.norm(dim=1,keepdim=True).clamp_min(1e-8);q=net.model.encode_text(tokens);q=q/q.norm(dim=1,keepdim=True).clamp_min(1e-8);s=f@q[0]
  rank=torch.stack([torch.nn.functional.softplus(-10*(s[i]-s[j])) for i,j in net.ordinal_pairs]).mean()
  if a.combine=='pcgrad':
   params=[p for p in net.model.parameters() if p.requires_grad]
   net.ordinal_grads=torch.autograd.grad(rank,params,retain_graph=True,allow_unused=True)
   net.ordinal_params=params
  return base+a.weight*rank
 net.tta_loss.forward=types.MethodType(ordinal_loss,net.tta_loss);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=[]
 class PCGradOptimizer:
  def __init__(self,opt):self.opt=opt
  def zero_grad(self,*x,**kw):return self.opt.zero_grad(*x,**kw)
  def step(self,*x,**kw):
   if getattr(net,'ordinal_grads',None):
    base=[];aux=[]
    for p,g in zip(net.ordinal_params,net.ordinal_grads):
     if p.grad is not None and g is not None:base.append(p.grad-a.weight*g);aux.append(g)
    dot=sum((x*y).sum() for x,y in zip(base,aux));den=sum((x*x).sum() for x in base).clamp_min(1e-12)
    if dot<0:aux=[g-dot/den*b for g,b in zip(aux,base)]
    for p,b,g in zip([p for p,g in zip(net.ordinal_params,net.ordinal_grads) if p.grad is not None and g is not None],base,aux):p.grad=b+a.weight*g
   net.ordinal_grads=None
   return self.opt.step(*x,**kw)
 for ds,vid in sorted(wanted):
  fp=campaign.FEAT_DIR(ds)/(vid+'.npy');gm=campaign.gt_meta(ds)
  if not fp.exists() or vid not in gm or vid not in amaps[ds]:continue
  F=np.load(fp);dur=gm[vid][0];rr=sorted(amaps[ds][vid],key=lambda r:(float(r['span'][0]),float(r['span'][1])));means=[];z=[]
  for r in rr:
   lo=max(0,min(len(F)-1,int(float(r['span'][0])/dur*len(F))));hi=min(len(F),max(lo+1,int(np.ceil(float(r['span'][1])/dur*len(F)))));means.append(F[lo:hi].mean(0));z.append(float(r.get('z_masked',r.get('z_isolated',-20))))
  confidence=np.asarray([1/(1+math.exp(-np.clip(v/4,-30,30))) for v in z]);pairs=[(i,j) for i in range(len(z)) for j in range(len(z)) if confidence[i]-confidence[j]>=a.pair_margin]
  if a.control=='shuffle':
   rng=np.random.default_rng(a.seed^zlib.crc32(vid.encode()));perm=rng.permutation(len(z));pairs=[(int(perm[i]),int(perm[j])) for i,j in pairs]
  if a.control=='reverse':pairs=[(j,i) for i,j in pairs]
  net.ordinal_pairs=pairs;net.ordinal_raw=torch.tensor(np.stack(means),device=device) if means else None
  ft=torch.from_numpy(F).float().unsqueeze(0);vseed=(a.seed^zlib.crc32(vid.encode()))%(2**31-1)
  if a.combine=='joint':_,_,sim=campaign.run_video(net,ft,vid,1,device,vseed)
  else:
   torch.manual_seed(vseed);np.random.seed(vseed%(2**31));net.force_index=1;net.last_similarity=None;net.model.load_state_dict(net.pristine,strict=False);opt=PCGradOptimizer(campaign.make_optimizer(net));x=(0,[vid],ft.clone().to(device));net(x,opt);sim=net.last_similarity
  curve=np.zeros(len(F),np.float32) if sim is None else np.asarray(sim[:len(F)],np.float32);od=out/ds;od.mkdir(exist_ok=True);np.save(od/(vid+'.npy'),curve);rec={'dataset':ds,'video_id':vid,'n_chunks':len(rr),'n_pairs':len(net.ordinal_pairs),'control':a.control,'combine':a.combine};meta.append(rec);print(json.dumps(rec),flush=True)
 (out/'run.json').write_text(json.dumps({'config':vars(a),'videos':meta},indent=2)+'\n')
if __name__=='__main__':main()
