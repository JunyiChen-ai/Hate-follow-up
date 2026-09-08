#!/usr/bin/env python3
"""Repaired T3AL with video-specific positive/confounder query directions."""
from __future__ import annotations
import argparse,json,sys,types,zlib,re
from pathlib import Path
import numpy as np,torch,open_clip
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'third_party/T3AL'));sys.path.insert(0,str(ROOT/'scripts/idea_discovery'))
import run_t3al_prequential_gate as gate
campaign=gate.campaign;tokenize=open_clip.get_tokenizer('coca_ViT-L-14')
def dynamic_query(self,class_label,device):
 pos=getattr(self,'video_positive',[]);neg=getattr(self,'video_confounder',[])
 if not pos:return self._fixed_compute('hateful content',device)
 pt=tokenize(['a video of '+x for x in pos]).to(device);p=self.model.encode_text(pt);p=p/p.norm(dim=1,keepdim=True)
 q=p.mean(0,keepdim=True)
 if neg:
  nt=tokenize(['a video of '+x for x in neg]).to(device);n=self.model.encode_text(nt);n=n/n.norm(dim=1,keepdim=True);q=q-n.mean(0,keepdim=True)
 return q/q.norm(dim=-1,keepdim=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',required=True);ap.add_argument('--queries',required=True);ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,default=20260825);ap.add_argument('--ablation',choices=['paired','positive_only','shuffled_confounder'],default='paired');a=ap.parse_args()
 co=[json.loads(x) for x in Path(a.cohort).read_text().splitlines()];Q={(r['dataset'],r['video_id']):r for r in map(json.loads,Path(a.queries).open())};device='cuda';net=campaign.build_net(campaign.PRESETS['D_anet'],device);net.eval();net.refine_with_captions=False;net.pristine={k:v.detach().clone() for k,v in net.model.state_dict().items()};net.gate_mode='repaired';net.gate_pool=.2;net.get_indices=types.MethodType(gate.gated_get_indices,net);net._fixed_compute=net.compute_tta_embedding;net.compute_tta_embedding=types.MethodType(dynamic_query,net)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=[];ordered_q=[Q[k] for k in sorted(Q)]
 for r in co:
  ds,v=r['dataset'],r['video_id'];q=Q.get((ds,v),{});net.video_positive=q.get('positive',[]);net.video_confounder=q.get('confounder',[])
  if a.ablation=='positive_only':net.video_confounder=[]
  elif a.ablation=='shuffled_confounder' and net.video_positive:
   i=sorted(Q).index((ds,v));net.video_confounder=ordered_q[(i+1)%len(ordered_q)].get('confounder',[])
  F=np.load(campaign.FEAT_DIR(ds)/f'{v}.npy');ft=torch.from_numpy(F).float().unsqueeze(0);seed=(a.seed^zlib.crc32(v.encode()))%(2**31-1);_,_,sim=campaign.run_video(net,ft,v,1,device,seed);curve=np.zeros(len(F),np.float32) if sim is None else np.asarray(sim[:len(F)],np.float32);od=out/ds;od.mkdir(exist_ok=True);np.save(od/f'{v}.npy',curve);rec={'dataset':ds,'video_id':v,'generated':bool(net.video_positive),'n_pos':len(net.video_positive),'n_neg':len(net.video_confounder),'ablation':a.ablation};meta.append(rec);print(json.dumps(rec),flush=True)
 (out/'run.json').write_text(json.dumps({'config':vars(a),'videos':meta},indent=2)+'\n')
if __name__=='__main__':main()
