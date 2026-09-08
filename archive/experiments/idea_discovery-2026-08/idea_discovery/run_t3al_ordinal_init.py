#!/usr/bin/env python3
"""Two-stage ordinal query initialization followed by the full repaired T3AL loop."""
from __future__ import annotations
import argparse,importlib.util,json,math,sys,types,zlib
from pathlib import Path
import numpy as np,torch,open_clip
ROOT=Path(__file__).resolve().parents[2];RETR=Path('/home/jehc223/Retrieval-hate');sys.path.insert(0,str(ROOT/'third_party/T3AL'))
spec=importlib.util.spec_from_file_location('campaign_t3al',RETR/'scripts/repro_campaign/run_t3al.py');campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)
from run_t3al_prequential_gate import gated_get_indices
ASR={d:ROOT/f'results/idea_discovery/paradigm_adapt/asr_inference_only/{d}.jsonl' for d in ('HateMM','HateClipSeg','MHC','MHC_zh')}
PROMPT='abusive or hateful behavior toward a person or group'
def load(p):
 d={}
 for x in p.read_text().splitlines():
  r=json.loads(x);forbidden=set(r)-{'video_id','chunk_index','span','text','z_masked','z_isolated'}
  if forbidden:raise RuntimeError(f'label/GT-bearing ASR fields rejected: {forbidden}')
  d.setdefault(r['video_id'],[]).append(r)
 return d
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',required=True);ap.add_argument('--out',required=True);ap.add_argument('--strength',type=float,default=1.);ap.add_argument('--pair-margin',type=float,default=.3);ap.add_argument('--topology-weight',type=float,default=.1);ap.add_argument('--stage',choices=['pre','post'],default='pre');ap.add_argument('--control',choices=['none','shuffle','reverse'],default='none');ap.add_argument('--seed',type=int,default=20260825);a=ap.parse_args();cohort=[json.loads(x) for x in Path(a.cohort).read_text().splitlines() if x.strip()];wanted={(x['dataset'],x['video_id']):float(x['duration']) for x in cohort};am={d:load(p) for d,p in ASR.items()};device='cuda';net=campaign.build_net(campaign.PRESETS['D_anet'],device);net.eval();net.refine_with_captions=False;net.pristine={k:v.detach().clone() for k,v in net.model.state_dict().items()};net.gate_mode='repaired';net.gate_pool=.2;net.get_indices=types.MethodType(gated_get_indices,net);net.inverted_cls[1]=PROMPT;original_compute=net.compute_tta_embedding
 def shifted(self,label,device):
  q=original_compute(label,device)
  if self.query_delta is not None and a.stage=='pre':q=q+a.strength*self.query_delta
  return q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
 net.compute_tta_embedding=types.MethodType(shifted,net);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=[]
 for ds,vid in sorted(wanted):
  fp=campaign.FEAT_DIR(ds)/(vid+'.npy')
  if not fp.exists() or vid not in am[ds]:continue
  F=np.load(fp);dur=wanted[(ds,vid)];rr=sorted(am[ds][vid],key=lambda r:(float(r['span'][0]),float(r['span'][1])));net.model.load_state_dict(net.pristine,strict=False);raw=torch.tensor(F,device=device);X=raw@net.model.visual.proj.detach();X=X/X.norm(dim=1,keepdim=True).clamp_min(1e-8);q0=original_compute(PROMPT,device).detach()[0];means=[];zs=[]
  for r in rr:
   lo=max(0,min(len(X)-1,int(float(r['span'][0])/dur*len(X))));hi=min(len(X),max(lo+1,int(np.ceil(float(r['span'][1])/dur*len(X)))));means.append(X[lo:hi].mean(0));zs.append(float(r.get('z_masked',r.get('z_isolated',-20))))
  confidence=np.asarray([1/(1+math.exp(-np.clip(v/4,-30,30))) for v in zs]);pairs=[(i,j) for i in range(len(zs)) for j in range(len(zs)) if confidence[i]-confidence[j]>=a.pair_margin]
  if a.control=='shuffle':
   rng=np.random.default_rng(a.seed^zlib.crc32(vid.encode()));perm=rng.permutation(len(zs));pairs=[(int(perm[i]),int(perm[j])) for i,j in pairs]
  if a.control=='reverse':pairs=[(j,i) for i,j in pairs]
  delta=torch.zeros_like(q0,requires_grad=True)
  if pairs:
   M=torch.stack(means);optq=torch.optim.Adam([delta],lr=.03)
   for _ in range(30):
    q=(q0+delta);q=q/q.norm();s=M@q;loss=torch.stack([torch.nn.functional.softplus(-10*(s[i]-s[j])) for i,j in pairs]).mean()+.5*(1-q@q0)
    if a.topology_weight and len(X)>1:
     edge=((X[1:]*X[:-1]).sum(1)+1)/2;fs=X@q;loss=loss+a.topology_weight*(edge*(fs[1:]-fs[:-1]).square()).mean()/(fs.var(unbiased=False)+1e-6)
    optq.zero_grad();loss.backward();optq.step()
  net.query_delta=delta.detach().unsqueeze(0) if a.stage=='pre' else None;vseed=(a.seed^zlib.crc32(vid.encode()))%(2**31-1);torch.manual_seed(vseed);np.random.seed(vseed%(2**31));net.force_index=1;net.last_similarity=None;opt=campaign.make_optimizer(net);x=(0,[vid],torch.from_numpy(F).float().unsqueeze(0).to(device));net(x,opt);sim=net.last_similarity
  if a.stage=='post':
   Xp=raw@net.model.visual.proj.detach();Xp=Xp/Xp.norm(dim=1,keepdim=True).clamp_min(1e-8);qp=original_compute(PROMPT,device).detach()[0];Mp=[]
   for r in rr:
    lo=max(0,min(len(Xp)-1,int(float(r['span'][0])/dur*len(Xp))));hi=min(len(Xp),max(lo+1,int(np.ceil(float(r['span'][1])/dur*len(Xp)))));Mp.append(Xp[lo:hi].mean(0))
   dp=torch.zeros_like(qp,requires_grad=True)
   if pairs:
    MM=torch.stack(Mp);oq=torch.optim.Adam([dp],lr=.03)
    for _ in range(30):
     q=(qp+dp);q=q/q.norm();ss=MM@q;ll=torch.stack([torch.nn.functional.softplus(-10*(ss[i]-ss[j])) for i,j in pairs]).mean()+.5*(1-q@qp);oq.zero_grad();ll.backward();oq.step()
   q=(qp+a.strength*dp.detach());q=q/q.norm();curve=(Xp@q).detach().cpu().numpy()
  else:curve=np.zeros(len(F),np.float32) if sim is None else np.asarray(sim[:len(F)],np.float32)
  od=out/ds;od.mkdir(exist_ok=True);np.save(od/(vid+'.npy'),curve);rec={'dataset':ds,'video_id':vid,'n_pairs':len(pairs),'control':a.control,'strength':a.strength,'stage':a.stage};meta.append(rec);print(json.dumps(rec),flush=True)
 (out/'run.json').write_text(json.dumps({'config':vars(a),'prompt':PROMPT,'videos':meta},indent=2)+'\n')
if __name__=='__main__':main()
