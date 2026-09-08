#!/usr/bin/env python3
"""Run repaired T3AL with matched-coverage reservoir write gates.

Modes differ only in positive reservoir selection.  `repaired` uses visual rank,
`agreement` uses contemporaneous frozen ASR margin, and `prequential` uses the
first-order effect of writing each candidate visual feature on held-out local ASR
evidence minus far-field drift.  The upstream all-frames get_indices bug is repaired
in every arm and is not counted as a contribution.
"""
from __future__ import annotations
import argparse, importlib.util, json, math, os, sys, types, zlib
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parents[2]
RETR=Path('/home/jehc223/Retrieval-hate')
sys.path.insert(0,str(HERE/'third_party/T3AL'))
spec=importlib.util.spec_from_file_location('campaign_t3al',RETR/'scripts/repro_campaign/run_t3al.py')
campaign=importlib.util.module_from_spec(spec); spec.loader.exec_module(campaign)

ASR={
 'HateMM':HERE/'results/hatemm_localization/per_chunk.jsonl',
 'HateClipSeg':HERE/'results/reproduction/ours/hateclipseg/per_chunk.jsonl',
 'MHC':HERE/'results/reproduction/ours/mhclip_en/per_chunk.jsonl',
 'MHC_zh':HERE/'results/reproduction/ours/mhclip_zh/per_chunk.jsonl'}

def load_asr(path):
 out={}
 for line in path.read_text().splitlines():
  r=json.loads(line); out.setdefault(r['video_id'],[]).append(r)
 return out

def asr_curve(rows,duration,n,temp):
 a=np.zeros(n); w=np.zeros(n)
 for r in rows:
  s,e=map(float,r['span']); z=float(r.get('z_masked',r.get('z_isolated',-20)))
  v=2/(1+math.exp(-np.clip(z/temp,-30,30)))-1
  i=max(0,int(s/duration*n)); j=min(n,max(i+1,int(math.ceil(e/duration*n))))
  a[i:j]+=v; w[i:j]+=1
 x=np.zeros(n); ok=w>0; x[ok]=a[ok]/w[ok]
 return x,ok

def evenly(idx,n):
 idx=idx.sort().values
 if len(idx)>=n:
  return idx[torch.linspace(0,len(idx)-1,n,device=idx.device).round().long()]
 return idx.repeat(math.ceil(n/max(len(idx),1)))[:n]

def gated_get_indices(self,signal):
 s=signal.reshape(-1); T=len(s); n=min(self.n,max(1,T//2)); pool=max(n,int(math.ceil(T*self.gate_pool)))
 pos=torch.topk(s,pool).indices; neg=torch.topk(-s,pool).indices
 # Negatives remain the same in all matched arms and cannot overlap positives.
 neg=evenly(neg,n)
 if self.gate_mode=='repaired': return evenly(pos,n),neg
 if self.gate_mode in ('hardneg','hardneg_agree'):
  target=torch.as_tensor(self.gate_target,device=s.device,dtype=s.dtype);known=torch.as_tensor(self.gate_known,device=s.device,dtype=torch.bool)
  psel=evenly(pos,n) if self.gate_mode=='hardneg' else evenly(pos[torch.argsort(target[pos],descending=True)],n)
  allowed=known & (target<=0)
  allowed[psel]=False
  # Hard negatives are visually confusable high-score frames contradicted by
  # the timestamped auxiliary stream, rather than trivial black/title frames.
  cand=torch.where(allowed)[0]
  if len(cand):nsel=evenly(cand[torch.argsort(s[cand],descending=True)],n)
  else:
   fallback=neg[~torch.isin(neg,psel)];nsel=evenly(fallback if len(fallback) else neg,n)
  return psel.sort().values,nsel.sort().values
 if self.gate_mode in ('external_joint','external_visual'):
  # External MLLM evidence never replaces the visual curve: it ranks candidates
  # inside the same T3AL visual pool, so selected writes remain visually grounded.
  target=torch.as_tensor(self.gate_target,device=s.device,dtype=s.dtype)
  return pos[torch.topk(target[pos],n).indices].sort().values,neg
 if self.gate_mode in ('persistence','persistent_prequential'):
  from scipy.ndimage import gaussian_filter1d
  from scipy.stats import rankdata
  a=s.detach().float().cpu().numpy(); ps=np.zeros(T,dtype=np.float32)
  # A frame is persistent when it remains in an upper level set across both
  # threshold and temporal-scale perturbations.  No label fixes these levels.
  for sigma in (0,4,16):
   y=gaussian_filter1d(a,sigma,mode='nearest') if sigma else a
   for q in (.6,.7,.8,.9): ps += y >= np.quantile(y,q)
  ps/=12.; persistence=torch.as_tensor(ps,device=s.device,dtype=s.dtype)
  if self.gate_mode=='persistence':
   return pos[torch.topk(persistence[pos],n).indices].sort().values,neg
 target=torch.as_tensor(self.gate_target,device=s.device,dtype=s.dtype)
 known=torch.as_tensor(self.gate_known,device=s.device,dtype=torch.bool)
 if self.gate_mode=='agreement':
  return pos[torch.topk(target[pos],n).indices].sort().values,neg
 X=self.gate_raw @ self.model.visual.proj
 X=X/(X.norm(dim=-1,keepdim=True)+1e-8)
 # A prototype write toward x_i changes every frame in proportion to x_t^T x_i.
 # Centering removes a candidate-independent temperature/offset shortcut.
 influence=X @ X[pos].T; influence=influence-influence.mean(0,keepdim=True)
 times=torch.arange(T,device=s.device); J=[]
 for col,i in enumerate(pos):
  local=(times-i).abs()<=self.gate_radius
  far=(times-i).abs()>2*self.gate_radius
  ww=target.abs()*known
  den=ww[local].sum()
  gain=((influence[local,col]*target[local]*ww[local]).sum()/den) if den>1e-8 else influence.new_tensor(0.)
  drift=influence[far,col].abs().mean() if far.any() else influence.new_tensor(0.)
  J.append(gain-self.gate_lambda*drift)
 J=torch.stack(J)
 if self.gate_mode=='persistent_prequential':
  # Rank aggregation prevents either modality's arbitrary score scale from
  # dominating.  Both persistence and held-out consequence must be high.
  jr=torch.as_tensor(rankdata(J.detach().cpu().numpy()),device=s.device,dtype=s.dtype)/len(J)
  pr=torch.as_tensor(rankdata(persistence[pos].detach().cpu().numpy()),device=s.device,dtype=s.dtype)/len(J)
  J=2*jr*pr/(jr+pr+1e-8)
 self.gate_js.append(J.detach().cpu().numpy()); self.gate_confs.append(s[pos].detach().cpu().numpy())
 return pos[torch.topk(J,n).indices].sort().values,neg

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--datasets',default='HateMM,MHC,MHC_zh,HateClipSeg')
 ap.add_argument('--limit',type=int,default=0); ap.add_argument('--mode',choices=['repaired','agreement','prequential','persistence','persistent_prequential','hardneg','hardneg_agree','external_joint','external_visual'],required=True)
 ap.add_argument('--pool',type=float,default=.2); ap.add_argument('--radius-sec',type=float,default=10)
 ap.add_argument('--lambda-drift',type=float,default=1); ap.add_argument('--text-temp',type=float,default=4)
 ap.add_argument('--control',choices=['none','reverse','shift'],default='none')
 ap.add_argument('--credit-file',default='')
 ap.add_argument('--seed',type=int,default=20260825); ap.add_argument('--out',required=True); a=ap.parse_args()
 device='cuda'; net=campaign.build_net(campaign.PRESETS['D_anet'],device); net.eval(); net.refine_with_captions=False
 net.pristine={k:v.detach().clone() for k,v in net.model.state_dict().items()}
 net.gate_mode=a.mode; net.gate_pool=a.pool; net.gate_lambda=a.lambda_drift; net.gate_radius=max(1,round(a.radius_sec*campaign.FPS))
 net.get_indices=types.MethodType(gated_get_indices,net)
 out=Path(a.out); out.mkdir(parents=True,exist_ok=True); summary={}
 credit={}
 if a.credit_file:
  for r in load_asr(Path(a.credit_file)).values():
   # load_asr groups by video id, so recover dataset from each record.
   for x in r: credit[(x['dataset'],x['video_id'])]=x
 for ds in a.datasets.split(','):
  amap=load_asr(ASR[ds]); meta=campaign.gt_meta(ds); fdir=campaign.FEAT_DIR(ds)
  ids=sorted(v for v,(dur,sp) in meta.items() if sp=='test' and (fdir/f'{v}.npy').exists() and v in amap)
  if a.limit:
   ids=ids[:a.limit]
  od=out/ds; od.mkdir(parents=True,exist_ok=True); summary[ds]=[]
  for vid in ids:
   F=np.load(fdir/f'{vid}.npy'); dur=meta[vid][0]; target,known=asr_curve(amap[vid],dur,len(F),a.text_temp)
   if a.control=='reverse':target=-target
   elif a.control=='shift':
    shift=max(1,len(target)//2);target=np.roll(target,shift);known=np.roll(known,shift)
   if a.mode.startswith('external_'):
    cr=credit.get((ds,vid))
    if cr is None: continue
    key='joint' if a.mode=='external_joint' else 'visual'
    vals=np.asarray([b['scores'][key] for b in cr['bins']],dtype=float)
    target=np.interp((np.arange(len(F))+.5)/len(F),(np.arange(len(vals))+.5)/len(vals),vals)
    known=np.ones(len(F),dtype=bool)
   net.gate_target=target; net.gate_known=known; net.gate_raw=torch.from_numpy(F).float().to(device); net.gate_js=[]; net.gate_confs=[]
   ft=torch.from_numpy(F).float().unsqueeze(0); vseed=(a.seed^zlib.crc32(vid.encode()))%(2**31-1)
   output,mask,sim=campaign.run_video(net,ft,vid,1,device,vseed)
   curve=np.zeros(len(F),np.float32) if sim is None else np.asarray(sim[:len(F)],np.float32)
   np.save(od/f'{vid}.npy',curve)
   rec={'video_id':vid,'T':len(F),'mode':a.mode,'n_steps_scored':len(net.gate_js)}
   if net.gate_js:
    j=np.concatenate(net.gate_js); c=np.concatenate(net.gate_confs)
    from scipy.stats import spearmanr
    rec['j_conf_spearman']=float(spearmanr(j,c).statistic)
   summary[ds].append(rec); print(json.dumps({'dataset':ds,**rec}),flush=True)
 (out/'run.json').write_text(json.dumps({'config':vars(a),'videos':summary},indent=2)+'\n')

if __name__=='__main__': main()
