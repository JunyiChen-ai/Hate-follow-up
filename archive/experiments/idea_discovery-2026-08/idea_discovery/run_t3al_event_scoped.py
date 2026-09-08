#!/usr/bin/env python3
"""Event-scoped reversible T3AL pilot.

The pretrained state is restored at every label-free event boundary, preventing
pseudo-label writes in one temporal regime from contaminating the next.  Boundaries
are detected without labels from visual geometry and/or timestamped ASR changes.
"""
from __future__ import annotations
import argparse,json,sys,types,zlib
from pathlib import Path
import numpy as np,torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[2]; RETR=Path('/home/jehc223/Retrieval-hate')
sys.path.insert(0,str(ROOT/'third_party/T3AL'));sys.path.insert(0,str(ROOT/'scripts/idea_discovery'))
import run_t3al_prequential_gate as gate
campaign=gate.campaign

def rank01(x):
 from scipy.stats import rankdata
 return (rankdata(x)-.5)/len(x)
def boundaries(F,text,mode,min_sec,max_sec):
 T=len(F); rate=campaign.FPS; min_gap=max(8,round(min_sec*rate)); max_gap=max(min_gap+1,round(max_sec*rate))
 X=F/(np.linalg.norm(F,axis=1,keepdims=True)+1e-8); w=max(2,round(2*rate)); vis=np.zeros(T)
 for i in range(w,T-w):vis[i]=1-X[i-w:i].mean(0)@X[i:i+w].mean(0)
 vis=rank01(gaussian_filter1d(vis,rate))
 if mode=='fixed': return list(range(max_gap,T,max_gap))
 if mode=='visual': score=vis
 else:
  txt=rank01(gaussian_filter1d(np.abs(np.gradient(text)),rate))
  # Harmonic rank requires both streams when both change; max-gap fallback still
  # permits genuinely unmatched long events.
  score=2*vis*txt/(vis+txt+1e-8)
 peaks,_=find_peaks(score,distance=min_gap,prominence=.15)
 cuts=[];last=0
 for p in peaks:
  while p-last>max_gap: last+=max_gap;cuts.append(last)
  if p-last>=min_gap:cuts.append(int(p));last=int(p)
 while T-last>max_gap:last+=max_gap;cuts.append(last)
 return sorted(set(x for x in cuts if min_gap<=x<=T-min_gap))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['fixed','visual','multimodal'],required=True)
 ap.add_argument('--cohort',required=True);ap.add_argument('--out',required=True);ap.add_argument('--min-sec',type=float,default=10);ap.add_argument('--max-sec',type=float,default=30);ap.add_argument('--seed',type=int,default=20260825);a=ap.parse_args()
 cohort=[json.loads(x) for x in Path(a.cohort).read_text().splitlines()];wanted={(r['dataset'],r['video_id']) for r in cohort}
 evidence={(r['dataset'],r['video_id']):r for r in cohort};device='cuda';net=campaign.build_net(campaign.PRESETS['D_anet'],device);net.eval();net.refine_with_captions=False
 net.pristine={k:v.detach().clone() for k,v in net.model.state_dict().items()};net.gate_mode='repaired';net.gate_pool=.2;net.get_indices=types.MethodType(gate.gated_get_indices,net)
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=[]
 for ds,vid in sorted(wanted):
  fp=campaign.FEAT_DIR(ds)/f'{vid}.npy';
  if not fp.exists():continue
  F=np.load(fp);cr=evidence[(ds,vid)];vals=np.asarray([b['scores']['speech'] for b in cr['bins']]);text=np.interp((np.arange(len(F))+.5)/len(F),(np.arange(len(vals))+.5)/len(vals),vals)
  cuts=boundaries(F,text,a.mode,a.min_sec,a.max_sec);edges=[0]+cuts+[len(F)];curve=np.zeros(len(F),np.float32)
  for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
   ft=torch.from_numpy(F[lo:hi]).float().unsqueeze(0);vseed=(a.seed^zlib.crc32(f'{vid}:{lo}:{hi}'.encode()))%(2**31-1)
   _,_,sim=campaign.run_video(net,ft,vid,1,device,vseed);curve[lo:hi]=0 if sim is None else np.asarray(sim[:hi-lo],np.float32)
  od=out/ds;od.mkdir(exist_ok=True);np.save(od/f'{vid}.npy',curve);meta.append({'dataset':ds,'video_id':vid,'T':len(F),'cuts':cuts})
  print(json.dumps(meta[-1]),flush=True)
 (out/'run.json').write_text(json.dumps({'config':vars(a),'videos':meta},indent=2)+'\n')
if __name__=='__main__':main()
