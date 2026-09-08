#!/usr/bin/env python3
"""Kill-test consensus change points from visual, acoustic, and transcript streams."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).parent));from prequential_write_gate_pilot import interp,ecdf,metrics
ROOT=Path(__file__).resolve().parents[2];RETR=Path('/home/jehc223/Retrieval-hate');GT=RETR/'data/gt/frame_gt_4fps';FEAT=RETR/'data/CLIP_Embedding';BASE=ROOT/'results/idea_discovery/prompt_query/abuse';COHORT=ROOT/'results/idea_discovery/causal_credit/label_blind32.jsonl'
ASR={'HateMM':ROOT/'results/hatemm_localization/per_chunk.jsonl','HateClipSeg':ROOT/'results/reproduction/ours/hateclipseg/per_chunk.jsonl','MHC':ROOT/'results/reproduction/ours/mhclip_en/per_chunk.jsonl','MHC_zh':ROOT/'results/reproduction/ours/mhclip_zh/per_chunk.jsonl'}
def media(ds,v):
 roots={'HateMM':Path('/home/jehc223/data/HateMM/video'),'HateClipSeg':ROOT/'idea-stage/pilots/b1_coverage_audit/data/videos','MHC':Path('/home/jehc223/data/Multihateclip/English/video_mp4'),'MHC_zh':Path('/home/jehc223/data/Multihateclip/Chinese/video')}
 for ext in ['.mp4','.mkv','.webm']:
  p=roots[ds]/(v+ext)
  if p.exists():return p
 return None
def audio_salience(p,n):
 if p is None:return np.zeros(n)
 x=np.frombuffer(subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-f','f32le','-ac','1','-ar','16000','-']),np.float32);edges=np.linspace(0,len(x),n+1).astype(int);r=np.array([np.sqrt(np.mean(x[edges[i]:edges[i+1]]**2)+1e-10) if edges[i+1]>edges[i] else 0 for i in range(n)]);return np.r_[0,np.abs(np.diff(np.log(r+1e-5)))]
def norm(x):return (rankdata(x)-1)/max(len(x)-1,1)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--all-test',action='store_true');ap.add_argument('--out',default=str(ROOT/'results/idea_discovery/multimodal_boundary_pilot.json'));ap.add_argument('--curve-out',default='');cfg=ap.parse_args();dev={(x['dataset'],x['video_id']) for x in map(json.loads,COHORT.read_text().splitlines())};co=[{'dataset':d,'video_id':v} for d,v in sorted(dev)];am={}
 for d,p in ASR.items():
  q={}
  for x in p.read_text().splitlines():r=json.loads(x);q.setdefault(r['video_id'],[]).append(r)
  am[d]=q
 if cfg.all_test:
  co=[]
  for d,q in am.items():
   z=np.load(GT/(d+'.npz'),allow_pickle=True);valid={str(v) for v,s in zip(z['video_ids'],z['split']) if str(s)=='test'};co += [{'dataset':d,'video_id':v} for v in sorted(set(q)&valid) if (BASE/d/(v+'.npy')).exists()]
 cache=[]
 for c in co:
  d,v=c['dataset'],c['video_id'];z=np.load(GT/(d+'.npz'),allow_pickle=True);gm={str(x):(float(du),np.asarray(y,int)) for x,du,y in zip(z['video_ids'],z['duration'],z['y4'])};dur,y=gm[v];s=np.load(BASE/d/(v+'.npy'));X=np.load(FEAT/d/'coca_vitL14_4fps'/(v+'.npy'));X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8);vs=np.r_[0,1-np.sum(X[1:]*X[:-1],axis=1)];aa=audio_salience(media(d,v),len(s));tc=np.zeros(len(s));last=None
  for r in am[d][v]:
   i=min(len(s)-1,max(0,int(float(r['span'][0])/dur*len(s))));val=float(r.get('z_masked',r.get('z_isolated',-20)));tc[i]=0 if last is None else abs(val-last);last=val
  cache.append((d,v,y,s,norm(vs),norm(aa),norm(tc)))
 result=[]
 weights={'w_1_1_2':(.25,.25,.5),'w_1_2_1':(.25,.5,.25),'w_1_2_2':(.2,.4,.4),'w_1_3_4':(.125,.375,.5),'w_1_4_3':(.125,.5,.375),'w_0_1_2':(0,1/3,2/3),'w_0_2_1':(0,2/3,1/3)}
 for fusion in ['visual','audio','text','visual_audio','visual_text','audio_text','mean','median','second','async_1','async_2','async_4',*weights]:
  for spacing in [5,10,20]:
   items=[];tagged=[]
   for d,v,y,s,vs,aa,tt in cache:
    streams={'visual':vs,'audio':aa,'text':tt}
    if fusion.startswith('async_'):
     radius=int(fusion.split('_')[1])*4;arr=[vs,aa,tt];scores=[]
     for i,x in enumerate(arr):
      support=np.maximum.reduce([maximum_filter1d(y,size=2*radius+1,mode='nearest') for j,y in enumerate(arr) if j!=i]);scores.append(2*x*support/(x+support+1e-8))
     sal=np.maximum.reduce(scores)
    elif fusion in weights:sal=weights[fusion][0]*vs+weights[fusion][1]*aa+weights[fusion][2]*tt
    elif fusion in streams:sal=streams[fusion]
    elif '_' in fusion:sal=np.mean([streams[x] for x in fusion.split('_')],0)
    elif fusion=='mean':sal=np.mean([vs,aa,tt],0)
    elif fusion=='median':sal=np.median([vs,aa,tt],0)
    else:sal=np.sort(np.stack([vs,aa,tt]),axis=0)[-2]
    gap=max(1,spacing*4);cand=np.argsort(sal)[::-1];cuts=[]
    for i in cand:
     if i>0 and i<len(s)-1 and all(abs(i-j)>=gap for j in cuts):cuts.append(int(i))
     if len(cuts)>=max(1,len(s)//gap):break
    bounds=[0]+sorted(cuts)+[len(s)];curve=np.empty_like(s)
    for lo,hi in zip(bounds[:-1],bounds[1:]):curve[lo:hi]=np.mean(s[lo:hi])
    item=(y,interp(ecdf(curve),len(y)));items.append(item);tagged.append((d,v,item))
    if cfg.curve_out and fusion=='w_1_3_4' and spacing==10:
     od=Path(cfg.curve_out)/d;od.mkdir(parents=True,exist_ok=True);np.save(od/(v+'.npy'),curve.astype(np.float32))
   m=metrics(items);rec={'fusion':fusion,'spacing_sec':spacing,**m};rec['dev32']=metrics([x for d,v,x in tagged if (d,v) in dev]);rec['heldout']=metrics([x for d,v,x in tagged if (d,v) not in dev]);rec['datasets']={ds:metrics([x for d,v,x in tagged if d==ds]) for ds in ASR};result.append(rec)
 out=Path(cfg.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(sorted(result,key=lambda x:x['within_macro_roc'],reverse=True),indent=2)+'\n');print(json.dumps(sorted(result,key=lambda x:x['within_macro_roc'],reverse=True)[:10],indent=2))
if __name__=='__main__':main()
