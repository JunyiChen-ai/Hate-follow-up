#!/usr/bin/env python3
"""Core-oriented multimodal kernel change-point boundary pilot."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
DATA_DIR={"HateMM":"hatemm","HateClipSeg":"hateclipseg","MHC":"mhclip_en","MHC_zh":"mhclip_zh"}
FEATURES={"V":"clip_b16_1fps","A":"vggish_1s","T":"bert_sentence_1fps"}
def load(path,method):
 out={}
 for l in Path(path).open():
  r=json.loads(l)
  if r.get('method')==method:out[(r['dataset'],r['video_id'])]=r
 return out
def interval(r):return tuple(map(float,r['intervals'][0][:2])) if r and r.get('intervals') else None
def norm(x):x=np.asarray(x,float);return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)
def rank(x):
 o=np.argsort(x,kind='stable');r=np.empty(len(x));r[o]=np.arange(len(x));return (r+.5)/len(x)
def boundary(curves,tight,broad,side):
 ts,te=map(int,map(round,tight));bs,be=map(int,map(round,broad));n=min(map(len,curves.values()));w=max(2,int(round(max(2.,tight[1]-tight[0])*.1)))
 if side==0:lo=max(w,min(bs,ts)-w);hi=min(n-w,max(ts,bs)+w)
 else:lo=max(w,min(te,be)-w);hi=min(n-w,max(te,be)+w)
 candidates=np.arange(lo,hi+1)
 if len(candidates)==0:return tight[side],{}
 stats={}
 for m,c in curves.items():
  vals=[]
  for t in candidates:
   left=float(np.mean(c[t-w:t]));right=float(np.mean(c[t:t+w]));vals.append(right-left if side==0 else left-right)
  stats[m]=np.asarray(vals)
 # Each view has equal ordinal authority; negative signed changes remain low rank.
 score=np.mean(np.stack([rank(x) for x in stats.values()]),axis=0)
 ix=int(np.argmax(score));return float(candidates[ix]),{'window':w,'candidates':candidates.tolist(),'selected_index':ix,'signed_change':{m:x.tolist() for m,x in stats.items()},'consensus':score.tolist()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--base-method',required=True);ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--tight-method',default='fact_less_t3al_dualgeo_shorter_v5');ap.add_argument('--broad-method',default='fact_less_t3al_dualgeo_union_v5');ap.add_argument('--feature-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 base=load(a.base,a.base_method);tight=load(a.bank,a.tight_method);broad=load(a.bank,a.broad_method);subsets={'V':'V','AT':'AT','VAT':'VAT'};changed={k:0 for k in subsets}
 with a.out.open('w') as h:
  for key,row in sorted(base.items()):
   ti,br=interval(tight.get(key)),interval(broad.get(key));raw={}
   if ti and br:
    vectors={m:norm(np.load(a.feature_root/folder/DATA_DIR[key[0]]/(key[1]+'.npy'))) for m,folder in FEATURES.items()}
    lo=max(0,int(np.floor(ti[0])));hi=max(lo+1,int(np.ceil(ti[1])));curves={m:x@(x[lo:min(hi,len(x))].mean(0)/max(np.linalg.norm(x[lo:min(hi,len(x))].mean(0)),1e-9)) for m,x in vectors.items()}
   else:curves={}
   for name,mods in subsets.items():
    out=dict(row);out['method']=f'core_kernel_boundary_{name}_v1';chosen=interval(row);audit={}
    if ti and br and all(m in curves for m in mods):
     left,la=boundary({m:curves[m] for m in mods},ti,br,0);right,ra=boundary({m:curves[m] for m in mods},ti,br,1);audit={'left':la,'right':ra}
     if left<right and left<=ti[0] and right>=ti[1]:chosen=(left,right);changed[name]+=int(chosen!=interval(row))
    out['intervals']=[] if chosen is None else [[chosen[0],chosen[1],1.0]];out['raw']={**out.get('raw',{}),'gt_access':False,'boundary_decoder':'core_oriented_multimodal_kernel_change','modalities':list(mods),'audit':audit};h.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps({'n':len(base),'changed':changed},indent=2))
if __name__=='__main__':main()
