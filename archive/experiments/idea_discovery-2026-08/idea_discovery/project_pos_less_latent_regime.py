#!/usr/bin/env python3
"""Dataset-blind latent-regime boundary jurisdiction for PoS-LESS."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load(path,method=None):
 out={}
 for r in map(json.loads,Path(path).open()):
  if method is None or r['method']==method:out[(r['dataset'],r['video_id'])]=r
 return out
def length(row):return sum(float(x[1])-float(x[0]) for x in row.get('intervals',[]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--core',type=Path,required=True);p.add_argument('--core-method',required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--casa',type=Path,required=True);p.add_argument('--clusters',type=int,default=3);p.add_argument('--exclude-consensus-features',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 core=load(a.core,a.core_method);base=load(a.base,'fact_less_t3al_dualgeo_midpoint_v5');casa=load(a.casa);keys=sorted(set(core)&set(base)&set(casa))
 feats=[]
 for k in keys:
  r=core[k];c=casa[k];e=c.get('modality_evidence',{});curve=np.asarray(r['score_curve'],float);d=float(r['duration']);bl=length(base[k]);cl=length(c)
  geom=[np.log1p(d),bl/max(d,1e-6),cl/max(d,1e-6),cl/max(bl,1e-6),float(np.mean(curve)),float(np.std(curve)),float(np.quantile(curve,.9)-np.quantile(curve,.1))]
  consensus=[int(bool(e.get('a08_nonempty'))),int(bool(e.get('a12_nonempty'))),abs(int(bool(e.get('a08_nonempty')))-int(bool(e.get('a12_nonempty'))))]
  feats.append(geom if a.exclude_consensus_features else geom+consensus)
 X=StandardScaler().fit_transform(np.asarray(feats));labels=KMeans(n_clusters=a.clusters,random_state=20260828,n_init=20).fit_predict(X)
 regimes={}
 for cluster in range(a.clusters):
  idx=np.flatnonzero(labels==cluster);pp=nn=0
  for i in idx:
   e=casa[keys[i]].get('modality_evidence',{});u=bool(e.get('a08_nonempty'));v=bool(e.get('a12_nonempty'));pp+=int(u and v);nn+=int(not u and not v)
  pv=float(binomtest(pp,pp+nn,.5,alternative='greater').pvalue) if pp+nn else 1.;regimes[cluster]={'n':len(idx),'positive':pp,'negative':nn,'pvalue':pv,'positive_dominant':pv<.05 and pp>nn}
 with a.out.open('w') as f:
  for i,k in enumerate(keys):
   row=dict(core[k]);reg=regimes[int(labels[i])];broad=reg['positive_dominant']
   if broad:row['intervals']=casa[k]['intervals']
   row['method']=f'pos_less_latent_regime_k{a.clusters}_v1';row['raw']={**row.get('raw',{}),'dataset_identity_access':False,'latent_cluster':int(labels[i]),'latent_regime':reg,'broad_authority':broad,'gt_access':False}
   f.write(json.dumps(row,separators=(',',':'))+'\n')
 print(json.dumps({'clusters':regimes,'broad_videos':sum(regimes[int(x)]['positive_dominant'] for x in labels)},indent=2))
if __name__=='__main__':main()
