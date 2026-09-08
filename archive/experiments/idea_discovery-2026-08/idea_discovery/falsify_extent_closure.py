#!/usr/bin/env python3
"""Length and stability falsification suite for proposal extent closure."""
from __future__ import annotations
import argparse,hashlib,json,math,random
from pathlib import Path
import numpy as np
from scripts.label_free_adapt.evaluate import binary_intervals,interval_f1,temporal_iou

DATASETS=('HateMM','HateClipSeg','MHC','MHC_zh')
def hs(k):return 'dev' if int(hashlib.sha256(f'mice-falsify-v1\0{k[0]}\0{k[1]}'.encode()).hexdigest(),16)%2==0 else 'holdout'
def comps(ps):
 ps=sorted((float(p['start']),float(p['end'])) for p in ps);groups=[]
 for a,b in ps:
  if groups and a<=groups[-1][1]:groups[-1][1]=max(groups[-1][1],b)
  else:groups.append([a,b])
 return groups
def closure(row,k,anchored=True):
 ps=row['proposals'][:k];groups=comps(ps)
 if not anchored:return [min(x[0] for x in groups),max(x[1] for x in groups),1.]
 c=.5*(ps[0]['start']+ps[0]['end']);g=next((x for x in groups if x[0]<=c<=x[1]),groups[0]);return [g[0],g[1],1.]
def same_length(row,target):
 p=row['proposals'][0];d=float(row['duration']);length=target[1]-target[0];center=.5*(p['start']+p['end']);a=center-length/2;b=center+length/2
 if a<0:b-=a;a=0.
 if b>d:a-=b-d;b=d
 return [max(0.,a),min(d,b),1.]
def expanded(row,factor):
 p=row['proposals'][0];d=float(row['duration']);length=(p['end']-p['start'])*factor;return same_length(row,[0,length,1.])
def rand_hull(row,k):
 seed=int(hashlib.sha256(f"random-hull-v1\0{row['dataset']}\0{row['video_id']}".encode()).hexdigest(),16);r=random.Random(seed);ps=r.sample(row['proposals'][:16],k);return [min(p['start'] for p in ps),max(p['end'] for p in ps),1.]
def evalm(keys,y,preds):
 per={}
 for d in DATASETS:
  kd=[k for k in keys if k[0]==d];per[d]=interval_f1({k:y[k] for k in kd},{k:{'intervals':[preds[k]]} for k in kd})
 macro={m:float(np.mean([per[d][m] for d in DATASETS])) for m in ('interval_F1@0.3','interval_F1@0.5','interval_F1@0.7')};return macro,per
def endpoint(keys,y,preds):
 vals=[]
 for k in keys:
  targets=binary_intervals(y[k]);p=preds[k]
  if not targets:continue
  g=max(targets,key=lambda x:temporal_iou(tuple(p[:2]),x));d=float(len(y[k])/4);vals.append(((p[0]-g[0])/d,(p[1]-g[1])/d,(p[1]-p[0])/d,(g[1]-g[0])/d))
 if not vals:return {}
 z=np.asarray(vals);return {'n':len(z),'median_start_error_frac':float(np.median(z[:,0])),'median_end_error_frac':float(np.median(z[:,1])),'median_pred_length_frac':float(np.median(z[:,2])),'median_gt_length_frac':float(np.median(z[:,3])),'mean_abs_start_error_frac':float(np.mean(abs(z[:,0]))),'mean_abs_end_error_frac':float(np.mean(abs(z[:,1])))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--proposals',type=Path,required=True);ap.add_argument('--gt-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();rows={(r['dataset'],r['video_id']):r for r in map(json.loads,a.proposals.open())};y={}
 for d in DATASETS:
  z=np.load(a.gt_dir/f'{d}.npz',allow_pickle=True)
  for i,v in enumerate(z['video_ids']):
   if (d,str(v)) in rows:y[(d,str(v))]=np.asarray(z['y4'][i],np.int8)
 keys=sorted(set(rows)&set(y));parts={s:[k for k in keys if hs(k)==s] for s in ('dev','holdout')};ratios=[(closure(rows[k],8)[1]-closure(rows[k],8)[0])/(rows[k]['proposals'][0]['end']-rows[k]['proposals'][0]['start']) for k in parts['dev']];factor=float(np.median(ratios));policies={}
 for k in (1,2,4,8,16):policies[f'closure_K{k}']={x:closure(rows[x],k) for x in keys}
 policies['unanchored_hull_K8']={x:closure(rows[x],8,False) for x in keys};policies['same_length_symmetric_K8']={x:same_length(rows[x],policies['closure_K8'][x]) for x in keys};policies['global_median_symmetric']={x:expanded(rows[x],factor) for x in keys};policies['longest_K8']={x:list(max(rows[x]['proposals'][:8],key=lambda p:p['end']-p['start'])[q] for q in ('start','end'))+[1.] for x in keys};policies['random_hull_8of16']={x:rand_hull(rows[x],8) for x in keys}
 result={'n':len(keys),'dev_median_expansion_factor':factor,'split_sizes':{s:len(v) for s,v in parts.items()},'policies':{}}
 for name,p in policies.items():
  result['policies'][name]={}
  for s,ks in parts.items():
   m,per=evalm(ks,y,p);result['policies'][name][s]={'macro':m,'per_dataset':per,'endpoint':endpoint(ks,y,p)}
 base=result['policies']['closure_K1']['holdout']['macro']['interval_F1@0.5'];gain=result['policies']['closure_K8']['holdout']['macro']['interval_F1@0.5']-base;sg=result['policies']['same_length_symmetric_K8']['holdout']['macro']['interval_F1@0.5']-base;result['same_length_gain_fraction']=sg/gain if gain else None
 a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'factor':factor,'same_length_gain_fraction':result['same_length_gain_fraction'],'holdout':{n:v['holdout']['macro'] for n,v in result['policies'].items()},'endpoint':{n:v['holdout']['endpoint'] for n,v in result['policies'].items() if n in ('closure_K1','closure_K8','same_length_symmetric_K8')}},indent=2))
if __name__=='__main__':main()
