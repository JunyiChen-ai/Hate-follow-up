#!/usr/bin/env python3
"""Evaluate CPO-TTA and controls on the identical four-dataset test cohort."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from prequential_write_gate_pilot import ecdf, interp, metrics

ROOT=Path(__file__).resolve().parents[2]
GT=Path('/home/jehc223/Retrieval-hate/data/gt/frame_gt_4fps')
METHODS={
 'fixed':ROOT/'results/idea_discovery/cpo_tta/fixed',
 'repaired_t3al':ROOT/'results/idea_discovery/t3al_real/repaired_full2',
 'ungated':ROOT/'results/idea_discovery/cpo_tta/ungated',
 'shuffled':ROOT/'results/idea_discovery/cpo_tta/shuffled',
 'chunk_gate_0':ROOT/'results/idea_discovery/cpo_tta/chunk0',
 'chunk_gate_01':ROOT/'results/idea_discovery/cpo_tta/chunk1',
 'chunk_gate_02':ROOT/'results/idea_discovery/cpo_tta/chunk2',
 'reverse_ordinal':ROOT/'results/idea_discovery/cpo_tta/reverse',
 'chunk50_all':ROOT/'results/idea_discovery/cpo_tta/chunk50_all',
 'chunk50_gate_0':ROOT/'results/idea_discovery/cpo_tta/chunk50_0',
 'chunk50_gate_01':ROOT/'results/idea_discovery/cpo_tta/chunk50_01',
 'chunk50_shuffle':ROOT/'results/idea_discovery/cpo_tta/chunk50_shuffle',
 'chunk50_reverse':ROOT/'results/idea_discovery/cpo_tta/chunk50_reverse',
 'ocq_abuse':ROOT/'results/idea_discovery/prompt_query/abuse',
 'ocq_abuse_shuffle':ROOT/'results/idea_discovery/prompt_query/abuse_shuffle',
 'ocq_abuse_reverse':ROOT/'results/idea_discovery/prompt_query/abuse_reverse',
 'cpo_tta':ROOT/'results/idea_discovery/cpo_tta/main',
}
DATASETS=['HateMM','HateClipSeg','MHC','MHC_zh']

def bootstrap(diffs, seed=20260825, n=10000):
 d=np.asarray(diffs,float)
 if len(d)==0:return None
 rng=np.random.default_rng(seed); means=np.empty(n)
 for k in range(n): means[k]=rng.choice(d,len(d),replace=True).mean()
 return {'mean_delta':float(d.mean()),'ci95':[float(x) for x in np.quantile(means,[.025,.975])],
         'wins':int((d>0).sum()),'ties':int((d==0).sum()),'losses':int((d<0).sum()),'n':len(d)}

def main():
 result={'methods':{},'paired_within_bootstrap':{}}
 per_method={m:{} for m in METHODS}; per_auc={m:{} for m in METHODS}
 for ds in DATASETS:
  z=np.load(GT/(ds+'.npz'),allow_pickle=True)
  gt={str(v):np.asarray(y,int) for v,s,y in zip(z['video_ids'],z['split'],z['y4']) if str(s)=='test'}
  common=set(gt)
  for p in METHODS.values(): common&={x.stem for x in (p/ds).glob('*.npy')}
  for m,p in METHODS.items():
   items=[]; aucs={}
   for vid in sorted(common):
    y=gt[vid]; score=interp(ecdf(np.load(p/ds/(vid+'.npy'))),len(y));items.append((y,score))
    if np.unique(y).size==2:aucs[vid]=float(roc_auc_score(y,score))
   per_method[m][ds]=items;per_auc[m][ds]=aucs
  for m in METHODS:result['methods'].setdefault(m,{})[ds]=metrics(per_method[m][ds])
 for m in METHODS:
  all_items=sum((per_method[m][d] for d in DATASETS),[])
  result['methods'][m]['all']=metrics(all_items)
 target='ocq_abuse' if 'ocq_abuse' in METHODS else 'cpo_tta'
 for base in ['fixed','repaired_t3al']:
  result['paired_within_bootstrap'][base]={}
  for ds in DATASETS+['all']:
   datasets=DATASETS if ds=='all' else [ds];diff=[]
   for d in datasets:
    for vid,v in per_auc[target][d].items():diff.append(v-per_auc[base][d][vid])
   result['paired_within_bootstrap'][base][ds]=bootstrap(diff)
 out=ROOT/'results/idea_discovery/cpo_tta/eval.json';out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
