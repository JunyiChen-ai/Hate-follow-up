#!/usr/bin/env python3
"""Combine a genuinely video-level latent posterior with a local label model."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.label_free_adapt.schema import Prediction,append_jsonl

def load(path,method):
 return {(r['dataset'],r['video_id']):r for r in map(json.loads,path.open()) if r['method']==method}
def logit(x):
 x=np.clip(np.asarray(x,float),1e-6,1-1e-6);return np.log(x/(1-x))
def main():
 p=argparse.ArgumentParser();p.add_argument('--global-model',type=Path,required=True);p.add_argument('--global-method',required=True);p.add_argument('--local-model',type=Path,required=True);p.add_argument('--local-method',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 g=load(a.global_model,a.global_method);l=load(a.local_model,a.local_method);keys=sorted(set(g)&set(l))
 for key in keys:
  gp=float(g[key]['modality_evidence']['global_posterior']);rz=logit(l[key]['score_curve']);rz-=rz.mean();curve=1/(1+np.exp(-np.clip(logit([gp])[0]+rz,-40,40)))
  append_jsonl(a.out,Prediction('less_twolevel_global_stronglocal_v1',key[0],key[1],float(g[key]['duration']),score_curve=curve.tolist(),intervals=[],calls=0,modality_evidence={'global_posterior':gp,'global_source':a.global_method,'local_source':a.local_method,'local_residual_mean':float(rz.mean())},raw={'gt_access':False,'model':'scale_specific_global_and_local_truth_models'}))
 print(json.dumps({'n':len(keys)}))
if __name__=='__main__':main()
