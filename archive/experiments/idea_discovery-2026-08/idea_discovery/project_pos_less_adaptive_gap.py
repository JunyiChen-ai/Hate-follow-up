#!/usr/bin/env python3
"""Threshold-free PoS projection using each video's largest semantic gap."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from scripts.idea_discovery.project_pos_less import load_chunks,project,shifted


def natural_margin(values):
 finite=np.asarray(values,float)
 unique=np.unique(finite[np.isfinite(finite)])
 if len(unique)<2:return None
 gaps=np.diff(unique);return float(gaps[int(np.argmax(gaps))])


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 chunks=load_chunks();audit={'rows':0,'with_text':0,'changed':0,'edges':0,'degenerate':0}
 with a.base.open() as src,a.out.open('w') as dst:
  for row in map(json.loads,src):
   if row['method']!='fact_less_t3al_dualgeo_midpoint_v5':continue
   audit['rows']+=1;key=(row['dataset'],row['video_id']);records=chunks.get(key,[])
   p=np.clip(np.asarray(row['score_curve'],float),1e-5,1-1e-5);base=np.log(p/(1-p));spans=[];raw=[]
   for chunk in records:
    lo=max(0,min(len(p)-1,int(float(chunk['span'][0])*4)));hi=min(len(p),max(lo+1,int(math.ceil(float(chunk['span'][1])*4))))
    value=float(chunk.get('z_masked',chunk.get('z_isolated',-20)))
    if math.isfinite(value):spans.append((lo,hi));raw.append(value)
   conf=np.asarray(raw,float);margin=natural_margin(conf)
   if spans:audit['with_text']+=1
   if margin is None:audit['degenerate']+=1
   for control,values in [('aligned',conf),('circular_shift',shifted(conf,*key))]:
    if spans and margin is not None:score,n_edges,feasible,norm=project(base,spans,values,margin)
    else:score,n_edges,feasible,norm=base.copy(),0,True,0.
    posterior=1/(1+np.exp(-np.clip(score,-30,30)));changed=bool(np.max(np.abs(posterior-p))>1e-8)
    if control=='aligned':audit['changed']+=int(changed);audit['edges']+=n_edges
    out=dict(row);out['method']=f'pos_less_adaptive_gap_{control}_v1';out['score_curve']=posterior.tolist();out['raw']={**row.get('raw',{}),'gt_access':False,'module':'per_video_natural_gap_ordinal_projection','semantic_values':'raw_chunk_log_odds','gap_rule':'largest_adjacent_unique_value_gap','video_margin':margin,'n_edges':n_edges,'projection_norm':norm,'feasible':feasible,'exact_fallback':not changed,'dataset_parameters':0,'label_selected_parameters':0};dst.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
