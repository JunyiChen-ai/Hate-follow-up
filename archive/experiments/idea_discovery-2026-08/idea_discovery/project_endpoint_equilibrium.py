#!/usr/bin/env python3
"""Parameter-free endpoint equilibrium around a single-source anchor span."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np


def component(mask,peak):
 if not mask[peak]:return None
 lo=peak;hi=peak+1
 while lo and mask[lo-1]:lo-=1
 while hi<len(mask) and mask[hi]:hi+=1
 return lo,hi


def decode(scores,interval,duration,reverse=False):
 n=len(scores);field=scores[::-1] if reverse else scores
 start,end=map(float,interval[:2]);li=int(np.clip(round(start/duration*(n-1)),0,n-1));ri=int(np.clip(round(end/duration*(n-1)),0,n-1))
 a,b=sorted((li,ri));peak=a+int(np.argmax(field[a:b+1]));threshold=float((field[li]+field[ri])/2)
 found=component(field>=threshold,peak)
 if found is None:return interval,{"fallback":"no_component"}
 lo,hi=found
 if reverse:lo,hi=n-hi,n-lo
 candidate=[lo/n*duration,hi/n*duration,1.0]
 return (candidate if candidate[1]>candidate[0] else interval),{"fallback":None,"endpoint_level":threshold,"anchor_indices":[li,ri],"peak_index":peak,"component_indices":[lo,hi],"field_reversed":reverse}


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--method',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 with a.out.open('w') as h:
  for row in map(json.loads,a.input.open()):
   if row.get('method')!=a.method:continue
   scores=np.asarray(row['score_curve'],float);base=row.get('intervals',[])
   for reverse,name in [(False,'endpoint_equilibrium_v1'),(True,'endpoint_equilibrium_reverse_v1')]:
    out=dict(row);out['method']=name
    if base:chosen,audit=decode(scores,base[0],float(row['duration']),reverse)
    else:chosen=[];audit={'fallback':'empty_anchor'}
    out['intervals']=[] if not chosen else [chosen]
    out['raw']={**out.get('raw',{}),'gt_access':False,'module':'single_anchor_endpoint_equilibrium','dataset_parameters':0,'label_selected_parameters':0,'boundary_audit':audit}
    h.write(json.dumps(out,separators=(',',':'))+'\n')
if __name__=='__main__':main()
