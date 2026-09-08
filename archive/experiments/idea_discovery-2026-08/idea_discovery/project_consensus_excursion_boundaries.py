#!/usr/bin/env python3
"""Parameter-free peak-component boundary readouts for a corrected dense field."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
def component(mask,peak):
 if not mask[peak]:return None
 a=peak;b=peak+1
 while a>0 and mask[a-1]:a-=1
 while b<len(mask) and mask[b]:b+=1
 return a,b
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--method',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 audit={x:{'n':0,'fallback':0} for x in ('median','mean','midrange')}
 with a.out.open('w') as f:
  for r in map(json.loads,a.input.open()):
   if r['method']!=a.method:continue
   s=np.asarray(r['score_curve'],float);peak=int(np.argmax(s));stats={'median':float(np.median(s)),'mean':float(np.mean(s)),'midrange':float((np.median(s)+np.max(s))/2)}
   for name,threshold in stats.items():
    audit[name]['n']+=1;c=component(s>=threshold,peak);out=dict(r);out['method']=f'consensus_excursion_{name}_v1'
    if c is None or c[1]<=c[0]:audit[name]['fallback']+=1
    else:
     start,end=c;duration=float(r['duration']);fps=len(s)/duration;out['intervals']=[[start/fps,min(duration,end/fps),float(np.mean(s[start:end]))]]
    out['raw']={**r.get('raw',{}),'gt_access':False,'boundary_readout':'peak_anchored_excursion_component','threshold_statistic':name,'threshold':threshold,'fallback':c is None}
    f.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
