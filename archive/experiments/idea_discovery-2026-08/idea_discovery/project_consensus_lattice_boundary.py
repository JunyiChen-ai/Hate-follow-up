#!/usr/bin/env python3
"""Use the corrected consensus field to traverse a fixed LESS endpoint lattice."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
METHODS={'tight':'fact_less_t3al_dualgeo_shorter_v5','midpoint':'fact_less_t3al_dualgeo_midpoint_v5','broad':'fact_less_t3al_dualgeo_union_v5'}
def load(path,method=None):return {(r['dataset'],r['video_id']):r for r in map(json.loads,Path(path).open()) if method is None or r['method']==method}
def iv(r):return tuple(map(float,r['intervals'][0][:2])) if r and r.get('intervals') else None
def span_mean(s,d,a,b):
 lo=max(0,min(len(s)-1,int(np.floor(a/d*len(s)))));hi=min(len(s),max(lo+1,int(np.ceil(b/d*len(s)))));return float(np.mean(s[lo:hi]))
def endpoint(field,d,intervals,side,rule):
 idx=0 if side=='left' else 1;tight=intervals['tight'];values=[intervals[x][idx] for x in ('tight','midpoint','broad')];nodes=sorted(set(values),reverse=(side=='left'))
 start=tight[idx];nodes.remove(start);nodes.insert(0,start);current=start;audit=[];background=float(np.median(field))
 for target in nodes[1:]:
  if side=='left':shell=(target,current);core=(current,tight[1])
  else:shell=(current,target);core=(tight[0],current)
  if shell[1]<=shell[0]:continue
  sm=span_mean(field,d,*shell);reference=background if rule=='median' else span_mean(field,d,*core);expand=sm>=reference;audit.append({'from':current,'to':target,'shell_mean':sm,'reference':reference,'expand':bool(expand)})
  if not expand:break
  current=target
 return current,audit
def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--field',type=Path,required=True);p.add_argument('--field-method',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 bank={m:load(a.bank,name) for m,name in METHODS.items()};fields=load(a.field,a.field_method);audit={r:{'n':0,'changed':0,'fallback':0} for r in ('median','core')}
 with a.out.open('w') as f:
  for key,row in sorted(bank['midpoint'].items()):
   intervals={m:iv(rows.get(key)) for m,rows in bank.items()};fr=fields.get(key)
   for rule in ('median','core'):
    audit[rule]['n']+=1;out=dict(fr if fr else row);out['method']=f'consensus_lattice_{rule}_v1'
    if fr is None or fr.get('raw',{}).get('exact_fallback',False) or any(x is None for x in intervals.values()):audit[rule]['fallback']+=1;out['intervals']=row['intervals'];la=ra=[]
    else:
     s=np.asarray(fr['score_curve'],float);d=float(row['duration']);left,la=endpoint(s,d,intervals,'left',rule);right,ra=endpoint(s,d,intervals,'right',rule);chosen=[left,right,1.0] if right>left else row['intervals'][0];out['intervals']=[chosen];audit[rule]['changed']+=int(list(map(float,row['intervals'][0][:2]))!=list(map(float,chosen[:2])))
    out['raw']={**out.get('raw',{}),'gt_access':False,'boundary_readout':'consensus_endpoint_lattice','shell_rule':rule,'left_audit':la,'right_audit':ra}
    f.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
