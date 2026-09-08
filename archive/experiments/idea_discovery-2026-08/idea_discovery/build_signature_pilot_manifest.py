#!/usr/bin/env python3
"""Deterministically select eight nonempty videos per dataset for signature pilot."""
import argparse,hashlib,json
from pathlib import Path

def load(path,method=None):
 out={}
 for l in Path(path).open():
  r=json.loads(l)
  if method is None or r.get('method')==method:out[(r['dataset'],r['video_id'])]=r
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',type=Path,required=True);ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--method',default='fact_less_t3al_dualgeo_shorter_v5');ap.add_argument('--per-dataset',type=int,default=8);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 cohort=load(a.cohort);bank=load(a.bank,a.method);chosen=[]
 for ds in ('HateMM','HateClipSeg','MHC','MHC_zh'):
  rows=[(hashlib.sha256(f'cohere-signature-v1::{ds}::{key[1]}'.encode()).hexdigest(),key,row) for key,row in cohort.items() if key[0]==ds and key in bank and bank[key].get('intervals')]
  for _,key,row in sorted(rows)[:a.per_dataset]:
   out=dict(row);out['selection']={'rule':'sha256 cohere-signature-v1','gt_access':False,'tight_interval':bank[key]['intervals'][0][:2]};chosen.append(out)
 a.out.write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in chosen));print(json.dumps({'n':len(chosen),'by_dataset':{d:sum(x['dataset']==d for x in chosen) for d in ('HateMM','HateClipSeg','MHC','MHC_zh')}},indent=2))
if __name__=='__main__':main()
