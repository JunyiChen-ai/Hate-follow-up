#!/usr/bin/env python3
"""Timestamped MLLM chunk discoveries as sparse multi-span proposals."""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
def load(path,method):
 out={}
 for l in Path(path).open():
  r=json.loads(l)
  if r.get('method')==method:out[(r['dataset'],r['video_id'])]=r
 return out
def merge(parts,gap=.5):
 out=[]
 for a,b in sorted(parts):
  if out and a-out[-1][1]<=gap:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return [[a,b,1.] for a,b in out]
def overlap(a,b):return max(0,min(a[1],b[1])-max(a[0],b[0]))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--base-method',required=True);ap.add_argument('--chunks',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 base=load(a.base,a.base_method);chunks=defaultdict(list)
 for l in a.chunks.open():
  r=json.loads(l);chunks[(r['dataset'],r['video_id'])].append(r)
 methods=('chunk_multispan_positive_v1','chunk_multispan_surplus_v1','base_plus_positive_chunks_v1');counts={m:0 for m in methods}
 with a.out.open('w') as h:
  for key,row in sorted(base.items()):
   existing=row.get('intervals',[]);positive=merge([(float(x['start']),float(x['end'])) for x in chunks.get(key,[]) if float(x['log_odds'])>0]);surplus=merge([(float(x['start']),float(x['end'])) for x in chunks.get(key,[]) if float(x['log_odds'])>-12])
   additional=[x for x in positive if not any(overlap(x,b[:2])>.5*(x[1]-x[0]) for b in existing)];variants={methods[0]:positive or existing,methods[1]:surplus or existing,methods[2]:existing+additional if existing else []}
   for m in methods:
    out=dict(row);out['method']=m;out['intervals']=variants[m];out['raw']={**out.get('raw',{}),'gt_access':False,'decoder':'timestamped_chunk_multispan','threshold':'zero_log_odds' if 'surplus' not in m else 'empty_null_surplus','chunk_count':len(chunks.get(key,[]))};counts[m]+=sum(len(variants[m])>1 for _ in [0]);h.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps({'n':len(base),'multispan_videos':counts},indent=2))
if __name__=='__main__':main()
