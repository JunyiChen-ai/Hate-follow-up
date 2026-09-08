#!/usr/bin/env python3
"""Select a label-blind disjoint confirmation cohort by fixed SHA order."""
import argparse,hashlib,json
from pathlib import Path

ALLOWED={"dataset","duration","transcript","video_id","video_path"}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--exclude',type=Path,required=True);ap.add_argument('--per-dataset',type=int,default=16);ap.add_argument('--salt',default='relation-envelope-confirm-v1');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 ex={(r['dataset'],r['video_id']) for r in (json.loads(x) for x in a.exclude.open())}
 groups={}
 for line in a.source.open():
  r=json.loads(line);extra=set(r)-ALLOWED
  if extra: raise RuntimeError(f'unsafe source fields: {extra}')
  if (r['dataset'],r['video_id']) not in ex: groups.setdefault(r['dataset'],[]).append(r)
 chosen=[]
 for ds,rows in sorted(groups.items()):
  rows.sort(key=lambda r:hashlib.sha256(f"{a.salt}/{ds}/{r['video_id']}".encode()).hexdigest())
  chosen.extend(rows[:min(len(rows),a.per_dataset)])
 with a.out.open('x') as f:
  for r in chosen:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(json.dumps({'count':len(chosen),'by_dataset':{d:sum(r['dataset']==d for r in chosen) for d in groups},'salt':a.salt}))
if __name__=='__main__':main()
