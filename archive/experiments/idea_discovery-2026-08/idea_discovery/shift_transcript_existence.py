#!/usr/bin/env python3
"""Deterministically cross-video shift transcript expert outputs within dataset."""
import argparse,json
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()];out=[]
 for d in sorted({r['dataset'] for r in rows}):
  z=sorted([r for r in rows if r['dataset']==d],key=lambda r:r['video_id']);values=[r['log_odds'] for r in z]
  for i,r in enumerate(z):q=dict(r);q['log_odds']=values[(i+1)%len(values)];q['positive']=q['log_odds']>0;q['shift_control']='next video_id within dataset';out.append(q)
 with a.out.open('x') as f:
  for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 print(json.dumps({'n':len(out)}))
if __name__=='__main__':main()
