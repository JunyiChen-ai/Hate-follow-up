#!/usr/bin/env python3
"""Certificate-strength ablation for the tribunal's semantic-core jurisdiction."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--tribunal',type=Path,required=True);p.add_argument('--tribunal-method',default='pos_less_tribunal_aligned_rank_sum_v1');p.add_argument('--name-prefix',default='pos_less_corecert');p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 base={(r['dataset'],r['video_id']):r for r in map(json.loads,a.base.open()) if r['method']=='fact_less_t3al_dualgeo_midpoint_v5'}
 rows=[r for r in map(json.loads,a.tribunal.open()) if r['method']==a.tribunal_method]
 configs=[(p,z) for p in (0.0625,0.125,0.1875,0.25,0.5,1.0) for z in (0.0,0.5,1.0,2.0)]
 counts={}
 with a.out.open('w') as f:
  for row in rows:
   key=(row['dataset'],row['video_id']);raw=row['raw'];rule=raw['selected_rule']
   for alpha,zmin in configs:
    accept=rule=='t3al' and float(raw['alignment_p'])<=alpha and float(raw['alignment_surplus'])>=zmin
    name=f'{a.name_prefix}_p{int(alpha*10000):04d}_z{int(zmin*10):02d}_v1'
    out=dict(row);out['method']=name
    if not accept:out['intervals']=base[key]['intervals']
    counts[name]=counts.get(name,0)+int(accept)
    out['raw']={**raw,'extent_jurisdiction':'certified_semantic_core_only','certificate_alpha':alpha,'certificate_zmin':zmin,'core_edit_applied':accept}
    f.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
