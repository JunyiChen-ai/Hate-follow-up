#!/usr/bin/env python3
"""GT-blind rule-family ablation for tribunal boundary edits."""
import argparse, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--tribunal',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists(): raise RuntimeError(f'refusing existing output: {a.out}')
    base={(r['dataset'],r['video_id']):r for r in map(json.loads,a.base.open()) if r['method']=='fact_less_t3al_dualgeo_midpoint_v5'}
    rows=[r for r in map(json.loads,a.tribunal.open()) if r['method']=='pos_less_tribunal_aligned_rank_sum_v1']
    families={'core':{'t3al'},'outer':{'union'},'core_outer':{'t3al','union'},'closure':{'cca'},'all':{'cca','t3al','union','casa'},'core_expand':{'t3al'},'core_contract':{'t3al'}}
    counts={k:0 for k in families}
    with a.out.open('w') as f:
      for row in rows:
       key=(row['dataset'],row['video_id']); rule=row['raw']['selected_rule']
       for name,allowed in families.items():
        out=dict(row);out['method']=f'pos_less_tribunal_{name}_v1'
        accept=rule in allowed
        if accept and name.startswith('core_') and name not in {'core_outer'}:
         bl=sum(float(x[1])-float(x[0]) for x in base[key]['intervals']);cl=sum(float(x[1])-float(x[0]) for x in row['intervals'])
         accept=(cl>=bl) if name=='core_expand' else (cl<bl)
        if not accept: out['intervals']=base[key]['intervals']
        else: counts[name]+=1
        out['raw']={**row['raw'],'allowed_rule_family':sorted(allowed),'rule_edit_applied':accept}
        f.write(json.dumps(out,separators=(',',':'))+'\n')
    print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
