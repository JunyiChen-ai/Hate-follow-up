#!/usr/bin/env python3
"""Compose cohort-level broad jurisdiction with video-level core appeals."""
import argparse,json
from pathlib import Path

def load(path, method=None):
 out={}
 for r in map(json.loads,Path(path).open()):
  if method is None or r['method']==method:out[(r['dataset'],r['video_id'])]=r
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--core',type=Path,required=True);p.add_argument('--core-method',required=True);p.add_argument('--casa',type=Path,required=True);p.add_argument('--score-source',type=Path);p.add_argument('--score-method');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 core=load(a.core,a.core_method);casa=load(a.casa);scores=load(a.score_source,a.score_method) if a.score_source else {};counts={'broad_regime':0,'core_regime':0}
 with a.out.open('w') as f:
  for key,row in sorted(core.items()):
   regime=casa[key].get('modality_evidence',{}).get('cohort_regime',{})
   broad=bool(regime.get('positive_dominant',False));out=dict(row)
   if key in scores:out['score_curve']=scores[key]['score_curve']
   if broad:out['intervals']=casa[key]['intervals'];counts['broad_regime']+=1
   else:counts['core_regime']+=1
   out['method']='pos_less_regime_appeal_v1'
   out['raw']={**row.get('raw',{}),'boundary_paradigm':'cohort_broad_jurisdiction_then_video_core_appeal','broad_authority':broad,'cohort_regime':regime,'gt_access':False}
   f.write(json.dumps(out,separators=(',',':'))+'\n')
 print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
