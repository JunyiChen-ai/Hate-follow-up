#!/usr/bin/env python3
"""P5: risk-layered set-valued temporal localization over P4 posteriors."""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_counterfactual_evidence import rows
from scripts.label_free_adapt.schema import Prediction,append_jsonl,curve_to_intervals

COHORT=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_8_clean.jsonl';P4=ROOT/'results/idea_discovery/paradigm_adapt/p4_stage_a_8.jsonl'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def interval_lists(curve,duration,threshold):return [x.as_list() for x in curve_to_intervals(curve,duration,threshold)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cohort',type=Path,default=COHORT);ap.add_argument('--p4',type=Path,default=P4)
 ap.add_argument('--possible-threshold',type=float,default=.25);ap.add_argument('--certain-threshold',type=float,default=.75);a=ap.parse_args()
 if not (math.isfinite(a.possible_threshold) and math.isfinite(a.certain_threshold) and 0<=a.possible_threshold<=a.certain_threshold<=1):
  raise ValueError('thresholds must satisfy finite 0 <= possible <= certain <= 1')
 cohort=rows(a.cohort);source_rows=rows(a.p4);source={(r['dataset'],r['video_id'],r['method']):r for r in source_rows};methods=('t3al_set_valued','poset_set_valued')
 seeds={int(r['seed']) for r in source_rows}
 if len(seeds)!=1:raise RuntimeError(f'P4 input must have one seed, got {seeds}')
 ids={r.get('raw',{}).get('config_id') for r in source_rows}
 if len(ids)!=1 or None in ids:raise RuntimeError(f'P4 input must have one config, got {ids}')
 config={'version':'p5_set_valued_v1','cohort':str(a.cohort.resolve()),'p4':str(a.p4.resolve()),'possible_threshold':a.possible_threshold,'certain_threshold':a.certain_threshold,
  'code_sha256':digest(__file__),'cohort_sha256':digest(a.cohort),'p4_sha256':digest(a.p4),'p4_config_id':next(iter(ids))}
 cid=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16];prior=rows(a.out) if a.out.exists() else []
 if any(r.get('raw',{}).get('config_id')!=cid for r in prior):raise RuntimeError('output config mismatch; use new --out')
 done={(r['dataset'],r['video_id'],r['method']) for r in prior};expected={(r['dataset'],r['video_id'],m) for r in cohort for m in methods}
 for row in cohort:
  d,v,dur=row['dataset'],row['video_id'],float(row['duration']);n=max(1,int(np.floor(dur*4)));ids4=np.clip((np.arange(n)/4/dur*16).astype(int),0,15)
  for base in ('t3al','poset'):
   method=f'{base}_set_valued'
   if (d,v,method) in done:continue
   src=source.get((d,v,f'{base}_boundary_dist'))
   if src is None:raise RuntimeError(f'missing P4 source {d}/{v}/{base}')
   bins=np.asarray(src.get('modality_evidence',{}).get('bin_active_posterior',[]),float)
   if len(bins)!=16:raise RuntimeError(f'invalid P4 posterior {d}/{v}/{base}')
   post=bins[ids4];possible=curve_to_intervals(post,dur,a.possible_threshold);certain=curve_to_intervals(post,dur,a.certain_threshold)
   unknown=((post>=a.possible_threshold)&(post<a.certain_threshold)).astype(float)
   pred=Prediction(method,d,v,dur,score_curve=list(map(float,src['score_curve'])),intervals=possible,calls=int(src.get('calls',0)),seed=int(src['seed']),
    modality_evidence={'possible_intervals':[x.as_list() for x in possible],'certain_intervals':[x.as_list() for x in certain],
     'uncertain_intervals':interval_lists(unknown,dur,.5),'bin_active_posterior':bins.tolist()},raw={'config_id':cid,'config':config})
   append_jsonl(a.out,pred);print(json.dumps({'dataset':d,'video_id':v,'method':method,'possible':len(possible),'certain':len(certain)}),flush=True)
 counts=Counter((r['dataset'],r['video_id'],r['method']) for r in rows(a.out));observed=set(counts)
 if expected-observed or observed-expected or any(counts[k]!=1 for k in expected):raise RuntimeError('coverage/duplicate failure')
 return 0
if __name__=='__main__':raise SystemExit(main())
