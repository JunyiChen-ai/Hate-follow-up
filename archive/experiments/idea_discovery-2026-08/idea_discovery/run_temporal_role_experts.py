#!/usr/bin/env python3
"""P6: frozen semantic/proposal/boundary role experts with label-free reliability routing."""
from __future__ import annotations
import argparse,hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_counterfactual_evidence import rows,transcript_rows,poset_curve,ecdf
from scripts.idea_discovery.run_distributional_boundaries import canonical_curve
from scripts.idea_discovery.run_visual_temporal_canvas import dense_bins,valid_chunks
from scripts.label_free_adapt.schema import Prediction,append_jsonl,curve_to_intervals

COHORT=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_8_clean.jsonl';P2=ROOT/'results/idea_discovery/paradigm_adapt/p2_stage_a_8.jsonl'
TIMELENS=ROOT/'results/idea_discovery/paradigm_adapt/p6_timelens_stage_a_8.jsonl';T3AL=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_t3al_clean';TOPO=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_topology_clean'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def tied_ecdf(x):
 x=np.asarray(x,float);n=len(x)
 if n<=1 or np.ptp(x)<=1e-8:return np.full(n,.5)
 _,inv,count=np.unique(x,return_inverse=True,return_counts=True);before=np.r_[0,np.cumsum(count)[:-1]]
 return (before[inv]+(count[inv]-1)/2)/(n-1)

def route(base,semantic,boundary):
 # Scale-invariant dispersion makes raw logits and [0,1] experts comparable.
 streams=[np.asarray(base,float),np.asarray(semantic,float),np.asarray(boundary,float)]
 dispersion=np.asarray([float(np.std(x))/(float(np.mean(np.abs(x)))+float(np.std(x))+1e-8) for x in streams])
 informative=dispersion>=.02
 expert_curves=[tied_ecdf(base),streams[1],streams[2]]
 expert_curves=[x if ok else np.full(len(x),.5) for x,ok in zip(expert_curves,informative)]
 reliability=np.where(informative,dispersion,0.);weights=reliability/reliability.sum() if reliability.sum()>0 else np.ones(3)/3
 curve=sum(w*x for w,x in zip(weights,expert_curves))
 return curve,weights

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cohort',type=Path,default=COHORT);ap.add_argument('--p2',type=Path,default=P2);ap.add_argument('--timelens',type=Path,default=TIMELENS)
 ap.add_argument('--t3al-curves',type=Path,default=T3AL);ap.add_argument('--topology-curves',type=Path,default=TOPO);a=ap.parse_args()
 cohort=rows(a.cohort);p2rows=rows(a.p2);p2={(r['dataset'],r['video_id'],r['method']):r for r in p2rows};tlrows=rows(a.timelens);tl={(r['dataset'],r['video_id']):r for r in tlrows};chunks=transcript_rows();methods=('t3al_role_experts','poset_role_experts')
 p2ids={r.get('raw',{}).get('config_id') for r in p2rows}
 if len(p2ids)!=1 or None in p2ids:raise RuntimeError(f'P2 input must have one config: {p2ids}')
 config={'version':'p6_role_experts_v1','cohort':str(a.cohort.resolve()),'p2':str(a.p2.resolve()),'timelens':str(a.timelens.resolve()),'t3al':str(a.t3al_curves.resolve()),'poset':str(a.topology_curves.resolve()),
  'roles':{'proposal':'T3AL_or_Poset','semantic':'Qwen3-VL-8B_P2','boundary':'TimeLens-8B'},'router':'scale-invariant std/(mean_abs+std); <.02 -> neutral/zero weight; tie-safe proposal ECDF',
  'code_sha256':digest(__file__),'cohort_sha256':digest(a.cohort),'p2_sha256':digest(a.p2),'timelens_sha256':digest(a.timelens),'p2_config_id':next(iter(p2ids)),
  't3al_run_sha256':digest(a.t3al_curves/'run.json'),'poset_run_sha256':digest(a.topology_curves/'run.json'),
  'counterfactual_utility_sha256':digest(ROOT/'scripts/idea_discovery/run_counterfactual_evidence.py'),
  'distributional_utility_sha256':digest(ROOT/'scripts/idea_discovery/run_distributional_boundaries.py'),
  'canvas_utility_sha256':digest(ROOT/'scripts/idea_discovery/run_visual_temporal_canvas.py'),
  'timelens_runner_sha256':digest(ROOT/'scripts/label_free_adapt/run_timelens.py')}
 cid=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16];prior=rows(a.out) if a.out.exists() else []
 if any(r.get('raw',{}).get('config_id')!=cid for r in prior):raise RuntimeError('output config mismatch; use new --out')
 done={(r['dataset'],r['video_id'],r['method']) for r in prior};expected={(r['dataset'],r['video_id'],m) for r in cohort for m in methods}
 for row in cohort:
  d,v,dur=row['dataset'],row['video_id'],float(row['duration']);n=max(1,int(np.floor(dur*4)));clean,_=valid_chunks(chunks.get((d,v),[]),dur)
  t=canonical_curve(np.load(a.t3al_curves/d/f'{v}.npy'),n);q=canonical_curve(np.load(a.topology_curves/d/f'{v}.npy'),n);p=poset_curve(q,clean,dur);tr=tl.get((d,v))
  if tr is None:raise RuntimeError(f'missing TimeLens expert {d}/{v}')
  expected_manifest_sha=digest(a.cohort);raw=tr.get('raw',{})
  if tr.get('method')!='A12_TimeLens8B' or tr.get('error') is not None or abs(float(tr.get('duration',-1))-dur)>1e-6 or float(tr.get('native_rate',-1))!=4.0:
   raise RuntimeError(f'invalid TimeLens record identity/schema {d}/{v}')
  if raw.get('model_id')!='TencentARC/TimeLens-8B' or not raw.get('model_revision') or raw.get('manifest_sha256')!=expected_manifest_sha or raw.get('runner_sha256')!=config['timelens_runner_sha256']:
   raise RuntimeError(f'unverified TimeLens provenance {d}/{v}')
  boundary=canonical_curve(tr.get('score_curve',[]),n)
  if len(boundary)!=n or not np.all(np.isfinite(boundary)):raise RuntimeError(f'invalid TimeLens curve {d}/{v}')
  for base_name,base,p2method in [('t3al',t,'t3al_canvas'),('poset',p,'poset_canvas')]:
   method=f'{base_name}_role_experts'
   if (d,v,method) in done:continue
   sr=p2.get((d,v,p2method));scores=sr.get('modality_evidence',{}).get('bin_scores',[]) if sr else []
   if len(scores)!=16:raise RuntimeError(f'invalid semantic expert {d}/{v}/{p2method}')
   ids=np.clip((np.arange(n)/4/dur*16).astype(int),0,15);semantic=np.asarray(scores,float)[ids]/100
   curve,weights=route(base,semantic,boundary);intervals=curve_to_intervals(curve,dur,.75)
   p2_calls=1 if sr.get('calls') is None else int(sr['calls']);tl_calls=1 if tr.get('calls') is None else int(tr['calls'])
   if p2_calls<0 or tl_calls<0:raise RuntimeError(f'invalid call count {d}/{v}')
   calls=p2_calls+tl_calls;pred=Prediction(method,d,v,dur,score_curve=curve.tolist(),intervals=intervals,calls=calls,seed=int(tr.get('seed',20250819)),
    modality_evidence={'router_weights':{'proposal':float(weights[0]),'semantic':float(weights[1]),'boundary':float(weights[2])},'semantic_bins':scores,'timelens_intervals':tr.get('intervals',[])},
    raw={'config_id':cid,'config':config,'timelens_method':tr.get('method'),'legacy_null_timelens_calls_interpreted_as_one':tr.get('calls') is None})
   append_jsonl(a.out,pred);print(json.dumps({'dataset':d,'video_id':v,'method':method,'weights':weights.tolist(),'intervals':len(intervals)}),flush=True)
 counts=Counter((r['dataset'],r['video_id'],r['method']) for r in rows(a.out));observed=set(counts)
 if expected-observed or observed-expected or any(counts[k]!=1 for k in expected):raise RuntimeError('coverage/duplicate failure')
 return 0
if __name__=='__main__':raise SystemExit(main())
