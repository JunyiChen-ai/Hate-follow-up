#!/usr/bin/env python3
"""P7: prequential cross-video query memory with label-free write certification."""
from __future__ import annotations
import argparse,hashlib,json,math,sys,zlib
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,torch,open_clip
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_counterfactual_evidence import ASR,rows,transcript_rows,poset_curve
from scripts.idea_discovery.run_distributional_boundaries import canonical_curve
from scripts.idea_discovery.run_visual_temporal_canvas import valid_chunks
from scripts.idea_discovery.run_temporal_role_experts import tied_ecdf
from scripts.label_free_adapt.schema import Prediction,append_jsonl,curve_to_intervals

COHORT=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_8_clean.jsonl';T3AL=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_t3al_clean';TOPO=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_topology_clean'
FEAT=Path('/home/jehc223/Retrieval-hate/data/CLIP_Embedding');PROMPT='abusive or hateful behavior toward a person or group'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pairloss(M,q,pairs):
 s=M@q;return torch.stack([torch.nn.functional.softplus(-10*(s[i]-s[j])) for i,j in pairs]).mean()
def fit_query(qbase,M,pairs):
 q=qbase.clone().requires_grad_(True)
 if pairs:
  opt=torch.optim.Adam([q],lr=.03)
  for _ in range(30):
   qn=q/q.norm();loss=pairloss(M,qn,pairs)+.5*(1-qn@qbase);opt.zero_grad();loss.backward();opt.step()
 return (q/q.norm()).detach()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cohort',type=Path,default=COHORT);ap.add_argument('--t3al-curves',type=Path,default=T3AL);ap.add_argument('--topology-curves',type=Path,default=TOPO)
 ap.add_argument('--order-seed',type=int,required=True);ap.add_argument('--beta',type=float,default=.5);ap.add_argument('--memory-size',type=int,default=16);ap.add_argument('--write-margin',type=float,default=0.);a=ap.parse_args()
 if a.order_seed<0 or a.memory_size<1 or not math.isfinite(a.beta) or a.beta<0 or not math.isfinite(a.write_margin) or a.write_margin<0:
  raise ValueError('require order_seed>=0, memory_size>=1, finite beta>=0, and finite write_margin>=0')
 if a.out.exists():raise RuntimeError('P7 is prequential and fail-closed on existing output; use a new --out path')
 cohort=rows(a.cohort);byds=defaultdict(list)
 for r in cohort:byds[r['dataset']].append(r)
 chunks=transcript_rows();device='cuda';torch.manual_seed(a.order_seed)
 model,_,_=open_clip.create_model_and_transforms('coca_ViT-L-14',pretrained='mscoco_finetuned_laion2B-s13B-b90k');model=model.to(device).eval();tok=open_clip.get_tokenizer('coca_ViT-L-14');P=model.visual.proj.detach()
 q0=model.encode_text(tok(['a video of action '+PROMPT]).to(device)).detach()[0];q0=q0/q0.norm()
 config={'version':'p7_xvideo_memory_v1','cohort':str(a.cohort.resolve()),'t3al':str(a.t3al_curves.resolve()),'poset':str(a.topology_curves.resolve()),'order_seed':a.order_seed,'beta':a.beta,'memory_size':a.memory_size,'write_margin':a.write_margin,
  'transductive':True,'prequential':True,'encoder':'coca_ViT-L-14/mscoco_finetuned_laion2B-s13B-b90k','code_sha256':digest(__file__),'cohort_sha256':digest(a.cohort),
  'counterfactual_utility_sha256':digest(ROOT/'scripts/idea_discovery/run_counterfactual_evidence.py'),'distributional_utility_sha256':digest(ROOT/'scripts/idea_discovery/run_distributional_boundaries.py'),'canvas_utility_sha256':digest(ROOT/'scripts/idea_discovery/run_visual_temporal_canvas.py'),
  'role_utility_sha256':digest(ROOT/'scripts/idea_discovery/run_temporal_role_experts.py'),
  'asr_inputs':{d:{'path':str(p.resolve()),'sha256':digest(p)} for d,p in ASR.items()},
  'feature_root':str(FEAT.resolve()),'feature_extractor_sha256':digest('/home/jehc223/Retrieval-hate/scripts/repro_campaign/extract_coca_4fps.py'),
  'open_clip_module':str(Path(open_clip.__file__).resolve()),'open_clip_module_sha256':digest(open_clip.__file__),'python':sys.executable,
  't3al_run_sha256':digest(a.t3al_curves/'run.json'),'poset_run_sha256':digest(a.topology_curves/'run.json')}
 cid=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16];suffix=f'o{a.order_seed}';methods=(f't3al_memcontrol_{suffix}',f't3al_xmemory_{suffix}',f'poset_memcontrol_{suffix}',f'poset_xmemory_{suffix}');expected={(r['dataset'],r['video_id'],m) for r in cohort for m in methods};order_log=[]
 for ds,items in sorted(byds.items()):
  items=sorted(items,key=lambda x:x['video_id']);rng=np.random.default_rng(a.order_seed^zlib.crc32(ds.encode()));rng.shuffle(items);memory=[]
  for position,row in enumerate(items):
   v,dur=row['video_id'],float(row['duration']);n=max(1,int(np.floor(dur*4)));clean,rejected=valid_chunks(chunks.get((ds,v),[]),dur);fp=FEAT/ds/'coca_vitL14_4fps'/f'{v}.npy'
   if not fp.exists():raise RuntimeError(f'missing CoCa features {fp}')
   X=torch.tensor(np.load(fp),device=device)@P;X=X/X.norm(dim=1,keepdim=True).clamp_min(1e-8);means=[];z=[]
   for r in clean:
    lo=max(0,min(len(X)-1,int(np.floor(float(r['span'][0])*4))));hi=min(len(X),max(lo+1,int(np.ceil(float(r['span'][1])*4))));means.append(X[lo:hi].mean(0));z.append(float(r.get('z_masked',r.get('z_isolated',-20))))
   M=torch.stack(means) if means else torch.empty((0,len(q0)),device=device);conf=1/(1+np.exp(-np.clip(np.asarray(z)/4,-30,30)));train_nodes=[i for i in range(len(clean)) if i%2];held_nodes=[i for i in range(len(clean)) if not i%2]
   def pairs(nodes):
    ns=set(nodes);return [(i,j) for i in ns for j in ns if conf[i]-conf[j]>=.4]
   train,held=pairs(train_nodes),pairs(held_nodes);mem=torch.stack(memory).mean(0) if memory else torch.zeros_like(q0);qb=q0+a.beta*mem;qb=qb/qb.norm();qmem=fit_query(qb,M,train);qctrl=fit_query(q0,M,train)
   improve=0.
   if held:
    before=pairloss(M,qb,held);after=pairloss(M,qmem,held);improve=float(((before-after)/(before+1e-8)).cpu())
   write=bool(train) and bool(held) and improve>a.write_margin
   visual_mem=canonical_curve((X@qmem).cpu().numpy(),n);visual_ctrl=canonical_curve((X@qctrl).cpu().numpy(),n)
   t=canonical_curve(np.load(a.t3al_curves/ds/f'{v}.npy'),n);q=canonical_curve(np.load(a.topology_curves/ds/f'{v}.npy'),n);p=poset_curve(q,clean,dur)
   for base_name,base in [('t3al',t),('poset',p)]:
    for variant,visual in [('memcontrol',visual_ctrl),('xmemory',visual_mem)]:
     method=f'{base_name}_{variant}_{suffix}';curve=.75*tied_ecdf(base)+.25*tied_ecdf(visual);pred=Prediction(method,ds,v,dur,score_curve=curve.tolist(),intervals=curve_to_intervals(curve,dur,.75),calls=0,seed=a.order_seed,
      modality_evidence={'position':position,'memory_len_before':len(memory),'train_pairs':len(train),'held_pairs':len(held),'held_improvement':improve,'write_after':write,'invalid_asr_spans_rejected':rejected},raw={'config_id':cid,'config':config})
     append_jsonl(a.out,pred)
   if write:memory=(memory+[(qmem-qb).detach()])[-a.memory_size:]
   order_log.append({'dataset':ds,'video_id':v,'position':position,'write':write,'memory_len_after':len(memory)});print(json.dumps(order_log[-1]),flush=True)
 actual=rows(a.out);counts=Counter((r['dataset'],r['video_id'],r['method']) for r in actual);observed=set(counts)
 if expected-observed or observed-expected or any(counts[k]!=1 for k in expected):raise RuntimeError('coverage/duplicate failure')
 Path(str(a.out)+'.run.json').write_text(json.dumps({'config_id':cid,'config':config,'order':order_log},indent=2)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
