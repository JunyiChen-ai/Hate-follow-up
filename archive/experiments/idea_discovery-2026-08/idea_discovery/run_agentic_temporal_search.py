#!/usr/bin/env python3
"""P3: label-free agentic coarse-to-fine temporal search over P2 canvases."""
from __future__ import annotations
import argparse,hashlib,json,math,re,sys
from collections import Counter
from pathlib import Path
import numpy as np,torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_counterfactual_evidence import rows,transcript_rows,poset_curve,ecdf
from scripts.idea_discovery.run_visual_temporal_canvas import (frames_at_times,make_canvas,timed_text,valid_chunks,
 dense_bins,bin_means,infer)
from scripts.label_free_adapt.mechanisms import parse_json_object
from scripts.label_free_adapt.schema import Prediction,append_jsonl,curve_to_intervals

COHORT=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_8_clean.jsonl'
T3AL=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_t3al_clean'
TOPO=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_topology_clean'
P2=ROOT/'results/idea_discovery/paradigm_adapt/p2_stage_a_8.jsonl'

def choose_window(base,coarse,duration,width=4):
 proposal=bin_means(ecdf(base),16);evidence=np.asarray(coarse,float)/100
 signal=.5*proposal+.5*evidence
 sums=np.convolve(signal,np.ones(width),mode='valid');start=int(np.argmax(sums))
 return start/16*duration,(start+width)/16*duration,start,signal.tolist()

def local_timed_text(chunks,lo,hi,nbins=16):
 out=[[] for _ in range(nbins)]
 for r in chunks:
  a,b=map(float,r['span']);text=str(r.get('text',''))
  units=[ch for ch in text if not ch.isspace()] if re.search(r'[\u3400-\u9fff]',text) else text.split()
  for k,unit in enumerate(units):
   wt=a+(k+.5)/len(units)*(b-a)
   if lo<=wt<hi:
    i=min(nbins-1,max(0,int((wt-lo)/(hi-lo)*nbins)));out[i].append(unit)
 return [' '.join(x) for x in out]

def parse_scores(raw):
 try:value=parse_json_object(raw).get('scores',{})
 except (ValueError,TypeError):return []
 scores=[value.get(str(i)) for i in range(16)] if isinstance(value,dict) else []
 return scores

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cohort',type=Path,default=COHORT)
 ap.add_argument('--t3al-curves',type=Path,default=T3AL);ap.add_argument('--topology-curves',type=Path,default=TOPO)
 ap.add_argument('--p2',type=Path,default=P2);ap.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct');ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 from transformers import AutoModelForImageTextToText,AutoProcessor
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True,max_pixels=896*800)
 model=AutoModelForImageTextToText.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cuda:0',local_files_only=True,attn_implementation='sdpa').eval()
 methods=('t3al_agent_search','poset_agent_search');cohort=rows(a.cohort);cohort=cohort[:a.limit] if a.limit>0 else cohort
 p2_rows=rows(a.p2);p2={(r['dataset'],r['video_id'],r['method']):r for r in p2_rows};chunks=transcript_rows()
 p2_ids={r.get('raw',{}).get('config_id') for r in p2_rows}
 if len(p2_ids)!=1 or None in p2_ids:raise RuntimeError(f'P2 input must have one non-null config_id, got {p2_ids}')
 p2_models={r.get('raw',{}).get('config',{}).get('model') for r in p2_rows}
 if p2_models!={a.model}:raise RuntimeError(f'P2 model mismatch: {p2_models} != {a.model}')
 def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
 config={'version':'p3_agent_search_v1','model':a.model,'cohort':str(a.cohort.resolve()),'p2':str(a.p2.resolve()),
  't3al':str(a.t3al_curves.resolve()),'poset':str(a.topology_curves.resolve()),'code_sha256':digest(__file__),
  'canvas_utility_sha256':digest(ROOT/'scripts/idea_discovery/run_visual_temporal_canvas.py'),'cohort_sha256':digest(a.cohort),
  'p2_sha256':digest(a.p2),'p2_config_id':next(iter(p2_ids)),
  't3al_run_sha256':digest(a.t3al_curves/'run.json'),'poset_run_sha256':digest(a.topology_curves/'run.json')}
 config_id=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
 prior=rows(a.out) if a.out.exists() else []
 if any(r.get('raw',{}).get('config_id')!=config_id for r in prior):raise RuntimeError('output contains a different/missing config; use a new --out path')
 done={(r['dataset'],r['video_id'],r['method']) for r in prior};expected={(r['dataset'],r['video_id'],m) for r in cohort for m in methods}
 for row in cohort:
  d,v,dur=row['dataset'],row['video_id'],float(row['duration']);clean,rejected=valid_chunks(chunks.get((d,v),[]),dur)
  t=np.load(a.t3al_curves/d/f'{v}.npy');q=np.load(a.topology_curves/d/f'{v}.npy');p=poset_curve(q,clean,dur)
  for base_name,base,coarse_method in [('t3al',t,'t3al_canvas'),('poset',p,'poset_canvas')]:
   method=f'{base_name}_agent_search'
   if (d,v,method) in done:continue
   coarse_row=p2.get((d,v,coarse_method))
   if coarse_row is None:raise RuntimeError(f'missing P2 record: {d}/{v}/{coarse_method}')
   coarse=coarse_row.get('modality_evidence',{}).get('bin_scores',[])
   if len(coarse)!=16:raise RuntimeError(f'invalid P2 scores: {d}/{v}/{coarse_method}')
   lo,hi,start,selection=choose_window(base,coarse,dur)
   ims,times,fallback=frames_at_times(row['video_path'],(np.arange(16)+.5)/16*(hi-lo)+lo)
   speech=local_timed_text(clean,lo,hi)
   speech_records=[{'bin':i,'start':round(lo+i*(hi-lo)/16,2),'end':round(lo+(i+1)*(hi-lo)/16,2),'text':x} for i,x in enumerate(speech) if x]
   centers=(np.arange(len(base))+.5)/len(base);inside=(centers>=start/16)&(centers<(start+4)/16)
   selected=np.flatnonzero(inside)
   if not len(selected):raise RuntimeError(f'empty zoom support: {d}/{v}/{method}')
   i0=int(selected[0]);i1=int(selected[-1]+1)
   canvas=make_canvas(ims,times,base[i0:i1],base_name.upper()+' ZOOM')
   prompt=('Act as a temporal search agent. A coarse first pass selected this zoom window because its multimodal evidence and noisy proposal were most suspicious. '
    'The image contains 16 ordered frames from only this window and one local proposal curve. Timestamped speech is '+json.dumps(speech_records,ensure_ascii=False)+'. '
    'Localize evidence, not the whole-video label. Direct protected-target slurs, attacks, threats, dehumanization, exclusion, or hateful stereotypes score 70-100; context-dependent hostility 30-69; benign 0-10. '
    'Do not spread evidence to neighboring bins unless those bins independently support it. Reporting, quotation, counterspeech, satire, and condemnation should be lowered when the local context makes that stance clear. '
    'Return JSON only with every key: {"scores":{"0":integer,"1":integer,"2":integer,"3":integer,"4":integer,"5":integer,"6":integer,"7":integer,"8":integer,"9":integer,"10":integer,"11":integer,"12":integer,"13":integer,"14":integer,"15":integer}}.')
   attempts=[];scores=[]
   for k in range(2):
    retry='' if k==0 else ' Previous JSON invalid. Return exactly keys 0 through 15 with finite numeric values.'
    raw=infer(canvas,prompt+retry,proc,model);attempts.append(raw);scores=parse_scores(raw)
    if len(scores)==16 and all(isinstance(x,(int,float)) and math.isfinite(float(x)) for x in scores):break
   if len(scores)!=16 or not all(isinstance(x,(int,float)) and math.isfinite(float(x)) for x in scores):raise RuntimeError(f'invalid zoom scores: {d}/{v}/{method}: {attempts!r}')
   evidence=dense_bins(np.asarray(coarse,float)/100,len(base))
   evidence[inside]=dense_bins(np.asarray(scores,float)/100,int(inside.sum()));curve=.75*ecdf(base)+.25*np.clip(evidence,0,1)
   calls=int(coarse_row.get('calls',1))+len(attempts)
   pred=Prediction(method,d,v,dur,score_curve=curve.tolist(),intervals=curve_to_intervals(curve,dur,.75),calls=calls,
    modality_evidence={'coarse_scores':coarse,'zoom_scores':scores,'zoom_span':[lo,hi],'selection_signal':selection,'invalid_asr_spans_rejected':rejected,'ffmpeg_fallback_frames':fallback},
    raw={'responses':attempts,'config_id':config_id,'config':config})
   append_jsonl(a.out,pred);print(json.dumps({'dataset':d,'video_id':v,'method':method,'zoom':[lo,hi],'scores':scores}),flush=True)
 counts=Counter((r['dataset'],r['video_id'],r['method']) for r in rows(a.out));observed=set(counts);missing=expected-observed;unexpected=observed-expected
 if missing or unexpected or any(counts[k]!=1 for k in expected):raise RuntimeError(f'coverage failure missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}')
 return 0
if __name__=='__main__':raise SystemExit(main())
