#!/usr/bin/env python3
"""Frozen label-free Relation-Envelope Localizer confirmation runner."""
from __future__ import annotations
import argparse,hashlib,json,math,random,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_melt import MLLM,evidence_bins,parse_relation,sanitized_cohort
from scripts.idea_discovery.run_role_binding import BINARY_RELATION_CHOICES,binary_relation_logits,generic_decode
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times,make_canvas,timed_text,valid_chunks
from scripts.label_free_adapt.mechanisms import parse_json_object
from scripts.label_free_adapt.schema import Interval,Prediction,append_jsonl

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def prompt(records):
 return ("The image is a numbered 16-bin video timeline and speech is aligned by bin: "+json.dumps(records,ensure_ascii=False)+". Parse one directed hateful event relation. Hate requires an asserted or endorsed attack, dehumanization, threat, exclusion, or hateful stereotype against a protected or socially salient group; reporting, quotation, counterspeech, satire and condemnation are not endorsement. Unknown fields must be 'unknown'. If no valid hateful relation exists, stance must be 'none'. Return JSON only: {\"event_relation\":{\"source\":\"...\",\"hostile_act\":\"...\",\"protected_target\":\"...\",\"stance\":\"endorsement|quotation|condemnation|ambiguous|none\"}}.")
def envelope(intervals,duration,radius=1):
 out=[]
 for x in intervals:
  a=max(0,x.start-duration/16*radius);b=min(duration,x.end+duration/16*radius)
  out.append(Interval(a,b,x.score))
 # deterministic merge after receptive-field expansion
 merged=[]
 for x in sorted(out,key=lambda z:z.start):
  if merged and x.start<=merged[-1].end:
   p=merged[-1];merged[-1]=Interval(p.start,max(p.end,x.end),max(p.score,x.score))
  else:merged.append(x)
 return merged
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct');ap.add_argument('--seed',type=int,default=20260828);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 cohort=sanitized_cohort(a.cohort);asr=transcript_rows();model=MLLM(a.model)
 config={'method':'RelationEnvelope-v1','cohort_sha256':sha(a.cohort),'runner_sha256':sha(__file__),'model':a.model,'seed':a.seed,'bins':16,'context_radius_bins':1,'phase_space':list(BINARY_RELATION_CHOICES),'gt_access':False}
 cid=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
 for row in cohort:
  d,v,duration=row['dataset'],row['video_id'],float(row['duration']);n4=max(1,math.floor(duration*4));before=model.calls
  chunks,invalid=valid_chunks(asr.get((d,v),[]),duration);speech=timed_text(chunks,duration)
  records=[{'bin':i,'start':round(i*duration/16,3),'end':round((i+1)*duration/16,3),'text':x} for i,x in enumerate(speech) if x]
  try:
   ims,times,fallback=frames_at_times(row['video_path'],(np.arange(16)+.5)/16*duration);canvas=make_canvas(ims,times,None)
   raw=model.infer(canvas,prompt(records),max_new_tokens=300);obj=parse_json_object(raw);relation=parse_relation(obj['event_relation'])
   scores=binary_relation_logits(model,canvas,records,relation)
   if relation['stance'] not in {'endorsement','ambiguous'}:
    scores[:]=0;scores[:,BINARY_RELATION_CHOICES.index('NOT_FULL')]=1
   coarse,margin,paths=generic_decode(scores,duration);intervals=envelope(coarse,duration,1)
   curve=np.asarray(scores[:,0])[np.minimum(15,((np.arange(n4)+.5)/n4*16).astype(int))]
   ev={'event_relation':relation,'relation_completion_scores':scores.tolist(),'coarse_paths':paths,'context_radius_bins':1,'invalid_asr_spans_rejected':invalid,'ffmpeg_fallback_frames':fallback,'generate_calls':1,'forward_batches':1,'semantic_queries':17}
   pred=Prediction('RelationEnvelope',d,v,duration,score_curve=curve.tolist(),intervals=intervals,calls=model.calls-before,seed=a.seed,modality_evidence=ev,raw={'config_id':cid,'config':config,'parser_response':raw})
  except Exception as exc:
   # Fail closed but keep the video in metric denominators.
   pred=Prediction('RelationEnvelope',d,v,duration,score_curve=[0.0]*n4,intervals=[],calls=model.calls-before,seed=a.seed,modality_evidence={'fail_closed':True},raw={'config_id':cid,'config':config},error=None)
   pred.raw['failure']=f'{type(exc).__name__}: {exc}'
  append_jsonl(a.out,pred);print(json.dumps({'dataset':d,'video_id':v,'intervals':len(pred.intervals),'failure':pred.raw.get('failure')}),flush=True)
if __name__=='__main__':main()
