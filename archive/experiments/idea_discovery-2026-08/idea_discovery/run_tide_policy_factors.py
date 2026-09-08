#!/usr/bin/env python3
"""Continuous policy-factor fields inside the TIDE temporal firewall."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_melt import MLLM,sanitized_cohort
from scripts.idea_discovery.run_tide_u32 import bins_to_curve,carrier,positive_intervals,score_batches
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times,timed_text,valid_chunks
from scripts.label_free_adapt.schema import Prediction,append_jsonl

FACTORS={
 'target':'Only decide whether a person or protected/social group is an explicit or implicit target. Ignore hostility and author stance.',
 'hostile_act':'Only decide whether an attack, dehumanization, threat, exclusion, or hateful stereotype is expressed. Ignore target identity and whether the author endorses it.',
 'endorsement':'Only decide whether the current author or speaker adopts or endorses the central proposition rather than quoting, reporting, satirizing, condemning, or countering it. Ignore whether that proposition is hostile or targets a protected group.',
 'visual_hostility':'Only decide whether the center-cell visual elements express targeting or hostility. Ignore the transcript and author stance.',
}
def factor_prompt(i,records,factor):
 return (f'The red cell is the only cell to judge; its immediate neighbors are local context. '
         f'Timestamp-aligned local speech records={json.dumps(records,ensure_ascii=False)}. '
         f'Judge exactly one independent policy factor and do not require the other factors. '
         f'Policy factor={factor}. {FACTORS[factor]} Answer Yes or No only:')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct');ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 partial=Path(str(a.out)+'.partial')
 if a.out.exists() or partial.exists():raise RuntimeError('output or partial already exists')
 cohort=sanitized_cohort(a.cohort);cohort=cohort[:a.limit] if a.limit else cohort
 asr=transcript_rows();model=MLLM(a.model)
 for row in cohort:
  d,v,dur=row['dataset'],row['video_id'],float(row['duration']);chunks,rejected=valid_chunks(asr.get((d,v),[]),dur);speech=timed_text(chunks,dur,nbins=32)
  remaining=12000;budget=[]
  for text in speech:
   value=text[:remaining] if remaining>0 else '';budget.append(value);remaining-=len(value)
  records=[{'bin':i,'start':round(i*dur/32,3),'end':round((i+1)*dur/32,3),'text':x} for i,x in enumerate(budget) if x]
  frames,times,fallback=frames_at_times(row['video_path'],(np.arange(32)+.5)/32*dur)
  if len(frames)!=32:raise RuntimeError(f'decoded {len(frames)}/32 for {(d,v)}')
  fields={};before=model.calls
  for factor in FACTORS:
   actual_images=[];null_images=[];actual_prompts=[];null_prompts=[]
   for i in range(32):
    ids={max(0,i-1),i,min(31,i+1)};local=[r for r in records if r['bin'] in ids]
    actual_images.append(carrier(frames,i,ids));null_images.append(carrier(frames,i,set()))
    actual_prompts.append(factor_prompt(i,local,factor));null_prompts.append(factor_prompt(i,[],factor))
   fields[factor]=(score_batches(model,actual_images,actual_prompts,a.batch_size)-score_batches(model,null_images,null_prompts,a.batch_size))
  core=np.stack([fields[x] for x in ('target','hostile_act','endorsement')])
  # Continuous conjunction margin: every independent factor must be positive.
  score=core.min(0)
  p=1/(1+np.exp(-np.clip(score,-30,30)))
  append_jsonl(partial,Prediction('tide_policy_factors',d,v,dur,score_curve=bins_to_curve(p,dur).tolist(),intervals=positive_intervals(score,dur),calls=model.calls-before,modality_evidence={'factor_log_odds':{k:x.tolist() for k,x in fields.items()},'combined_log_odds':score.tolist(),'invalid_asr_spans_rejected':rejected,'ffmpeg_fallback_frames':fallback},raw={'semantic_queries':8*32,'gt_access':False}))
  print(json.dumps({'dataset':d,'video_id':v,'calls':model.calls-before}),flush=True)
 rows=[json.loads(x) for x in partial.open()];keys={(r['dataset'],r['video_id']) for r in rows}
 if len(rows)!=len(cohort) or len(keys)!=len(cohort):raise RuntimeError('invalid partial result cardinality')
 partial.replace(a.out)
if __name__=='__main__':
 try:main()
 except Exception:
  if '--out' in sys.argv:
   i=sys.argv.index('--out');Path(sys.argv[i+1]+'.partial').unlink(missing_ok=True)
  raise
