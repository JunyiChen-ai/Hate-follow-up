#!/usr/bin/env python3
"""Holistic native multimodal existence judge for a frozen temporal closure."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_melt import MLLM,sanitized_cohort
from scripts.idea_discovery.run_ncl_tribunal import POLICY,PHASES,binary_logits,carrier,complement_intervals,load,overlap_text,uniform_times
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times,valid_chunks

QUESTION=("Does the TARGET interval contain sufficient evidence of one complete asserted or endorsed hateful event? "
          "Use context only to determine speaker stance. Answer Yes or No only:")
def prompt(arm,target,context):
 if arm=='visual':media='No transcript is supplied. Judge the shown native frames only.'
 elif arm=='text':media=f'TARGET transcript={json.dumps(target,ensure_ascii=False)}. Context transcript={json.dumps(context,ensure_ascii=False)}. No frames are supplied.'
 else:media=f'TARGET transcript={json.dumps(target,ensure_ascii=False)}. Context transcript={json.dumps(context,ensure_ascii=False)}. Use the native frames and aligned transcript jointly.'
 return f'{POLICY} {media} {QUESTION}'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',type=Path,required=True);ap.add_argument('--extent',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True);ap.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct');ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 a.out.parent.mkdir(parents=True,exist_ok=True);cohort=sanitized_cohort(a.cohort);cohort=cohort[:a.limit] if a.limit else cohort
 extent,chunks,model=load(a.extent),transcript_rows(),MLLM(a.model)
 config={'version':'ncl_holistic_v1','model':a.model,'phases':list(PHASES),'gt_access':False,
  'cohort_sha256':hashlib.sha256(a.cohort.read_bytes()).hexdigest(),'extent_sha256':hashlib.sha256(a.extent.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 with a.out.open('x',encoding='utf8') as f:
  for i,row in enumerate(cohort,1):
   key=(row['dataset'],row['video_id']);duration=float(row['duration']);iv=extent[key]['intervals'];start=min(float(x[0]) for x in iv);end=max(float(x[1]) for x in iv)
   clean,rejected=valid_chunks(chunks.get(key,[]),duration);outside=complement_intervals(start,end,duration)
   if not outside:
    w=max(duration/16,1e-3);outside=[(0,min(w,duration)),(max(0,duration-w),duration)]
   target=overlap_text(clean,start,end);context=' '.join(overlap_text(clean,x,y,800) for x,y in outside)[:1600]
   obs=[]
   for phase in PHASES:
    tt=uniform_times([(start,end)],4,phase);ct=uniform_times(outside,4,phase)
    tf,_,a_fb=frames_at_times(Path(row['video_path']),tt);cf,_,b_fb=frames_at_times(Path(row['video_path']),ct);image=carrier(tf,cf,tt,ct)
    for arm in ('visual','text','joint'):
     z=float(binary_logits(model,None if arm=='text' else [image],[prompt(arm,target,context)],a.batch_size)[0])
     obs.append({'phase':phase,'arm':arm,'log_odds':z})
   f.write(json.dumps({'dataset':key[0],'video_id':key[1],'duration':duration,'closure':[start,end],
    'observations':obs,'target_text_chars':len(target),'context_text_chars':len(context),'invalid_asr_spans_rejected':rejected,'config':config},ensure_ascii=False)+'\n');f.flush()
   print(json.dumps({'i':i,'n':len(cohort),'key':key,'scores':{x['arm']+str(x['phase']):x['log_odds'] for x in obs}}),flush=True)
 print(json.dumps({'videos':len(cohort),'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest(),'calls':model.calls}))
if __name__=='__main__':main()
