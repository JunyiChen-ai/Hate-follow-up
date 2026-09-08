#!/usr/bin/env python3
"""Multimodal constrained duel between rank-1 and closure extents (GT blind)."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from PIL import ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_melt import MLLM,sanitized_cohort
from scripts.idea_discovery.run_ncl_tribunal import load
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times,make_canvas,timed_text,valid_chunks

POLICY=("Hateful content is an asserted or endorsed attack, dehumanization, threat, exclusion, or hateful stereotype directed at a person or protected/social group. Quotation, neutral reporting, counterspeech, satire, and condemnation are not hateful unless endorsed.")
def decorate(images,times,duration,rank1,closure):
 c=make_canvas(images,times,None,'',16);draw=ImageDraw.Draw(c);y0=4*156+18;w=c.width
 for y,(a,b),color,name in [(y0+15,closure,'blue','B closure'),(y0+48,rank1,'red','A rank1')]:
  draw.text((4,y-14),name,fill=color);draw.line((int(a/duration*w),y,int(b/duration*w),y),fill=color,width=12)
 return c
@torch.inference_mode()
def choices(model,images,prompts,batch=8):
 out=[]
 for s in range(0,len(images),batch):
  ims=images[s:s+batch];ps=prompts[s:s+batch];texts=[]
  for im,p in zip(ims,ps):texts.append(model.processor.apply_chat_template([{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':p}]}],tokenize=False,add_generation_prompt=True))
  x=model.processor(text=texts,images=ims,padding=True,return_tensors='pt').to(model.model.device);z=model.model(**x,use_cache=False,logits_to_keep=1).logits[:,-1].float();out.extend(z[:,model.phase_token_ids[:3]].cpu().tolist());model.calls+=1
 return np.asarray(out)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',type=Path,required=True);ap.add_argument('--rank1',type=Path,required=True);ap.add_argument('--closure',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct');ap.add_argument('--batch-size',type=int,default=8);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 a.out.parent.mkdir(parents=True,exist_ok=True);cohort=sanitized_cohort(a.cohort);r1,cl,chunks=load(a.rank1),load(a.closure),transcript_rows();model=MLLM(a.model);examples=[];meta=[]
 for row in cohort:
  k=(row['dataset'],row['video_id']);d=float(row['duration']);ra,rb=map(float,r1[k]['intervals'][0][:2]);ca,cb=map(float,cl[k]['intervals'][0][:2]);frac=(rb-ra)/d
  if not (.5<=frac<.7):continue
  clean,rejected=valid_chunks(chunks.get(k,[]),d);speech=timed_text(clean,d);records=[{'bin':i,'start':round(i*d/16,2),'end':round((i+1)*d/16,2),'text':x} for i,x in enumerate(speech) if x]
  prompt=(f"{POLICY} The image is a 16-bin chronological video canvas. Red A=[{ra:.2f},{rb:.2f}] is a tight rank-1 proposal; blue B=[{ca:.2f},{cb:.2f}] is its broader connected closure. Timestamp-aligned transcript={json.dumps(records,ensure_ascii=False)}. Select the minimal interval that still contains the complete asserted/endorsed hateful event: A if red contains all event evidence and blue adds irrelevant context; B if evidence necessary to the same event lies outside red but inside blue; C if no hateful event is supported or the boundary choice is genuinely indeterminate. Proposal geometry is a hypothesis, not evidence. Choose exactly one next letter A, B, or C:")
  for phase in (.25,.75):
   times=(np.arange(16)+phase)/16*d;ims,_,fallback=frames_at_times(Path(row['video_path']),times);examples.append(decorate(ims,times,d,(ra,rb),(ca,cb)));meta.append({'dataset':k[0],'video_id':k[1],'duration':d,'phase':phase,'rank1':[ra,rb],'closure':[ca,cb],'rank1_fraction':frac,'ffmpeg_fallback_frames':fallback,'invalid_asr_spans_rejected':rejected,'prompt':prompt})
 logits=choices(model,examples,[x['prompt'] for x in meta],a.batch_size);config={'version':'extent_duel_v1','model':a.model,'gt_access':False,'cohort_sha256':hashlib.sha256(a.cohort.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 with a.out.open('x',encoding='utf8') as f:
  for m,z in zip(meta,logits):
   m.pop('prompt');winner='ABC'[int(np.argmax(z))];f.write(json.dumps({**m,'logits':dict(zip('ABC',map(float,z))),'winner':winner,'config':config},ensure_ascii=False)+'\n')
 print(json.dumps({'videos':len({(x["dataset"],x["video_id"]) for x in meta}),'rows':len(meta),'winners':{x:sum('ABC'[int(np.argmax(z))]==x for z in logits) for x in 'ABC'},'sha256':hashlib.sha256(a.out.read_bytes()).hexdigest(),'calls':model.calls}))
if __name__=='__main__':main()
