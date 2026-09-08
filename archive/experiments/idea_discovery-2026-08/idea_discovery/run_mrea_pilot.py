#!/usr/bin/env python3
"""MREA: fixed-length, modality-robust left/right extent allocation pilot."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_melt import MLLM,sanitized_cohort
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times,valid_chunks
from scripts.idea_discovery.run_hyper_tournament import sample_regions,text_for,nearest
from scripts.label_free_adapt.schema import Interval,Prediction,append_jsonl,intervals_to_curve

FACTOR=1.4899859333664718;METHODS=('mrea_robust','mrea_joint','mrea_mean','mrea_geometry','mrea_symmetric')
def allocate(row,a):
 p=row['proposals'][0];s,e,d=float(p['start']),float(p['end']),float(row['duration']);delta=(e-s)*(FACTOR-1);x=max(0.,s-a*delta);y=min(d,e+(1-a)*delta);target=min(d,(e-s)*FACTOR)
 if y-x<target:
  if x<=1e-9:y=min(d,x+target)
  elif y>=d-1e-9:x=max(0.,y-target)
 return [x,y]
def geometry_a(row):
 p=row['proposals'][0];bank=row['proposals'][:8];l=max(0.,p['start']-min(x['start'] for x in bank));r=max(0.,max(x['end'] for x in bank)-p['end']);raw=.5 if l+r<=1e-8 else l/(l+r);return min((.25,.5,.75),key=lambda x:abs(x-raw))
def card(anchor,left,right,frames,times,order,visual=True):
 rows=[('ANCHOR',[anchor]),('FIRST',[left if order==0 else right]),('SECOND',[right if order==0 else left])];im=Image.new('RGB',(784,396),(118,118,118));draw=ImageDraw.Draw(im)
 for ri,(label,regions) in enumerate(rows):
  ts=sample_regions(regions,4)
  for j,t in enumerate(ts):
   x,y=j*196,ri*132;pic=nearest(frames,times,t).resize((196,110)) if visual else Image.new('RGB',(196,110),(127,127,127));im.paste(pic,(x,y));draw.text((x+3,y+3),f'{label} {t:.1f}s',fill='yellow',stroke_width=2,stroke_fill='black')
  draw.text((4,ri*132+112),f'{label} [{regions[0][0]:.1f},{regions[0][1]:.1f})',fill='white')
 return im
@torch.inference_mode()
def logits(model,images,prompts):
 msgs=[[{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':p}]}] for im,p in zip(images,prompts)];texts=[model.processor.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in msgs];inp=model.processor(text=texts,images=images,padding=True,return_tensors='pt').to(model.model.device);z=model.model(**inp,use_cache=False,logits_to_keep=1).logits[:,-1].float()[:,model.phase_token_ids[:3]].cpu().numpy();model.calls+=1;return z
def verdict(z0,z1):
 s=np.array([(z0[0]+z1[1])/2,(z0[1]+z1[0])/2,(z0[2]+z1[2])/2]);j=int(np.argmax(s));return ('L' if j==0 else 'R' if j==1 else 'U'),s.tolist()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',type=Path,required=True);ap.add_argument('--proposals',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct');a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 cohort=sanitized_cohort(a.cohort);props={(r['dataset'],r['video_id']):r for r in map(json.loads,a.proposals.open())};asr=transcript_rows();model=MLLM(a.model);policy='Hateful event means an asserted or endorsed attack, dehumanization, threat, exclusion, or hateful stereotype toward a person or protected/social group; reporting, quotation, counterspeech, satire, and condemnation are not endorsement.'
 for row in cohort:
  key=(row['dataset'],row['video_id']);pr=props[key];d=float(row['duration']);p=pr['proposals'][0];delta=(p['end']-p['start'])*(FACTOR-1);left=(max(0.,p['start']-.75*delta),max(0.,p['start']-.25*delta));right=(min(d,p['end']+.25*delta),min(d,p['end']+.75*delta));anchor=(p['start'],p['end']);chunks,rejected=valid_chunks(asr.get(key,[]),d);times=(np.arange(64)+.5)/64*d;frames,times,fb=frames_at_times(row['video_path'],times);images=[];prompts=[];meta=[]
  for modality in ('joint','visual','text'):
   for order in (0,1):
    first,left_name=(left,'LEFT') if order==0 else (right,'RIGHT');second,right_name=(right,'RIGHT') if order==0 else (left,'LEFT');evidence={'anchor':text_for(chunks,[anchor]),'first':text_for(chunks,[first]),'second':text_for(chunks,[second])} if modality!='visual' else {'transcript':'WITHHELD'};images.append(card(anchor,left,right,frames,times,order,visual=modality!='text'));prompts.append(f'{policy} The ANCHOR is the detected event core. FIRST and SECOND are equal-duration competing outside rings. Modality view={modality}. Timestamp transcript={json.dumps(evidence,ensure_ascii=False)}. Which outside ring is more likely a continuation of the SAME event rather than unrelated context? Choose exactly one next letter: A=FIRST, B=SECOND, C=UNCERTAIN. Answer one letter only:');meta.append((modality,order))
  z=logits(model,images,prompts);views={}
  for i,m in enumerate(('joint','visual','text')):views[m],scores=verdict(z[2*i],z[2*i+1]);views[m]={'verdict':views[m],'scores':scores,'forward':z[2*i].tolist(),'reverse':z[2*i+1].tolist()}
  active=['joint','visual']+(['text'] if text_for(chunks,[left])+text_for(chunks,[right]) else []);decisions=[views[m]['verdict'] for m in active];robust=.75 if decisions and all(x=='L' for x in decisions) else .25 if decisions and all(x=='R' for x in decisions) else .5;joint=.75 if views['joint']['verdict']=='L' else .25 if views['joint']['verdict']=='R' else .5;mean_scores=np.mean([views[m]['scores'] for m in active],axis=0);mean=.75 if np.argmax(mean_scores)==0 else .25 if np.argmax(mean_scores)==1 else .5;alloc={'mrea_robust':robust,'mrea_joint':joint,'mrea_mean':mean,'mrea_geometry':geometry_a(pr),'mrea_symmetric':.5}
  for method,aa in alloc.items():
   x,y=allocate(pr,aa);iv=Interval(x,y,1.);append_jsonl(a.out,Prediction(method,key[0],key[1],d,score_curve=intervals_to_curve([iv],d),intervals=[iv],calls=model.calls,modality_evidence={'allocation':aa,'view_verdicts':views,'active_modalities':active,'invalid_asr_spans_rejected':rejected,'ffmpeg_fallback_frames':fb},raw={'gt_access':False,'factor':FACTOR,'fixed_total_length':True}))
  print(json.dumps({'dataset':key[0],'video_id':key[1],'views':{m:x['verdict'] for m,x in views.items()},'allocations':alloc}),flush=True)
if __name__=='__main__':main()
