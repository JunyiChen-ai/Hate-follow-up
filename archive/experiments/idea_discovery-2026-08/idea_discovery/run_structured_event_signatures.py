#!/usr/bin/env python3
"""Generate factual and shifted-core structured event signatures with Qwen3-VL."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_counterfactual_evidence import transcript_rows
from scripts.idea_discovery.run_visual_temporal_canvas import frames_at_times,valid_chunks
from scripts.label_free_adapt.mechanisms import parse_json_object

def shifted_interval(core,duration):
 s,e=map(float,core);w=e-s; candidates=((s+.5*duration)%duration,(s-.5*duration)%duration)
 for start in candidates:
  start=min(max(0.,start),max(0.,duration-w))
  if min(e,start+w)-max(s,start)<=.1*w:return start,start+w
 return (0.,min(duration,w)) if s>.5*duration else (max(0.,duration-w),duration)
def sample_times(interval):
 s,e=interval;return np.linspace(s,e,6)[1:-1] if e>s else np.full(4,s)
def transcript(chunks,interval):
 s,e=interval;parts=[str(x.get('text','')) for x in chunks if float(x['span'][1])>=s and float(x['span'][0])<=e]
 return ' '.join(parts)[:2500]
def canvas(a,b):
 out=Image.new('RGB',(896,260),'white');draw=ImageDraw.Draw(out);draw.text((4,2),'FACTUAL CORE',fill='black');draw.text((4,130),'SHIFTED PLACEBO CORE',fill='black')
 for row,imgs in enumerate((a,b)):
  for i,img in enumerate(imgs):out.paste(img.resize((224,108)),(224*i,18+130*row))
 return out
def infer(proc,model,image,prompt):
 msg=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}];txt=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True);inp=proc(text=[txt],images=[image],return_tensors='pt').to(model.device)
 with torch.inference_mode():o=model.generate(**inp,max_new_tokens=500,do_sample=False)
 return proc.batch_decode(o[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct');a=ap.parse_args()
 prior=list(map(json.loads,a.out.open())) if a.out.exists() else [];done={(x['dataset'],x['video_id']) for x in prior}
 from transformers import AutoModelForImageTextToText,AutoProcessor
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True,max_pixels=896*300);model=AutoModelForImageTextToText.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cuda:0',local_files_only=True,attn_implementation='sdpa').eval();chunks=transcript_rows()
 cfg={'model':a.model,'version':'structured-signature-v1','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'gt_access':False}
 with a.out.open('a') as h:
  for row in map(json.loads,a.manifest.open()):
   key=(row['dataset'],row['video_id']);
   if key in done:continue
   duration=float(row['duration']);core=tuple(map(float,row['selection']['tight_interval']));shift=shifted_interval(core,duration);clean,rejected=valid_chunks(chunks.get(key,[]),duration)
   fi,_,_=frames_at_times(row['video_path'],sample_times(core));si,_,_=frames_at_times(row['video_path'],sample_times(shift));pic=canvas(fi,si);ft,st=transcript(clean,core),transcript(clean,shift)
   prompt=('Top row and FACTUAL transcript come from a label-free candidate event core. Bottom row and SHIFTED transcript are a same-length temporal placebo. '
           'For each independently summarize only observable event identity, not whether the dataset labels it hateful. Produce a compact cross-modal signature with actor/source, target, action_or_proposition, stance, acoustic_cue, and one concise grounding_query. Write every value in English, translating non-English speech. Use unknown when unsupported. '
           f'FACTUAL transcript: {json.dumps(ft,ensure_ascii=False)} SHIFTED transcript: {json.dumps(st,ensure_ascii=False)} '
           'Return JSON only: {"factual":{"actor":"...","target":"...","action_or_proposition":"...","stance":"...","acoustic_cue":"...","grounding_query":"..."},"shifted":{same six fields}}.')
   raw=infer(proc,model,pic,prompt);obj=parse_json_object(raw)
   if not all(isinstance(obj.get(x),dict) and obj[x].get('grounding_query') for x in ('factual','shifted')):raise RuntimeError(f'invalid {key}: {raw}')
   rec={'dataset':key[0],'video_id':key[1],'duration':duration,'core':core,'shifted_core':shift,'factual':obj['factual'],'shifted':obj['shifted'],'factual_transcript':ft,'shifted_transcript':st,'raw_response':raw,'config':cfg,'invalid_asr_spans_rejected':rejected};h.write(json.dumps(rec,separators=(',',':'),ensure_ascii=False)+'\n');h.flush();print(json.dumps({'dataset':key[0],'video_id':key[1],'query':obj['factual']['grounding_query']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
