#!/usr/bin/env python3
"""Collect intervention-based modality credit on local hateful-event windows.

Each temporal bin is scored under three *actual* inputs: joint video+speech,
visual-only, and speech-only.  This does not ask for stance or self-reported
attribution.  The three forward responses permit interventional unique/shared
credit and later typed routing into T3AL.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import numpy as np,torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
GT=Path('/home/jehc223/Retrieval-hate/data/gt/frame_gt_4fps')
ASR={
 'HateMM':ROOT/'results/hatemm_localization/per_chunk.jsonl',
 'HateClipSeg':ROOT/'results/reproduction/ours/hateclipseg/per_chunk.jsonl',
 'MHC':ROOT/'results/reproduction/ours/mhclip_en/per_chunk.jsonl',
 'MHC_zh':ROOT/'results/reproduction/ours/mhclip_zh/per_chunk.jsonl'}
STAMP={
 'HateClipSeg':ROOT/'results/interleaved_timeline/hateclipseg/timestamped_chunks.jsonl',
 'MHC':ROOT/'results/interleaved_timeline/mhclip_en/timestamped_chunks.jsonl',
 'MHC_zh':ROOT/'results/interleaved_timeline/mhclip_zh/timestamped_chunks.jsonl'}

def load_jsonl(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def parse_score(x):
 m=re.search(r'\{[^{}]*"score"\s*:\s*(\d+(?:\.\d+)?)[^{}]*\}',x,re.S)
 if not m: raise ValueError('no score JSON: '+x[:200])
 return np.clip(float(m.group(1))/100,0,1)
def infer(frames,prompt,processor,model):
 content=[{'type':'image','image':x} for x in frames]+[{'type':'text','text':prompt}]
 msg=[{'role':'user','content':content}]; text=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
 inp=processor(text=[text],images=frames if frames else None,return_tensors='pt').to(model.device)
 with torch.inference_mode(): out=model.generate(**inp,max_new_tokens=48,do_sample=False)
 return processor.batch_decode(out[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
def cohort(per_ds, selection):
 manifest={(r['dataset'],r['video_id']):r for r in load_jsonl(ROOT/'results/label_free_adapt/manifests/all_test.jsonl')}
 out=[]
 for ds,p in ASR.items():
  av={r['video_id'] for r in load_jsonl(p)};z=np.load(GT/(ds+'.npz'),allow_pickle=True)
  cand=[]
  for v,sp,y in zip(z['video_ids'],z['split'],z['y4']):
   k=(ds,str(v))
   if sp=='test' and str(v) in av and k in manifest:
    yy=np.asarray(y); cand.append((0 if len(np.unique(yy))==2 else 1,str(v)))
  if selection=='mechanism': order=[v for _,v in sorted(cand)]
  else: order=sorted(v for _,v in cand)  # label-blind, deterministic ID order
  for v in order[:per_ds]:out.append(manifest[(ds,v)])
 return out
def chunks_by_video():
 out={}
 stamped={}
 for ds,p in STAMP.items():
  for r in load_jsonl(p): stamped[(ds,r['video_id'])]=r.get('chunks',[])
 for ds,p in ASR.items():
  for r in load_jsonl(p):
   if 'text' not in r:
    cs=stamped.get((ds,r['video_id']),[]);i=int(r['chunk_index'])
    r['text']=cs[i].get('text','') if i<len(cs) else ''
   out.setdefault((ds,r['video_id']),[]).append(r)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--per-dataset',type=int,default=4)
 ap.add_argument('--selection',choices=['mechanism','label_blind'],default='mechanism')
 ap.add_argument('--bins',type=int,default=6);ap.add_argument('--frames-per-bin',type=int,default=3)
 ap.add_argument('--model',default='Qwen/Qwen3-VL-4B-Instruct');a=ap.parse_args()
 from transformers import AutoModelForImageTextToText,AutoProcessor
 from scripts.label_free_adapt.run_numpro import sample_numbered_frames
 model=AutoModelForImageTextToText.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='auto',local_files_only=True).eval()
 proc=AutoProcessor.from_pretrained(a.model,max_pixels=448*448,local_files_only=True); cmap=chunks_by_video();out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 done={(r['dataset'],r['video_id']) for r in load_jsonl(out)} if out.exists() else set()
 base=('Judge only whether this local video window contains hateful, demeaning, dehumanizing, threatening, or exclusionary content directed at a person or group. '
       'Use only supplied evidence. Do not infer the rest of the video. Return JSON only: {"score": integer 0..100}. ')
 for row in cohort(a.per_dataset,a.selection):
  key=(row['dataset'],row['video_id']);
  if key in done:continue
  frames,times=sample_numbered_frames(row['video_path'],a.bins*a.frames_per_bin);dur=float(row['duration']);bins=[]
  for b in range(a.bins):
   s=b*dur/a.bins;e=(b+1)*dur/a.bins;ims=[x for x,t in zip(frames,times) if s<=t<e] or [frames[min(b*len(frames)//a.bins,len(frames)-1)]]
   speech=' '.join(r['text'] for r in cmap[key] if float(r['span'][1])>s and float(r['span'][0])<e)[:2000]
   prompts={
    'joint':base+'Evidence includes the images and timestamp-aligned speech: '+json.dumps(speech,ensure_ascii=False),
    'visual':base+'Evidence includes images only; no speech transcript is available.',
    'speech':base+'Evidence includes timestamp-aligned speech only; no visual evidence is available: '+json.dumps(speech,ensure_ascii=False)}
   ans={};scores={}
   for view in ('joint','visual','speech'):
    vf=[] if view=='speech' else ims;ans[view]=infer(vf,prompts[view],proc,model);scores[view]=parse_score(ans[view])
   j,v,t=scores['joint'],scores['visual'],scores['speech']
   # Two-player Möbius decomposition. Unique credits cannot be negative; a
   # negative interaction is retained as conflict rather than reassigned.
   credit={'visual_unique':max(0,j-t),'speech_unique':max(0,j-v),'shared':max(0,j-v-t),'conflict':max(0,v+t-j)}
   bins.append({'start':s,'end':e,'speech':speech,'scores':scores,'credit':credit,'responses':ans})
   print(json.dumps({'dataset':key[0],'video_id':key[1],'bin':b,'scores':scores}),flush=True)
  with out.open('a') as f:f.write(json.dumps({'dataset':key[0],'video_id':key[1],'duration':dur,'bins':bins},ensure_ascii=False)+'\n')

if __name__=='__main__':main()
