#!/usr/bin/env python3
"""Generate video-specific, visually groundable positive/confounder query pairs."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
def infer(frames,prompt,p,m):
 c=[{'type':'image','image':x} for x in frames]+[{'type':'text','text':prompt}];msg=[{'role':'user','content':c}]
 text=p.apply_chat_template(msg,tokenize=False,add_generation_prompt=True);inp=p(text=[text],images=frames,return_tensors='pt').to(m.device)
 with torch.inference_mode():o=m.generate(**inp,max_new_tokens=256,do_sample=False)
 return p.batch_decode(o[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
def parse(s):
 m=re.search(r'\{.*\}',s,re.S)
 if not m:raise ValueError('no JSON')
 x=json.loads(m.group());
 if 'positive' not in x:x['positive']=x.get('positive_queries') or x.get('queries') or []
 if 'confounder' not in x:x['confounder']=x.get('confounders') or x.get('negative') or []
 for k in ('positive','confounder'):
  if not isinstance(x.get(k),list) or not x[k]:raise ValueError('bad '+k)
  x[k]=[str(q)[:160] for q in x[k][:4]]
 return x
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cohort',required=True);ap.add_argument('--out',required=True);ap.add_argument('--nframes',type=int,default=12);ap.add_argument('--model',default='Qwen/Qwen3-VL-4B-Instruct');ap.add_argument('--force-hypotheses',action='store_true');a=ap.parse_args()
 from transformers import AutoModelForImageTextToText,AutoProcessor
 from scripts.label_free_adapt.run_numpro import sample_numbered_frames
 model=AutoModelForImageTextToText.from_pretrained(a.model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='auto',local_files_only=True).eval();proc=AutoProcessor.from_pretrained(a.model,max_pixels=448*448,local_files_only=True)
 rows=[json.loads(x) for x in Path(a.cohort).read_text().splitlines()];out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);done={(r['dataset'],r['video_id']) for r in map(json.loads,out.open())} if out.exists() else set()
 for r in rows:
  key=(r['dataset'],r['video_id']);
  if key in done:continue
  # The causal-credit cohort stores local speech; concatenate once for semantic
  # query construction, never exposing temporal labels.
  speech=' '.join(b.get('speech','') for b in r.get('bins',[]))[:5000]
  manifest=next(x for x in map(json.loads,(ROOT/'results/label_free_adapt/manifests/all_test.jsonl').open()) if (x['dataset'],x['video_id'])==key)
  frames,_=sample_numbered_frames(manifest['video_path'],a.nframes)
  prompt=('Create text queries for a frozen vision-language model to find local hateful or demeaning events in THIS video. '
   'Do not classify the whole video and do not discuss quotation, endorsement, or speaker stance. '
   'positive: 2-4 short visually groundable phrases describing concrete people, actions, gestures, symbols, or visible text that may coincide with targeted hostility. '
   'confounder: 2-4 equally concrete matched phrases likely elsewhere in this same video but not themselves targeted hostility. '
   'Avoid abstract words such as hateful, toxic, offensive, bad, or harmful. Return JSON only: {"positive":[...],"confounder":[...]}. '
   'Timestamped speech transcript, supplied only to identify concrete entities/actions: '+json.dumps(speech,ensure_ascii=False))
  if a.force_hypotheses:
   prompt += (' This is retrieval-hypothesis generation, not video classification. You MUST return 2-4 nonempty positive hypotheses even if evidence is uncertain or the video may be benign. '
              'Use the most plausible concrete visible person/action/gesture/symbol/text hypotheses; do not return empty lists.')
  raw=infer(frames,prompt,proc,model)
  try:q=parse(raw)
  except Exception as e:
   print(json.dumps({'dataset':key[0],'video_id':key[1],'error':str(e),'raw':raw},ensure_ascii=False),flush=True);continue
  rec={'dataset':key[0],'video_id':key[1],**q,'raw':raw}
  with out.open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n');print(json.dumps(rec,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
