#!/usr/bin/env python3
"""Frozen text-only stance-aware hateful-event existence expert."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_melt import MLLM,sanitized_cohort

@torch.inference_mode()
def score(model,prompts,batch):
 out=[]
 for i in range(0,len(prompts),batch):
  msgs=[[{'role':'user','content':[{'type':'text','text':p}]}] for p in prompts[i:i+batch]];texts=[model.processor.apply_chat_template(m,tokenize=False,add_generation_prompt=True) for m in msgs];inp=model.processor(text=texts,padding=True,return_tensors='pt').to(model.model.device);z=model.model(**inp,use_cache=False,logits_to_keep=1).logits[:,-1].float()[:,model.binary_token_ids];out.extend((z[:,0]-z[:,1]).cpu().tolist());model.calls+=1
 return out


def make_prompt(text, policy, question, char_cap):
 sample=text if len(text)<=char_cap else text[:char_cap//2]+' ... [middle omitted deterministically] ... '+text[-char_cap//2:]
 return f'{policy} Transcript: {json.dumps(sample,ensure_ascii=False)} {question} Answer Yes or No only:'


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct');ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--char-cap',type=int,default=6000);ap.add_argument('--prompt-mode',choices=('stance','generic'),default='stance');a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 rows=sanitized_cohort(a.manifest);model=MLLM(a.model);policy=('Hateful content is an asserted or endorsed attack, dehumanization, threat, exclusion, or hateful stereotype directed at a person or protected/social group. Quotation, neutral reporting, counterspeech, satire, and condemnation are not endorsement.' if a.prompt_mode=='stance' else 'Hateful content includes hateful, demeaning, insulting, threatening, stereotypical, or slur-containing language about a person or group.');question=('Does this transcript contain at least one complete asserted or endorsed hateful event?' if a.prompt_mode=='stance' else 'Does this transcript contain hateful or demeaning content?');prompts=[]
 for r in rows:
  text=str(r.get('transcript',''));prompts.append(make_prompt(text,policy,question,a.char_cap))
 # A single explicit empty-transcript query is the deployment-time null
 # evidence anchor. It requires no target labels or dataset statistics.
 null_log_odds=score(model,[make_prompt('',policy,question,a.char_cap)],1)[0]
 values=score(model,prompts,a.batch_size);config={'model':a.model,'char_cap':a.char_cap,'prompt_mode':a.prompt_mode,'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'threshold':0.0,'null_log_odds':null_log_odds,'null_prompt':'explicit_empty_transcript','batch_size':a.batch_size,'padding_side':model.processor.tokenizer.padding_side,'next_token_position':-1}
 with a.out.open('x',encoding='utf-8') as f:
  for r,z in zip(rows,values):f.write(json.dumps({'dataset':r['dataset'],'video_id':r['video_id'],'log_odds':z,'positive':z>0,'config':config},ensure_ascii=False)+'\n')
 print(json.dumps({'n':len(values),'positive':sum(x>0 for x in values),'calls':model.calls,'null_log_odds':null_log_odds,'min':min(values),'max':max(values)}))
if __name__=='__main__':main()
