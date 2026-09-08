#!/usr/bin/env python3
"""Preprocessing: BERT CLS timeline for an arbitrary sealed manifest."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np,torch
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--chunks',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True);ap.add_argument('--model',default='bert-base-uncased');a=ap.parse_args()
 manifest={r['video_id']:r for r in map(json.loads,a.manifest.open())};chunks={r['video_id']:r.get('chunks',[]) for r in map(json.loads,a.chunks.open()) if r['video_id'] in manifest};a.out_dir.mkdir(parents=True,exist_ok=True)
 from transformers import AutoModel,AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);model=AutoModel.from_pretrained(a.model,local_files_only=True,add_pooling_layer=False).eval().cuda();done=0
 for vid,row in manifest.items():
  dst=a.out_dir/(vid+'.npy')
  if dst.exists():done+=1;continue
  cs=[x for x in chunks.get(vid,[]) if str(x.get('text','')).strip() and x.get('start') is not None and x.get('end') is not None];n=max(1,int(math.ceil(float(row['duration']))));out=np.zeros((n,768),np.float32)
  if cs:
   values=[]
   for s in range(0,len(cs),64):
    inp=tok([x['text'] for x in cs[s:s+64]],padding=True,truncation=True,max_length=64,return_tensors='pt').to('cuda')
    with torch.inference_mode():values.append(model(**inp).last_hidden_state[:,0].float().cpu().numpy())
   z=np.concatenate(values)
   for x,v in zip(cs,z):
    lo=max(0,int(math.floor(float(x['start']))));hi=min(n,max(lo+1,int(math.ceil(float(x['end'])))));out[lo:hi]=v
  np.save(dst,out);done+=1
 print(json.dumps({'n_manifest':len(manifest),'n_with_chunks':sum(bool(chunks.get(v)) for v in manifest),'n_written_or_existing':done,'out':str(a.out_dir)},indent=2))
if __name__=='__main__':main()
