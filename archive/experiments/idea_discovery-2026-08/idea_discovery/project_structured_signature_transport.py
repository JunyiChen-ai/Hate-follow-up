#!/usr/bin/env python3
"""Ground structured event signatures and transport them into the dense field."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
DATA_DIR={"HateMM":"hatemm","HateClipSeg":"hateclipseg","MHC":"mhclip_en","MHC_zh":"mhclip_zh"}
BERT_ID={"HateMM":"bert-base-uncased","HateClipSeg":"bert-base-uncased","MHC":"bert-base-uncased","MHC_zh":"bert-base-chinese"}
IB_DIR=ROOT/'third_party/lavad/libs/ImageBind';IB_CKPT=ROOT/'results/label_free_adapt/assets/imagebind/imagebind_huge.pth'

def load(path,method=None):
 out={}
 for l in Path(path).open():
  r=json.loads(l)
  if method is None or r.get('method')==method:out[(r['dataset'],r['video_id'])]=r
 return out
def rank01(x):
 x=np.asarray(x,float);o=np.argsort(x,kind='stable');r=np.empty(len(x));r[o]=np.arange(len(x));return (r+.5)/len(x)
def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))
def transport(base,evidence,g=.01):
 base=np.asarray(base,float);z=np.log(np.clip(base,1e-6,1-1e-6)/np.clip(1-base,1e-6,1));r=np.asarray(evidence,float);r-=r.mean();rz=np.median(abs(z-np.median(z)))+1e-6;rr=np.median(abs(r-np.median(r)))+1e-6;changed=z+g*r*rz/rr;target=base.mean();lo,hi=-20.,20.
 for _ in range(60):
  m=(lo+hi)/2
  if sigmoid(changed+m).mean()<target:lo=m
  else:hi=m
 return sigmoid(changed+(lo+hi)/2)
def normalize(x):x=np.asarray(x,float);return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)
def signature_text(s):
 return '; '.join(f'{k.replace("_"," ")}: {s.get(k)}' for k in ('actor','target','action_or_proposition','stance','acoustic_cue') if str(s.get(k,'')).lower() not in ('','unknown','none')) or 'unknown event'
def resize_curve(x,n):return np.interp(np.linspace(0,len(x)-1,n),np.arange(len(x)),x)

def imagebind_text(texts):
 sys.path.insert(0,str(IB_DIR));old=Path.cwd();os.chdir(ROOT/'third_party/lavad')
 try:
  from imagebind import data as ibdata
  from imagebind.models import imagebind_model
  from imagebind.models.imagebind_model import ModalityType
  model=imagebind_model.imagebind_huge(pretrained=False);model.load_state_dict(torch.load(IB_CKPT,map_location='cpu'));model=model.eval().cuda()
  outputs=[]
  for i in range(0,len(texts),16):
   tok=ibdata.load_and_transform_text(texts[i:i+16],'cuda')
   with torch.inference_mode():outputs.append(model({ModalityType.TEXT:tok})[ModalityType.TEXT].float().cpu().numpy())
  return normalize(np.concatenate(outputs))
 finally:os.chdir(old)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--signatures',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--base-method',required=True);ap.add_argument('--feature-root',type=Path,required=True);ap.add_argument('--audio-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise RuntimeError(f'refusing existing output: {a.out}')
 sig=load(a.signatures);base=load(a.base,a.base_method);keys=sorted(set(sig)&set(base));texts=[]
 for k in keys:
  for arm in ('factual','shifted'):texts.append(signature_text(sig[k][arm]))
 # Frozen text encoders for visual and transcript streams.
 from transformers import CLIPModel,CLIPTokenizerFast,AutoModel,AutoTokenizer
 clip=CLIPModel.from_pretrained('openai/clip-vit-base-patch16',local_files_only=True).eval().cuda();ctok=CLIPTokenizerFast.from_pretrained('openai/clip-vit-base-patch16',local_files_only=True)
 with torch.inference_mode():
  z=ctok(texts,padding=True,truncation=True,return_tensors='pt').to('cuda');clip_text=normalize(clip.get_text_features(**z).float().cpu().numpy())
 del clip;torch.cuda.empty_cache();ib_text=imagebind_text(texts);torch.cuda.empty_cache()
 bert_text=np.empty((len(texts),768),np.float32)
 for ds,model_id in sorted(set(BERT_ID.items()),key=lambda x:x[1]):
  ids=[2*i+j for i,k in enumerate(keys) if k[0]==ds for j in (0,1)]
  if not ids:continue
  tok=AutoTokenizer.from_pretrained(model_id,local_files_only=True);model=AutoModel.from_pretrained(model_id,local_files_only=True,add_pooling_layer=False).eval().cuda()
  for start in range(0,len(ids),32):
   take=ids[start:start+32];inp=tok([texts[i] for i in take],padding=True,truncation=True,max_length=96,return_tensors='pt').to('cuda')
   with torch.inference_mode():v=model(**inp).last_hidden_state[:,0].float().cpu().numpy()
   bert_text[take]=normalize(v)
  del model;torch.cuda.empty_cache()
 methods=('signature_factual_V_v1','signature_factual_A_v1','signature_factual_T_v1','signature_factual_VAT_v1','signature_shifted_VAT_v1',
          'signature_factual_VAT_g0p1_v2','signature_shifted_VAT_g0p1_v2','signature_factual_T_g0p1_v2','signature_shifted_T_g0p1_v2')
 with a.out.open('w') as h:
  for i,k in enumerate(keys):
   row=base[k];n=len(row['score_curve']);corpus=DATA_DIR[k[0]];fields={}
   visual=normalize(np.load(a.feature_root/'clip_b16_1fps'/corpus/(k[1]+'.npy')));audio=normalize(np.load(a.audio_root/k[0]/(k[1]+'.npy')));lang=normalize(np.load(a.feature_root/'bert_sentence_1fps'/corpus/(k[1]+'.npy')))
   for j,arm in enumerate(('factual','shifted')):
    ix=2*i+j;fields[(arm,'V')]=resize_curve(visual@clip_text[ix],n);fields[(arm,'A')]=resize_curve(audio@ib_text[ix],n);fields[(arm,'T')]=resize_curve(lang@bert_text[ix],n)
   evidence={methods[0]:rank01(fields['factual','V']),methods[1]:rank01(fields['factual','A']),methods[2]:rank01(fields['factual','T'])}
   evidence[methods[3]]=np.mean(np.log(np.clip(np.stack([rank01(fields['factual',m]) for m in 'VAT']),1e-6,1)),axis=0)
   evidence[methods[4]]=np.mean(np.log(np.clip(np.stack([rank01(fields['shifted',m]) for m in 'VAT']),1e-6,1)),axis=0)
   evidence[methods[5]]=evidence[methods[3]];evidence[methods[6]]=evidence[methods[4]]
   evidence[methods[7]]=rank01(fields['factual','T']);evidence[methods[8]]=rank01(fields['shifted','T'])
   for method in methods:
    gain=.1 if 'g0p1' in method else .01
    out=dict(row);out['method']=method;out['score_curve']=transport(row['score_curve'],evidence[method],gain).tolist();out['raw']={**out.get('raw',{}),'gt_access':False,'signature_arm':'shifted' if 'shifted' in method else 'factual','signature':sig[k]['shifted' if 'shifted' in method else 'factual'],'transport_gain':gain,'video_mean_preserved':True};h.write(json.dumps(out,separators=(',',':'),ensure_ascii=False)+'\n')
 print(json.dumps({'n_videos':len(keys),'methods':methods},indent=2))
if __name__=='__main__':main()
