#!/usr/bin/env python3
"""P2: training-free visual temporal canvas localization."""
from __future__ import annotations
import argparse,hashlib,io,json,math,re,subprocess,sys
from collections import Counter
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.idea_discovery.run_counterfactual_evidence import rows,transcript_rows,poset_curve,ecdf
from scripts.label_free_adapt.mechanisms import parse_json_object
from scripts.label_free_adapt.schema import Prediction,append_jsonl,curve_to_intervals

COHORT=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_8_clean.jsonl'
T3AL=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_t3al_clean'
TOPO=ROOT/'results/idea_discovery/paradigm_adapt/stage_a_topology_clean'

def frame_at(cap,t,path):
 ok=False;x=None
 if cap is not None:cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000);ok,x=cap.read()
 if ok:return Image.fromarray(cv2.cvtColor(x,cv2.COLOR_BGR2RGB)).resize((224,126)),False
 cmd=['ffmpeg','-v','error','-ss',f'{float(t):.6f}','-i',str(path),'-frames:v','1','-f','image2pipe','-vcodec','png','pipe:1']
 z=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False,timeout=30)
 if z.returncode or not z.stdout:raise RuntimeError(f'frame decode failed at {t:.3f}s for {path}: {z.stderr.decode(errors="replace")[-500:]}')
 return Image.open(io.BytesIO(z.stdout)).convert('RGB').resize((224,126)),True

def frames_at_times(path,times):
 times=np.asarray(times,float);nbins=len(times);cap=cv2.VideoCapture(str(path))
 code=int(cap.get(cv2.CAP_PROP_FOURCC));fourcc=''.join(chr((code>>8*i)&255) for i in range(4)).lower()
 decoder=None if fourcc in {'av01','av1 '} else cap
 # Container duration can follow a slightly longer audio stream.  Clamp visual
 # sampling to the final video-frame centre instead of asking both decoders for
 # a timestamp beyond the video stream (e.g. HateMM non_hate_video_141).
 fps=float(cap.get(cv2.CAP_PROP_FPS));frame_count=float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 visual_last=((frame_count-1)/fps if fps>0 and frame_count>=1 else float('inf'))
 decode_times=np.minimum(times,max(0.,visual_last))
 try:pairs=[frame_at(decoder,t,path) for t in decode_times]
 finally:cap.release()
 def suspicious(xs):
  stats=[(float(np.asarray(im).mean()),float(np.asarray(im).std())) for im,_ in xs]
  return sum(mean<1.0 or std<0.5 for mean,std in stats)>=nbins-1
 # A suspicious OpenCV batch can mean either a decoder failure or a legitimate
 # black/static segment (several HateMM videos are audio-led with a black
 # screen).  Re-decode every requested timestamp with ffmpeg, whose pipe path
 # raises on an actual decode failure.  A successfully decoded black frame is
 # real content and must not be silently excluded from evaluation.
 if decoder is not None and suspicious(pairs):pairs=[frame_at(None,t,path) for t in decode_times]
 return [x[0] for x in pairs],times,sum(x[1] for x in pairs)

def video_frames(path,duration,nbins=16):
 return frames_at_times(path,(np.arange(nbins)+.5)/nbins*duration)

def bin_means(curve,nbins=16):
 curve=np.asarray(curve,float);ids=np.minimum(nbins-1,((np.arange(len(curve))+.5)/len(curve)*nbins).astype(int))
 return np.asarray([curve[ids==i].mean() if np.any(ids==i) else np.interp((i+.5)/nbins,(np.arange(len(curve))+.5)/len(curve),curve) for i in range(nbins)])

def make_canvas(ims,times,curve=None,curve_name='',nbins=16):
 canvas=Image.new('RGB',(4*224,4*156+150),'white');draw=ImageDraw.Draw(canvas)
 for i,(im,t) in enumerate(zip(ims,times)):
  x=(i%4)*224;y=(i//4)*156;canvas.paste(im,(x,y));draw.text((x+4,y+128),f'{i:02d}  {t:.1f}s',fill='black')
 y0=4*156+20
 if curve is None:draw.text((5,y0-17),'No proposal curve (multimodal evidence only)',fill='black')
 else:
  draw.text((5,y0-17),f'blue={curve_name} proposal',fill='black')
  vals=bin_means(ecdf(curve),nbins)
  pts=[(int((i+.5)/nbins*canvas.width),int(y0+110-100*v)) for i,v in enumerate(vals)]
  draw.line(pts,fill='blue',width=4)
 for i in range(nbins+1):
  x=int(i/nbins*canvas.width);draw.line((x,y0,x,y0+115),fill=(210,210,210),width=1)
 return canvas

def timed_text(chunks,duration,nbins=16,epsilon=.05):
 bins=[[] for _ in range(nbins)]
 for r in chunks:
  a,b=map(float,r['span'])
  if b>duration and b-duration<=epsilon:b=duration
  if not (0<=a<b<=duration):continue
  text=str(r.get('text',''))
  words=[ch for ch in text if not ch.isspace()] if re.search(r'[\u3400-\u9fff]',text) else text.split()
  for k,word in enumerate(words):
   wt=a+(k+.5)/len(words)*(b-a);i=min(nbins-1,max(0,int(wt/duration*nbins)));bins[i].append(word)
 return [' '.join(x) for x in bins]

def valid_chunks(chunks,duration,epsilon=.05):
 clean=[];rejected=[]
 for r in chunks:
  span=r.get('span',[])
  if len(span)!=2:rejected.append(span);continue
  a,b=map(float,span)
  if b>duration and b-duration<=epsilon:b=duration
  if not (0<=a<b<=duration):rejected.append(span);continue
  x=dict(r);x['span']=[a,b];clean.append(x)
 return clean,rejected

def infer(canvas,prompt,processor,model):
 msg=[{'role':'user','content':[{'type':'image','image':canvas},{'type':'text','text':prompt}]}]
 text=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
 inp=processor(text=[text],images=[canvas],return_tensors='pt').to(model.device)
 with torch.inference_mode():out=model.generate(**inp,max_new_tokens=256,do_sample=False)
 return processor.batch_decode(out[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)[0]

def dense_bins(scores,n):
 ids=np.minimum(len(scores)-1,((np.arange(n)+.5)/n*len(scores)).astype(int))
 return np.asarray(scores,float)[ids]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cohort',type=Path,default=COHORT)
 ap.add_argument('--t3al-curves',type=Path,default=T3AL);ap.add_argument('--topology-curves',type=Path,default=TOPO)
 ap.add_argument('--model',default='Qwen/Qwen3-VL-8B-Instruct');ap.add_argument('--limit',type=int,default=0)
 ap.add_argument('--methods',default='canvas_only,t3al_canvas,poset_canvas',
  help='comma-separated subset of canvas_only,t3al_canvas,poset_canvas')
 a=ap.parse_args()
 from transformers import AutoModelForImageTextToText,AutoProcessor
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True,max_pixels=896*800)
 model=AutoModelForImageTextToText.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cuda:0',local_files_only=True,attn_implementation='sdpa').eval()
 chunks=transcript_rows();allowed=('canvas_only','t3al_canvas','poset_canvas')
 methods=tuple(x.strip() for x in a.methods.split(',') if x.strip())
 if not methods or any(x not in allowed for x in methods):
  raise ValueError(f'--methods must be a nonempty subset of {allowed}, got {methods}')
 config={'version':'p2_canvas_v4_selectable_methods','model':a.model,'methods':methods,
         'cohort':str(a.cohort.resolve()),'t3al':str(a.t3al_curves.resolve()),'poset':str(a.topology_curves.resolve()),
         'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 config_id=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
 prior=rows(a.out) if a.out.exists() else []
 bad=[r for r in prior if r.get('raw',{}).get('config_id')!=config_id]
 if bad:raise RuntimeError(f'output contains {len(bad)} rows from a different/missing config; use a new --out path')
 done={(r['dataset'],r['video_id'],r['method']) for r in prior}
 cohort=rows(a.cohort)
 if a.limit>0:cohort=cohort[:a.limit]
 expected={(r['dataset'],r['video_id'],m) for r in cohort for m in methods}
 for row in cohort:
  d,v,dur=row['dataset'],row['video_id'],float(row['duration'])
  if all((d,v,m) in done for m in methods):continue
  t=(np.load(a.t3al_curves/d/f'{v}.npy') if 't3al_canvas' in methods else None)
  q=(np.load(a.topology_curves/d/f'{v}.npy') if 'poset_canvas' in methods else None)
  video_chunks,invalid_spans=valid_chunks(chunks.get((d,v),[]),dur)
  p=(poset_curve(q,video_chunks,dur) if q is not None else None);speech=timed_text(video_chunks,dur)
  ims,times,fallback_frames=video_frames(row['video_path'],dur)
  speech_records=[{'bin':i,'start':round(i*dur/16,2),'end':round((i+1)*dur/16,2),'text':text}
                  for i,text in enumerate(speech) if text]
  prompt=('The image is a 16-bin video timeline, optionally followed by one blue noisy label-free proposal curve. '
   'Timestamp-aligned speech records are '+json.dumps(speech_records,ensure_ascii=False)+'. Assign each bin a calibrated 0..100 HATE-EVIDENCE score. '
   'Score direct use of a protected-group slur or an explicit attack, dehumanization, threat, exclusion, or hateful stereotype high (70-100), even when the target is implicit. '
   'Score plausible but context-dependent hostility 30-69, benign content 0-10. Lower a high lexical score only when the local visual/speech context clearly shows quotation, reporting, counterspeech, satire, or condemnation rather than endorsement. '
   'Use both modalities and temporal neighbors; proposal curves are uncertain hints, not truth. The value at scores[i] MUST describe bin i, so text tagged bin 12 can only directly raise scores[12], not scores[10]. Produce a graded timeline rather than an all-or-none video verdict. '
   'Return JSON only with every key present exactly once: {"scores":{"0":integer,"1":integer,"2":integer,"3":integer,"4":integer,"5":integer,"6":integer,"7":integer,"8":integer,"9":integer,"10":integer,"11":integer,"12":integer,"13":integer,"14":integer,"15":integer}}.')
  candidates={'canvas_only':(None,''),'t3al_canvas':(t,'T3AL'),'poset_canvas':(p,'POSET')}
  for method in methods:
   base_curve,curve_name=candidates[method]
   if (d,v,method) in done:continue
   canvas=make_canvas(ims,times,base_curve,curve_name)
   attempts=[];scores=[]
   for call_index in range(2):
    retry='' if call_index==0 else (' Your previous output was invalid: '+attempts[-1]+'. '
     'Return a scores object with exactly the 16 named keys "0" through "15"; do not omit or merge keys.')
    raw=infer(canvas,prompt+retry,proc,model);attempts.append(raw)
    try:obj=parse_json_object(raw)
    except (ValueError,TypeError):obj={}
    value=obj.get('scores',{})
    scores=[value.get(str(i)) for i in range(16)] if isinstance(value,dict) else []
    if len(scores)==16 and all(isinstance(x,(int,float)) and math.isfinite(float(x)) for x in scores):break
   if len(scores)!=16 or not all(isinstance(x,(int,float)) and math.isfinite(float(x)) for x in scores):
    raise RuntimeError(f'{d}/{v}/{method}: invalid scores after {len(attempts)} calls; raw={attempts!r}')
   dense_length=max(1,int(np.floor(dur*4)))
   dense=np.clip(dense_bins(np.asarray(scores,float)/100,dense_length),0,1)
   curve=dense if base_curve is None else .75*ecdf(base_curve)+.25*dense
   pred=Prediction(method,d,v,dur,score_curve=curve.tolist(),intervals=curve_to_intervals(curve,dur,.75),calls=len(attempts),
    modality_evidence={'bin_scores':scores,'invalid_asr_spans_rejected':invalid_spans,'ffmpeg_fallback_frames':fallback_frames},raw={'responses':attempts,'config_id':config_id,'config':config})
   append_jsonl(a.out,pred)
   print(json.dumps({'dataset':d,'video_id':v,'method':method,'scores':scores}),flush=True)
 counts=Counter((r['dataset'],r['video_id'],r['method']) for r in rows(a.out));observed=set(counts)
 missing=expected-observed;duplicates={k:n for k,n in counts.items() if k in expected and n!=1};unexpected=observed-expected
 if missing:raise RuntimeError(f'incomplete output: {len(missing)} rows missing; examples={sorted(missing)[:3]}')
 if duplicates:raise RuntimeError(f'duplicate output rows: {list(duplicates.items())[:3]}')
 if unexpected:raise RuntimeError(f'unexpected output rows: {sorted(unexpected)[:3]}')
 return 0
if __name__=='__main__':raise SystemExit(main())
