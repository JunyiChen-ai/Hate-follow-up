"""Actual narrow36-layer reader, source-status-only change and R1 identity."""
import copy,json,inspect,re,socket,os,time,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np,torch
from PIL import Image
from transformers import Qwen3VLConfig,Qwen3VLModel
sys.path.insert(0,str(next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').exists())))
from src.mllm_judge import Judge,VIDEO_QUESTION
from src.structured_source_generation import Stream
from inputs import ROOT,SPEC
import partition,reader,validate
OUT=ROOT/'runs/20261006_m1_texttiling/r2_reader_cpu_checks';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.manual_seed(0)
class Tok:
 all_special_ids=[]
 def __init__(self):self.v={};self.rev={}
 def encode(self,text,add_special_tokens=False):
  out=[]
  for word in re.findall(r'\w+|[^\w\s]',text):
   if word not in self.v:self.v[word]=128+len(self.v);self.rev[self.v[word]]=word
   out.append(self.v[word])
  assert max(out or [0])<4096
  return out
 def __call__(self,text,**kw):return dict(input_ids=self.encode(text))
 def decode(self,tokens,**kw):return ' '.join(self.rev.get(i,'?') for i in tokens)
cfg=Qwen3VLConfig(text_config=dict(vocab_size=4096,hidden_size=64,intermediate_size=128,num_hidden_layers=36,num_attention_heads=8,num_key_value_heads=2,head_dim=8,max_position_embeddings=8192,rope_parameters=dict(rope_type='default',rope_theta=5000000.,mrope_section=[2,1,1],mrope_interleaved=True)),vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa'
m=Qwen3VLModel(cfg).eval();m.requires_grad_(False);head=torch.nn.Linear(64,4096,bias=False)
j=Judge.__new__(Judge);j.model=SimpleNamespace(model=m,config=cfg,get_output_embeddings=lambda:head)
j.device=torch.device('cpu');j.dtype=torch.float32;j.family='qwen3_vl';j.image_token_id=127;j.softcap=None
j.forward_params=set(inspect.signature(m.forward).parameters);j.tok=Tok();j.processor=SimpleNamespace(image_processor=SimpleNamespace(merge_size=2));j.yes_ids=[70];j.no_ids=[71];j.img_kw={}
j.turn=lambda role,text:dict(role=role,content=text)
j.prefix_messages=lambda frames,segments:([dict(role='system',content='P'),dict(role='user',content=str(segments))],[p for _,p in frames])
def render(msgs,gen):return json.dumps(msgs,ensure_ascii=False)+(' ASSISTANT' if gen else '')
j.render=render
def encode(text,images):
 ids=j.tok.encode(text);enc={}
 if images:
  pixels=[]
  for image in images:
   x=torch.tensor(np.asarray(image).copy(),dtype=torch.float32).permute(2,0,1)/255
   px=x.reshape(3,4,2,4,2).permute(1,3,0,2,4).reshape(-1,3,4)
   pixels.append(px[:,:,None,:].expand(-1,-1,2,-1).reshape(-1,24))
  ids=[10]+sum(([125]+[127]*4+[126,73] for _ in images),[])+ids
  enc.update(pixel_values=torch.cat(pixels),image_grid_thw=torch.tensor([[1,4,4]]*len(images)))
 ids=torch.tensor([ids]);enc.update(input_ids=ids,attention_mask=torch.ones_like(ids))
 if 'mm_token_type_ids' in j.forward_params:enc['mm_token_type_ids']=(ids==127).long()
 return enc
j.encode=encode
def prefix(msgs,files):
 images=[Image.open(p).convert('RGB') for p in files];text=render(msgs,False);enc=encode(text,images)
 for x in images:x.close()
 j.img_tokens=[4]*len(files);j._prefix_text=text;return text,enc
j.encode_prefix=prefix
j.branch_ids=lambda msgs,q,history=None,head_text=None: (j.tok.encode(q),q)
j.answer_ids=lambda msgs,q,s:([62],s)
j.forward_calls=j.vision_calls=0
hooks=[m.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),m.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
p=OUT/'native.png';Image.fromarray(np.full((8,8,3),101,np.uint8)).save(p);frames=[(float(i),p) for i in range(20)]

prior={'__name__':'before_R2','__file__':str(ROOT/'experiments/20261006_m1_texttiling/reader.py')}
exec(compile((OUT/'before_R2_reader.py').read_text(),'before_R2_reader.py','exec'),prior)
class Clock:
 def __init__(self):self.i=0
 def __call__(self,*args):self.i+=1;return float(self.i)
row=dict(dataset='HateMM',video_id='fixture',duration=24.)
segments=[(0.,7.,'first literal speech'),(8.,15.,'second literal speech')]
words=[dict(id=i,text=t,start=a,end=a+.2) for i,(t,a) in enumerate([(' local',1.),(' act',2.),(' context',9.),(' explanation',10.)])]
seg=partition.partition(words);source=dict(words=words,windows=[partition.scope(words,seg,a,a+8) for a in (0,8,16)],source_seconds=2.,counts={'encoder':3,'decoder':5},host=socket.gethostname())
original_choose=Stream.choose
mode='compiled'
def choose(self,options):
 if self.replay:return original_choose(self,options)
 selected=next((x for x in options if x.startswith('W')),options[0]) if mode=='compiled' else options[0]
 ids=j.tok.encode(selected);start=len(self.tokens)
 for token in ids:self.append(token)
 self.events.append(dict(kind='choice',options=options,selected=selected,start=start,end=len(self.tokens)))
 return selected
cases=[]
with torch.no_grad(),patch('reader.frame_paths',return_value=frames),patch('src.native_input_binding.frame_paths',return_value=frames),patch.object(Stream,'choose',choose):
 prior['frame_paths']=lambda *args:frames
 for mode in ('compiled','unknown'):
  reader.REVISION=validate.REVISION=1;reader.clock=Clock();before=reader.read_video(j,row,segments,source,OUT/'production',True);validate.validate_bundle(j,row,segments,source,before,True)
  prior['clock']=Clock();old=prior['read_video'](j,row,segments,source,OUT/'production',True)
  def stable(value):
   if isinstance(value,dict):return {k:stable(v) for k,v in value.items() if k not in ('seconds','times','timings','standalone_seconds','repeat_seconds')}
   if isinstance(value,list):return [stable(v) for v in value]
   return value
  assert stable(before)==stable(old),'R1 numerical/source behavior changed'
  reader.REVISION=validate.REVISION=2;reader.clock=Clock();after=reader.read_video(j,row,segments,source,OUT/'production',True);validate.validate_bundle(j,row,segments,source,after,True)
  assert stable(after['base'])==stable(before['base']) and after['native_ctx']==before['native_ctx'] and after['native_rope']==before['native_rope']
  assert stable(after['packets'])==stable(before['packets']) and after['checks']['parser_forwards']==before['checks']['parser_forwards']
  for a,b,pkt in zip(after['traces'],before['traces'],after['packets']):
   if pkt['reason']=='compiled':assert a['new_question']==b['new_question'] and a['new_speech']==b['new_speech']
   else:assert a['new_speech']==a['native_speech']
  bad=copy.deepcopy(after);bad.pop('reader_revision')
  try:validate.validate_bundle(j,row,segments,source,bad,True)
  except AssertionError:pass
  else:raise AssertionError('wrong revision accepted')
  cases.append(dict(mode=mode,R1_snapshot_exact=True,packets_exact=True,compiledS_R1_exact=True,unavailableS_native_exact=True,native_allraw_exact=True,clone_exact=True))
for h in hooks:h.remove()
(OUT/'model_summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,pretrained=False,cases=cases,
 scope='actual36-layer narrow randomQwen 8Q/2KVx8, explicit tokenizer/choice fixture; no semantic8B/GT/performance proof'),indent=2)+'\n')
print('R2_READER_MODEL_PASS',flush=True)
