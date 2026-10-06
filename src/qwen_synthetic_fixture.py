"""Shared explicit software chat/token/image fixture; no pretrained/tokenizer claim. Promoted from reviewed Candidate39 fixture on2026-10-07; active39 retains its local frozen copy."""
import inspect
import re
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
from src.mllm_judge import Judge


class Tokens:
    all_special_ids=[125,126,127]
    def __init__(self):self.mapping={'«':125,'»':126,'~':127,'Yes':70,'No':71}
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
        ids=[];offsets=[]
        for match in re.finditer(r'\w+|[^\w\s]',text):
            word=match.group()
            if word not in self.mapping:self.mapping[word]=128+len(self.mapping)
            ids.append(self.mapping[word]);offsets.append(match.span())
        assert max(ids,default=0)<512
        return dict(input_ids=ids,offset_mapping=offsets)


def fixture(model,count,dtype,native_path):
    j=Judge.__new__(Judge);head=torch.nn.Linear(64,512,bias=False).to(dtype)
    j.model=SimpleNamespace(model=model,config=model.config,get_output_embeddings=lambda:head)
    j.device=torch.device('cpu');j.dtype=dtype;j.family='qwen3_vl';j.image_token_id=127;j.img_kw={}
    j.same_turn=False;j.loose_stance_seam=False;j.list_content=False;j.softcap=None;j.yes_ids=[70];j.no_ids=[71]
    j.forward_params=set(inspect.signature(model.forward).parameters);j.tok=Tokens()
    j.processor=SimpleNamespace(image_processor=SimpleNamespace(merge_size=2,patch_size=2))
    def render(msgs,add_generation_prompt):
        text=''
        for message in msgs:
            text+=message['role'].upper()+' '
            content=message['content']
            if isinstance(content,str):text+=content
            else:
                for item in content:
                    if item['type']=='text':text+=item['text']+' '
                    else:
                        with Image.open(item['image']) as image:h,w=image.height,image.width
                        text+='«'+'~'*(h*w//16)+'» '
            text+=' END '
        if add_generation_prompt:text+='ASSISTANT '
        return text
    def encode(text,images):
        ids=torch.tensor([j.tok(text)['input_ids']]);out=dict(input_ids=ids,attention_mask=torch.ones_like(ids))
        if images:
            pixels=[];grids=[]
            for image in images:
                a=np.asarray(image.convert('RGB')).copy();h,w=a.shape[:2];assert h%4==w%4==0
                x=torch.tensor(a,dtype=dtype).permute(2,0,1)/255
                raw=x.reshape(3,h//2,2,w//2,2).permute(1,3,0,2,4).reshape(-1,3,4)
                pixels.append(raw[:,:,None,:].expand(-1,-1,2,-1).reshape(-1,24));grids.append([1,h//2,w//2])
            out.update(pixel_values=torch.cat(pixels),image_grid_thw=torch.tensor(grids))
        return out
    def prefix_messages(frames,segments):
        files=[p for _,p in frames]
        content=[dict(type='text',text='Native overview and literal speech '+str(segments))]
        content.extend(dict(type='image',image=str(p)) for p in files)
        return [dict(role='system',content='Fixture policy'),dict(role='user',content=content)],files
    j.render=render;j.encode=encode;j.prefix_messages=prefix_messages;j.forward_calls=j.vision_calls=0
    hooks=[model.language_model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),
        model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    return j,[(float(i),native_path) for i in range(count)],hooks
