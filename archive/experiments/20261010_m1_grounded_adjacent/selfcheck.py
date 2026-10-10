"""CPU fixture check of grounded.py on a random-weight 36-layer Qwen3-VL shape; plumbing only, no pretrained claim."""
import inspect
import json
import re
import socket
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from grounded import ROOT,SPEC,read
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
    def decode(self,ids,skip_special_tokens=True):
        inv={v:k for k,v in self.mapping.items()};return ' '.join(inv.get(i,'?') for i in ids)


def fixture(model,count,dtype,native_path):
    j=Judge.__new__(Judge);head=torch.nn.Linear(64,512,bias=False).to(dtype)
    j.model=SimpleNamespace(model=model,config=model.config,get_output_embeddings=lambda:head)
    j.device=torch.device('cpu');j.dtype=dtype;j.family='qwen3_vl';j.image_token_id=127;j.img_kw={}
    j.same_turn=False;j.loose_stance_seam=False;j.list_content=False;j.softcap=None;j.yes_ids=[70];j.no_ids=[71];j.eos_ids=set()
    j.forward_params=set(inspect.signature(model.forward).parameters);j.tok=Tokens()
    j.processor=SimpleNamespace(image_processor=SimpleNamespace(merge_size=2,patch_size=2))
    def render(msgs,add_generation_prompt):
        text=''
        for message in msgs:
            text+=message['role'].upper()+' ';content=message['content']
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
                x=torch.tensor(a,dtype=dtype).permute(2,0,1)/255;raw=x.reshape(3,h//2,2,w//2,2).permute(1,3,0,2,4).reshape(-1,3,4)
                pixels.append(raw[:,:,None,:].expand(-1,-1,2,-1).reshape(-1,24));grids.append([1,h//2,w//2])
            out.update(pixel_values=torch.cat(pixels),image_grid_thw=torch.tensor(grids))
        return out
    def prefix_messages(frames,segments):
        files=[p for _,p in frames];content=[dict(type='text',text='Native overview and literal speech '+str(segments))]
        content.extend(dict(type='image',image=str(p)) for p in files)
        return [dict(role='system',content='Fixture policy'),dict(role='user',content=content)],files
    j.render=render;j.encode=encode;j.prefix_messages=prefix_messages;j.forward_calls=j.vision_calls=0
    hooks=[model.language_model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    return j,[(float(i),native_path) for i in range(count)],hooks


def main():
    torch.set_num_threads(4);torch.manual_seed(0);print(socket.gethostname(),flush=True)
    out=ROOT/'runs/20261010_m1_grounded_adjacent/cpu_checks';out.mkdir(parents=True,exist_ok=True)
    native=out/'native.png';Image.fromarray(np.full((8,8,3),100,np.uint8)).save(native)
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=512,hidden_size=64,intermediate_size=128,num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,head_dim=128,
        max_position_embeddings=4096,rope_parameters={'rope_type':'default','rope_theta':5000000.,'mrope_section':[24,20,20],'mrope_interleaved':True}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),
        image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa'
    row=dict(dataset='HateMM',video_id='fixture',duration=24.);segments=[(0.,8.,'Fixture speech'),(8.,16.,'Later speech')];cases=[]
    with torch.no_grad():
        for dtype,count in ((torch.float32,20),(torch.bfloat16,18)):
            model=Qwen3VLModel(cfg).eval().to(dtype);model.requires_grad_(False);j,ff,hooks=fixture(model,count,dtype,native)
            try:
                with patch('grounded.frame_paths',return_value=ff):b=read(j,row,segments)
                tr=b['traces'];assert len(tr)==3 and [len(t['shown_times']) for t in tr]==[8,8,count-16]
                for t in tr:
                    assert t['adjacent'] is not None and t['generated']==SPEC['max_new_tokens'] and isinstance(t['reply'],str)
                    v,z,g=t['native_visual'],t['adjacent'],t['grounded'];acc=(z>v and g) or (z<=v and not g)
                    assert t['accepted']==acc and t['final']==(z if acc else v) and z!=v
                    assert g==any(abs(x-u)<=SPEC['match_tolerance']+1e-9 for x in t['numbers'] for u in t['shown_times'])
                for name,key in (('accept_all','adjacent'),('grounded','final')):
                    assert [w['z_visual'] for w in b['predictions'][name]['extra']['windows']]==[t[key] for t in tr]
                c=b['checks'];assert c['covered']==3 and c['actual_forwards']==b['predictions']['base']['calls']+6+c['generated_tokens'] and c['actual_vision']==4
                cases.append(dict(dtype=str(dtype),native_frames=count,generated=c['generated_tokens'],accepted=c['accepted'],grounded=c['grounded']))
                (out/(str(dtype).split('.')[-1]+'.json')).write_text(json.dumps(b)+'\n');print('C41_CASE_PASS',dtype,count,flush=True)
            finally:
                for hook in hooks:hook.remove()
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,pretrained=False,cases=cases,scope='random-weight fixture; plumbing only'),indent=2)+'\n');print('C41_CPU_PASS',flush=True)


if __name__=='__main__':main()
