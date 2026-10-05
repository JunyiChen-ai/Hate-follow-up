"""Real36layer cached new-image execution with independent full-position reference."""
import copy
import inspect
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from inputs import ROOT
from src.mllm_judge import Judge,VIDEO_QUESTION
from src.stance_cache import build,positions,margin
from src.source_image_branch import encode_branch,margin as image_margin
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel


def image_pixels(images,dtype):
    pixels=[];grids=[]
    for image in images:
        a=np.asarray(image.convert('RGB')).copy();h,w=a.shape[:2];assert h%4==w%4==0
        p=torch.tensor(a,dtype=dtype).permute(2,0,1)/255
        # Actual3D patch input ordering, with one static image repeated in time.
        patches=p.reshape(3,h//2,2,w//2,2).permute(1,3,0,2,4).reshape(-1,3,4)
        pixels.append(patches[:,:,None,:].expand(-1,-1,2,-1).reshape(-1,24));grids.append([1,h//2,w//2])
    return torch.cat(pixels),torch.tensor(grids)


def fixture(model,frames,dtype,original):
    head=torch.nn.Linear(64,256,bias=False).to(dtype)
    j=Judge.__new__(Judge);j.model=SimpleNamespace(model=model,config=model.config,get_output_embeddings=lambda:head)
    j.device=torch.device('cpu');j.dtype=dtype;j.family='qwen3_vl';j.image_token_id=127
    j.forward_params=set(inspect.signature(model.forward).parameters);j.processor=SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
    j.yes_ids=[70];j.no_ids=[71];j.softcap=None
    ff=[(float(i),original) for i in range(frames)]
    def prefix_messages(frames,segments):return [dict(role='system',content='P')],[p for _,p in frames]
    def encode(text,images):
        pixels,grids=image_pixels(images,dtype);counts=[int(t*h*w)//4 for t,h,w in grids.tolist()]
        ids=[10 if text=='P' else 72]
        for n in counts:ids.extend([125]+[127]*n+[126,73])
        ids.extend([50,51,52] if text=='P' else [74,75])
        ids=torch.tensor([ids]);enc=dict(input_ids=ids,pixel_values=pixels,image_grid_thw=grids,attention_mask=torch.ones_like(ids))
        if 'mm_token_type_ids' in j.forward_params:enc['mm_token_type_ids']=(ids==127).long()
        return enc
    def encode_prefix(msgs,files):
        with Image.open(original) as image:enc=encode('P',[image]*len(files))
        j.img_tokens=[4]*len(files);j._prefix_text='P';return 'P',enc
    def render(msgs,gen):
        s='P'
        for m in msgs[1:]:
            if m['role']=='assistant':s+='A'+m['content']
            elif m['content']==[dict(type='text',text=VIDEO_QUESTION)]:s+='Q'
            else:s+='S'
        return s
    j.prefix_messages=prefix_messages;j.encode_prefix=encode_prefix;j.encode=encode;j.render=render
    j.branch_ids=lambda *a,**k:([60,61],'Q');j.answer_ids=lambda msgs,q,stance:([62],'A'+stance)
    j.turn=lambda role,text:dict(role=role,content=text)
    return j,ff


def main():
    torch.set_num_threads(1);torch.manual_seed(0);print('host',socket.gethostname(),flush=True)
    out=ROOT/'runs/20261005_m1_spatial_search/model_cpu_checks';out.mkdir(parents=True,exist_ok=True);(out/'run.pid').write_text(str(os.getpid()))
    rng=np.random.default_rng(0);original=out/'original.png';cropped=out/'crop.png'
    image=Image.fromarray(rng.integers(0,256,(8,8,3),dtype=np.uint8));image.save(original);crop=image.crop((2,2,6,6));crop.save(cropped);crop.close();image.close()
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=256,hidden_size=64,intermediate_size=128,num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,
        head_dim=128,max_position_embeddings=512,rope_scaling={'rope_type':'default','mrope_section':[24,20,20]}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,
            out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1]),image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa';result=[]
    with torch.no_grad():
        for dtype in (torch.float32,torch.bfloat16):
            for count in (18,20):
                model=Qwen3VLModel(cfg).eval().to(dtype);j,ff=fixture(model,count,dtype,original)
                cache,ctx=build(j,ff,[]);initial=copy.deepcopy(cache);_,base=j.encode_prefix(ctx['msgs'],ctx['files'])
                content=[dict(type='text',text='Current actual source.'),dict(type='image'),dict(type='text',text='Exact original-pixel crop.'),dict(type='image')]
                encoded,p,evidence=encode_branch(j,ctx,'fixture query',content,[original,cropped])
                fullids=torch.cat([base['input_ids'],torch.tensor([[60,61,62]]),encoded['input_ids']],1)
                grids=torch.cat([base['image_grid_thw'],encoded['image_grid_thw']],0);fullp,_=positions(j,fullids,grids)
                assert torch.equal(p,fullp[:,:,ctx['stance_cache_tokens']:])
                z,e=image_margin(j,cache,ctx,'fixture query',content,[original,cropped]);clone=copy.deepcopy(cache)
                z2,e2=image_margin(j,clone,ctx,'fixture query',content,[original,cropped]);assert z==z2 and e==e2==evidence
                assert cache.get_seq_length()==clone.get_seq_length()==initial.get_seq_length()
                assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,initial.layers)) and torch.equal(model.rope_deltas,ctx['rope'])
                assert margin(j,cache,ctx,'native fixture question')==margin(j,initial,ctx,'native fixture question')
                # Full original conversation is an independent un-cached reference.
                kw=dict(input_ids=fullids,pixel_values=torch.cat([base['pixel_values'],encoded['pixel_values']],0),image_grid_thw=grids,position_ids=fullp,use_cache=True)
                if 'mm_token_type_ids' in j.forward_params:kw['mm_token_type_ids']=(fullids==127).long()
                full=model(**kw);reference=j.margins_fp32(full.last_hidden_state[0,-1:])[0];difference=abs(z-reference)
                assert difference<=(1e-4 if dtype==torch.float32 else .025)
                # Same source text/shape with different actual pixels must change reading.
                changed=out/'changed.png';Image.fromarray(np.full((4,4,3),255,dtype=np.uint8)).save(changed)
                different,_=image_margin(j,cache,ctx,'fixture query',content,[original,changed]);assert different!=z
                result.append(dict(dtype=str(dtype),native_frames=count,layers=36,source_images=2,unequal_source_tokens=evidence['image_counts'],
                    whole_position_reference_exact=True,source_clone_exact=True,allKV_restored_exact=True,native_replay_exact=True,
                    full_uncached_margin_difference=difference,full_reference_tolerance=1e-4 if dtype==torch.float32 else .025,actual_pixel_changes_margin=True))
                print(dtype,count,'PASS',difference,flush=True)
    (out/'summary.json').write_text(json.dumps(dict(host=socket.gethostname(),GT_read=False,PASS=True,checks=result),indent=2)+'\n');print('MODEL_CPU_PASS',flush=True)


if __name__=='__main__':main()
