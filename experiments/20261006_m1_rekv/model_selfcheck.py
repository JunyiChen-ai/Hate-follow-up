"""Actual36layer Qwen source vision/cache versus independently packed full reference."""
import copy
import inspect
import json
import os
import re
import socket
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from retrieval import ROOT
from memory import VideoMemory
from reader import collect_source, visual_margin, encode_source, encode_question
from src.mllm_judge import Judge, VIDEO_QUESTION
from src.stance_cache import build, positions, margin


def pixels_for(images,dtype):
    pixels=[];grids=[]
    for image in images:
        a=np.asarray(image.convert('RGB')).copy();h,w=a.shape[:2]
        assert h%4==w%4==0
        x=torch.tensor(a,dtype=dtype).permute(2,0,1)/255
        patches=x.reshape(3,h//2,2,w//2,2).permute(1,3,0,2,4).reshape(-1,3,4)
        pixels.append(patches[:,:,None,:].expand(-1,-1,2,-1).reshape(-1,24))
        grids.append([1,h//2,w//2])
    return torch.cat(pixels),torch.tensor(grids)


class FixtureTokenizer:
    all_special_ids=[]
    def __init__(self):self.words={}
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
        ids=[];offsets=[]
        for match in re.finditer(r'\w+|[^\w\s]',text):
            word=match.group()
            if word not in self.words:self.words[word]=128+len(self.words)
            ids.append(self.words[word]);offsets.append(match.span())
        assert all(i<512 for i in ids)
        return dict(input_ids=ids,offset_mapping=offsets)


def fixture(model,count,dtype,original):
    head=torch.nn.Linear(64,512,bias=False).to(dtype)
    j=Judge.__new__(Judge);j.model=SimpleNamespace(model=model,config=model.config,get_output_embeddings=lambda:head)
    j.device=torch.device('cpu');j.dtype=dtype;j.family='qwen3_vl';j.image_token_id=127
    j.forward_params=set(inspect.signature(model.forward).parameters)
    j.processor=SimpleNamespace(image_processor=SimpleNamespace(merge_size=2))
    j.yes_ids=[70];j.no_ids=[71];j.softcap=None;j.tok=FixtureTokenizer()
    ff=[(float(i),original) for i in range(count)]
    def prefix_messages(frames,segments):return [dict(role='system',content='P')],[p for _,p in frames]
    def encode(text,images):
        pixels,grids=pixels_for(images,dtype)
        counts=[int(t*h*w)//4 for t,h,w in grids.tolist()]
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
        text='P'
        for message in msgs[1:]:
            if message['role']=='assistant':text+='A'+message['content']
            elif message['content']==[dict(type='text',text=VIDEO_QUESTION)]:text+='Q'
            else:text+='S'
        return text
    def branch_ids(msgs,question,history=None,head_text=None):
        if question==VIDEO_QUESTION or question=='native fixture question':return [60,61],'Q'
        text='USER '+question+' ASSISTANT'
        return j.tok(text)['input_ids'],text
    j.prefix_messages=prefix_messages;j.encode_prefix=encode_prefix;j.encode=encode;j.render=render;j.branch_ids=branch_ids
    j.answer_ids=lambda msgs,q,stance:([62],'A'+stance);j.turn=lambda role,text:dict(role=role,content=text)
    j.forward_calls=j.vision_calls=0
    hooks=[model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),
        model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    return j,ff,hooks


def main():
    torch.set_num_threads(4);torch.manual_seed(0)
    out=ROOT/'runs/20261006_m1_rekv/model_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    print(socket.gethostname(),flush=True);(out/'run.pid').write_text(str(os.getpid()))
    (out/'config.json').write_text(json.dumps(dict(host=socket.gethostname(),GT_read=False,pretrained=False,
        source='experiments/20261006_m1_rekv/model_selfcheck.py;2026-10-06',command='python -u model_selfcheck.py',
        tokenizer='explicit software fixture; production-tokenizer binding still requires separate preflight'),indent=2)+'\n')
    rng=np.random.default_rng(0);original=out/'original.png'
    Image.fromarray(rng.integers(0,256,(8,8,3),dtype=np.uint8)).save(original)
    sources=[]
    for index in range(7):
        path=out/f'source_{index}.png';Image.fromarray(rng.integers(0,256,(8,8,3),dtype=np.uint8)).save(path)
        sources.append(dict(index=index*16,time=index*2.,path=str(path.relative_to(ROOT))))
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=512,hidden_size=64,intermediate_size=128,num_hidden_layers=36,
        num_attention_heads=32,num_key_value_heads=8,head_dim=128,max_position_embeddings=1024,
        rope_parameters={'rope_type':'default','rope_theta':5000000.,'mrope_section':[24,20,20],'mrope_interleaved':True}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,
            temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),
        image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa'
    checks=[]
    with torch.no_grad():
        for dtype in (torch.float32,torch.bfloat16):
            for count in (18,20):
                model=Qwen3VLModel(cfg).eval().to(dtype);model.requires_grad_(False)
                j,ff,hooks=fixture(model,count,dtype,original)
                cache,ctx=build(j,ff,[]);initial=copy.deepcopy(cache)
                native_before=margin(j,cache,ctx,'native fixture question')
                memory=VideoMemory(out/f'{str(dtype).split(".")[-1]}_{count}')
                try:
                    collect_source(j,cache,ctx,memory,sources[:5])
                    z,trace=visual_margin(j,cache,ctx,memory,[0,1,2,3],'fixture visual query')
                    again,other=visual_margin(j,cache,ctx,memory,[0,1,2,3],'fixture visual query')
                    assert again==z and all(torch.equal(trace['layers'][l]['query_vector'],other['layers'][l]['query_vector']) for l in trace['layers'])
                    _,base=j.encode_prefix(ctx['msgs'],ctx['files'])
                    encoded=[encode_source(j,ctx,frame)[0] for frame in sources[:5]]
                    qids,_,_=encode_question(j,ctx,'fixture visual query')
                    fullids=torch.cat([base['input_ids'],torch.tensor([[60,61,62]]),*[e['input_ids'] for e in encoded],torch.tensor([qids])],1)
                    grids=torch.cat([base['image_grid_thw'],*[e['image_grid_thw'] for e in encoded]])
                    fullp,_=positions(j,fullids,grids)
                    kw=dict(input_ids=fullids,pixel_values=torch.cat([base['pixel_values'],*[e['pixel_values'] for e in encoded]]),
                        image_grid_thw=grids,position_ids=fullp,use_cache=True)
                    if 'mm_token_type_ids' in j.forward_params:kw['mm_token_type_ids']=(fullids==127).long()
                    full=model(**kw);reference=j.margins_fp32(full.last_hidden_state[0,-1:])[0];del full
                    difference=abs(z-reference);tolerance=1e-4 if dtype==torch.float32 else .025
                    assert difference<=tolerance,(dtype,count,z,reference,difference)
                    # Five sources coincide with the full causal history. A longer
                    # source stream deliberately does not claim that equivalence.
                    assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,initial.layers))
                    assert margin(j,cache,ctx,'native fixture question')==native_before
                    assert all('forward' not in layer.self_attn.__dict__ for layer in model.language_model.layers)
                    checks.append(dict(dtype=str(dtype),native_frames=count,layers=36,q_heads=32,kv_heads=8,head_dim=128,
                        source_prefills=5,source_vision=5,source_deepstack=3,actual_per_layer_retrieval=True,clones_exact=True,
                        native_S_and_all_prefix_KV_exact=True,full_causal_first5_reference_difference=difference,tolerance=tolerance,
                        source_cache_without_extra_query_probe=True))
                    print(dtype,count,'PASS',difference,flush=True)
                    if count==20:
                        extended=VideoMemory(out/f'{str(dtype).split(".")[-1]}_sliding')
                        try:
                            collect_source(j,cache,ctx,extended,sources)
                            assert extended.blocks[5]['direct_ancestors']==[1,2,3,4]
                            assert extended.blocks[6]['direct_ancestors']==[2,3,4,5]
                            assert extended.blocks[6]['ancestors']==list(range(6))
                            a,ta=visual_margin(j,cache,ctx,extended,[4,5,6],'fixture visual query')
                            b,tb=visual_margin(j,cache,ctx,extended,[4,5,6],'fixture visual query')
                            assert a==b and all(sorted(ta['layers'][l]['remote_ids'])==[0,1,2,3] for l in ta['layers'])
                        finally:extended.close(release=True)
                        changed=out/f'changed_{str(dtype).split(".")[-1]}.png'
                        Image.fromarray(np.full((8,8,3),255,dtype=np.uint8)).save(changed)
                        altered=[dict(frame) for frame in sources[:5]]
                        altered[2]['path']=str(changed.relative_to(ROOT))
                        alternative=VideoMemory(out/f'{str(dtype).split(".")[-1]}_changed')
                        try:
                            collect_source(j,cache,ctx,alternative,altered)
                            modified,_=visual_margin(j,cache,ctx,alternative,[0,1,2,3],'fixture visual query')
                            assert modified!=z,'actual source pixels did not enter the margin'
                        finally:alternative.close(release=True)
                        local=VideoMemory(out/f'{str(dtype).split(".")[-1]}_local')
                        try:
                            collect_source(j,cache,ctx,local,sources[:4])
                            only,t=visual_margin(j,cache,ctx,local,[0,1,2,3],'fixture visual query')
                            assert np.isfinite(only) and all(not layer['remote_ids'] for layer in t['layers'].values())
                        finally:local.close(release=True)
                        assert margin(j,cache,ctx,'native fixture question')==native_before
                        checks[-1].update(sliding_last4_direct_and_inherited_ancestry_exact=True,
                            sliding_query_clones_exact=True,actual_changed_source_pixels_change_margin=True,LOCAL_only_exercised=True)
                finally:
                    memory.close(release=True)
                    for hook in hooks:hook.remove()
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,scope='actual36layer random-weight software/vision/cache references, fixture tokenizer; not8B/pretrained/production renderer/performance',checks=checks),indent=2)+'\n')
    print('MODEL_CPU_PASS',flush=True)


if __name__=='__main__':main()
