"""Real native tokenizer/processor and36 layers: shared clip cache vs full read."""
import copy
import inspect
import json
import os
import socket
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from timeline import ROOT,SPEC
from relevance import read_clip,validate_clip,encode_prefix
from src.mllm_renderer import cpu_renderer
from src.stance_cache import positions
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel


def main():
    out=ROOT/'runs/20261006_m1_vtimecot/relevance_cpu_checks';out.mkdir(parents=True,exist_ok=True);(out/'run.pid').write_text(str(os.getpid()));print('host',socket.gethostname(),flush=True)
    j=cpu_renderer();j.img_kw=dict(size=dict(shortest_edge=1024,longest_edge=2048));torch.set_num_threads(1);torch.manual_seed(0)
    folder=out/'source';(folder/'frames').mkdir(parents=True,exist_ok=True)
    for i in range(8):Image.fromarray(np.full((32,32,3),i*25,np.uint8)).save(folder/'frames'/f'frame_{i:08d}.png')
    source=dict(duration=8.,entries=[dict(index=i,time=float(i)) for i in range(8)]);window=dict(i=0,start=0.,end=8.,body='A cup moves near a visible sign.',sample_ids=list(range(8)))
    patch=j.processor.image_processor.patch_size
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=len(j.tok),hidden_size=64,intermediate_size=128,num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,head_dim=128,max_position_embeddings=2048,rope_scaling={'rope_type':'default','mrope_section':[24,20,20]}),vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=patch,spatial_merge_size=2,temporal_patch_size=2,out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1]),image_token_id=j.image_token_id,video_token_id=151656,vision_start_token_id=151652,vision_end_token_id=151653)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa';results=[]
    for dtype in (torch.float32,torch.bfloat16):
        model=Qwen3VLModel(cfg).eval().to(dtype);head=torch.nn.Linear(64,len(j.tok),bias=False).to(dtype);j.model=SimpleNamespace(model=model,config=cfg,get_output_embeddings=lambda:head);j.dtype=dtype;j.softcap=None;j.forward_params=set(inspect.signature(model.forward).parameters);j.forward_calls=j.vision_calls=0;outputs=[];snapshot=[]
        def record(module,args,out):
            outputs.append(out.last_hidden_state[0,-1].detach().float().clone())
            if len(outputs)==1:snapshot.append(copy.deepcopy(out.past_key_values))
            if len(outputs)>=2:
                for old,new in zip(snapshot[0].layers,out.past_key_values.layers):assert torch.equal(old.keys,new.keys[:,:,:old.keys.shape[2]]) and torch.equal(old.values,new.values[:,:,:old.values.shape[2]])
        hooks=[model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1)),model.register_forward_hook(record)]
        with torch.no_grad():recorded=read_clip(j,window,source,folder,['moving cup','visible sign'])
        for hook in hooks:hook.remove()
        validate_clip(j,recorded,window,source,folder,['moving cup','visible sign']);assert j.vision_calls==1 and j.forward_calls==1+sum(g['actual_forwards'] for g in recorded['generations'])
        _,_,_,enc=encode_prefix(j,window,source,folder);delta=model.rope_deltas.clone();cursor=1;differences=[]
        for g in recorded['generations']:
            for k in range(len(g['tokens'])+1):
                ids=torch.tensor([g['input_tokens']+g['tokens'][:k]])
                p,expected_delta=positions(j,ids,enc['image_grid_thw']);assert int(expected_delta[0,0])==recorded['rope_delta']
                kw=dict(enc);kw['input_ids']=ids;kw['attention_mask']=torch.ones_like(ids);kw['position_ids']=p
                if 'mm_token_type_ids' in kw:kw['mm_token_type_ids']=(ids==j.image_token_id).long()
                with torch.no_grad():fresh=model(**j.model_inputs(kw),use_cache=False).last_hidden_state[0,-1].float()
                difference=float((fresh-outputs[cursor]).abs().max());assert difference<=(1e-4 if dtype==torch.float32 else .05);differences.append(difference);cursor+=1
        assert cursor==len(outputs)
        results.append(dict(dtype=str(dtype),layers=36,source_images=8,native_processor_tokenizer=True,test_processor_resolution_only=[1024,2048],prefix_vision_calls=1,queries=2,actual_forwards=j.forward_calls,whole_positions_exact=True,all_generated_states_full_reference_max_difference=max(differences)))
        print('RELEVANCE_CASE_PASS',dtype,max(differences),flush=True)
        del snapshot,outputs,model,head
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,scope='randomweights/CPU and deliberately small processor geometry, not pretrained accuracy',checks=results),indent=2)+'\n');print('RELEVANCE_CPU_PASS',flush=True)


if __name__=='__main__':main()
