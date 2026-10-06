"""Actual source encoding, per-layer memory retrieval, one independent V margin."""
import time
from pathlib import Path
from PIL import Image
import torch
from transformers.cache_utils import DynamicCache
from retrieval import ROOT, SPEC
from layout import pack_positions, rotate_key, suffix_positions
from attention import capture_source, attention_scope, retrieval_factory
from src.stance_cache import positions


def clock(j):
    if j.device.type == 'cuda':
        torch.cuda.synchronize()
    return time.perf_counter()


def source_message(frame):
    return dict(role='user',content=[
        dict(type='text',text=SPEC['source_text'].format(time=frame.get('presented_time',frame['time']))),
        dict(type='image',image=frame['path'])])


def encode_source(j,ctx,frame):
    message=source_message(frame)
    full=j.render(ctx['msgs']+ctx['history']+[message],False)
    assert full.startswith(ctx['head'])
    suffix=full[len(ctx['head']):]
    path=ROOT/frame['path']
    with Image.open(path) as image:
        image=image.convert('RGB')
        try:encoded=j.encode(suffix,[image])
        finally:image.close()
    ids=encoded['input_ids'][0].tolist()
    image_rows=[i for i,token in enumerate(ids) if token==j.image_token_id]
    assert len(encoded['image_grid_thw'])==1 and image_rows
    relative,_=positions(j,encoded['input_ids'].to(j.device),encoded['image_grid_thw'].to(j.device))
    evidence=dict(message=message,suffix_text=suffix,suffix_ids=ids,
        image_grid=encoded['image_grid_thw'].tolist(),image_rows=image_rows,
        relative_positions=relative[:,0].tolist())
    return encoded,relative,image_rows,evidence


def history_cache(j,native_cache,ctx,memory,history_ids):
    packed,end=pack_positions([memory.blocks[index]['positions'] for index in history_ids],ctx['stance_cache_logical_start'])
    rotary=j.model.model.language_model.rotary_emb
    cache=DynamicCache()
    for layer,prefix in enumerate(native_cache.layers):
        keys=[prefix.keys];values=[prefix.values]
        for index,position in zip(history_ids,packed):
            key,value,_=memory.layer(index,layer,j.device)
            keys.append(rotate_key(key,position,rotary));values.append(value)
        cache.update(torch.cat(keys,dim=2),torch.cat(values,dim=2),layer)
    return cache,end


@torch.no_grad()
def collect_source(j,native_cache,ctx,memory,frames,previous_frames=None):
    start=clock(j)
    before=j.forward_calls
    before_vision=j.vision_calls
    saved_rope=j.model.model.rope_deltas.clone()
    records=[]
    previous_frames=SPEC['source_previous_frames'] if previous_frames is None else previous_frames
    assert previous_frames in (0,SPEC['source_previous_frames'])
    try:
        for index,frame in enumerate(frames):
            history_ids=list(range(max(0,index-previous_frames),index))
            cache,end=history_cache(j,native_cache,ctx,memory,history_ids)
            encoded,relative,image_rows,evidence=encode_source(j,ctx,frame)
            position=relative+end
            kwargs=j.model_inputs(encoded)
            kwargs.pop('attention_mask',None)
            j.model.model.rope_deltas=ctx['rope'].clone()
            physical_start=cache.get_seq_length()
            with capture_source(j) as (keys,values):
                out=j.model.model(**kwargs,position_ids=position,past_key_values=cache,use_cache=True)
                assert out.past_key_values is cache
                assert cache.get_seq_length()==physical_start+len(evidence['suffix_ids'])
                del out
            block=memory.append(frame['index'],frame['time'],position,image_rows,
                [keys[layer] for layer in range(len(keys))],
                [values[layer] for layer in range(len(values))],history_ids,evidence)
            records.append(dict(block_id=block['block_id'],history_ids=history_ids,
                logical_start=end,physical_start=physical_start,suffix_tokens=len(evidence['suffix_ids'])))
            del keys,values,cache,encoded
        assert j.forward_calls-before==len(frames) and j.vision_calls-before_vision==len(frames)
        assert native_cache.get_seq_length()==ctx['stance_cache_tokens']
        return dict(source_forwards=len(frames),source_vision=len(frames),
            seconds=clock(j)-start,records=records)
    finally:
        j.model.model.rope_deltas=saved_rope


def encode_question(j,ctx,question):
    content=SPEC['reader_role_text']+'\n\n'+question
    ids,text=j.branch_ids(ctx['msgs'],content,ctx['history'],head_text=ctx['head'])
    encoding=j.tok(text,add_special_tokens=False,return_offsets_mapping=True)
    assert encoding['input_ids']==ids
    start=text.index(question)
    end=start+len(question)
    rows=[i for i,(a,b) in enumerate(encoding['offset_mapping']) if a<end and b>start]
    assert rows and all(ids[row] not in j.tok.all_special_ids for row in rows)
    return ids,rows,dict(suffix_text=text,suffix_ids=ids,question_start=start,
        question_end=end,question_rows=rows,offsets=[list(pair) for pair in encoding['offset_mapping']])


@torch.no_grad()
def visual_margin(j,native_cache,ctx,memory,local_ids,question,selection=None):
    assert local_ids and native_cache.get_seq_length()==ctx['stance_cache_tokens']
    start=clock(j)
    first=j.forward_calls;vision=j.vision_calls
    ids,rows,evidence=encode_question(j,ctx,question)
    trace={}
    factory=retrieval_factory(j,memory,native_cache,ctx,local_ids,rows,trace,selection=selection)
    position=suffix_positions(len(ids),ctx['stance_cache_logical_start'],j.device)
    saved_rope=j.model.model.rope_deltas.clone()
    try:
        with attention_scope(j,factory):
            out=j.model.model(input_ids=torch.tensor([ids],device=j.device),position_ids=position,
                past_key_values=native_cache,use_cache=True)
            hidden=out.last_hidden_state[0,-1].clone();del out
        result=j.margins_fp32(hidden[None])[0]
        assert len(trace)==len(native_cache.layers)
        assert j.forward_calls-first==1 and j.vision_calls-vision==0
        assert native_cache.get_seq_length()==ctx['stance_cache_tokens']
        return result,dict(input=evidence,layers=trace,seconds=clock(j)-start,
            actual_forwards=1,actual_vision=0)
    finally:
        j.model.model.rope_deltas=saved_rope
