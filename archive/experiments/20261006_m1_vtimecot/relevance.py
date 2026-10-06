"""One actual clip prefill shared by independent same-model query branches."""
import time
import torch
from PIL import Image
from timeline import ROOT,SPEC
from interface import relevance_writer,compile_relevance
from src.structured_source_generation import Stream,Capped,image_rope_delta


def prefix_messages(j,window,source,folder):
    from inputs import media_content
    items,paths=media_content(source,window['sample_ids'],folder,save=False)
    items.insert(0,dict(type='text',text=f"Actual clip [{window['start']},{window['end']}); literal ASR:\n{window['body']}\nNo relevance query has yet been supplied."))
    return [j.turn('system',SPEC['relevance_system']),dict(role='user',content=items),j.turn('assistant',SPEC['source_acknowledgement'])],paths


def encode_prefix(j,window,source,folder):
    msgs,paths=prefix_messages(j,window,source,folder);head=j.render(msgs,False)
    images=[Image.open(ROOT/p).convert('RGB') for p in paths]
    try:enc=j.encode(head,images)
    finally:
        for image in images:image.close()
    return msgs,paths,head,enc


def query_suffix(j,msgs,head,query):
    text=j.render(msgs+[j.turn('user','Observable search condition (not a fact): '+query+'\nReturn bounded clip relevance 0–10 or UNKNOWN.')],True)
    assert text.startswith(head);ids=j.tok.encode(text[len(head):],add_special_tokens=False);assert ids
    return text,ids


def clock(j):
    if j.device.type=='cuda':torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def read_clip(j,window,source,folder,queries):
    if not queries or not window['sample_ids']:return None
    start=clock(j);before=j.forward_calls;vision=j.vision_calls;msgs,paths,head,enc=encode_prefix(j,window,source,folder)
    j.model.model.rope_deltas=None;out=j.model.model(**j.model_inputs(enc),use_cache=True);cache=out.past_key_values;del out
    n=cache.get_seq_length();delta=j.model.model.rope_deltas.clone();logical=int(delta[0,0]);assert logical==image_rope_delta(j,enc)
    prefix_seconds=clock(j)-start;assert j.forward_calls-before==1 and j.vision_calls-vision==1
    record=dict(head=head,paths=paths,input_tokens=enc['input_ids'][0].tolist(),image_grid=enc['image_grid_thw'].tolist(),prefix_tokens=n,rope_delta=logical,
        prefix_seconds=prefix_seconds,prefix_forwards=1,prefix_vision=1,generations=[])
    if hasattr(j,'structured_W32'):del j.structured_W32
    j.structured_W32=j.model.get_output_embeddings().weight.float()
    for query in queries:
        start=clock(j);before=j.forward_calls;text,ids=query_suffix(j,msgs,head,query);j.model.model.rope_deltas=delta.clone()
        try:
            p=torch.arange(n+logical,n+logical+len(ids),device=j.device)[None,None,:].expand(3,1,-1)
            output=j.model.model(input_ids=torch.tensor([ids],device=j.device),past_key_values=cache,position_ids=p,use_cache=True)
            hidden=output.last_hidden_state[0,-1].clone();del output
            stream=Stream(j,SPEC['relevance_generation_tokens'],hidden,cache,rope_delta=logical);del hidden;complete=True;value=None
            try:value=relevance_writer(stream)
            except Capped:complete=False
            g=dict(query=query,prompt=text,input_tokens=record['input_tokens']+ids,tokens=stream.tokens,events=stream.events,selection=value,truncated=not complete,
                text=j.tok.decode(stream.tokens,skip_special_tokens=True).strip(),max_tokens=SPEC['relevance_generation_tokens'],prefill_cache_tokens=n+len(ids),rope_delta=logical,positions=stream.positions,
                actual_forwards=j.forward_calls-before,actual_vision_forwards=0,seconds=clock(j)-start)
            assert g['actual_forwards']==1+len(stream.tokens) and stream.positions==list(range(n+len(ids)+logical,n+len(ids)+logical+len(stream.tokens)))
            record['generations'].append(g)
        finally:cache.crop(n);j.model.model.rope_deltas=delta.clone()
        assert cache.get_seq_length()==n
    del cache,j.structured_W32;return record


def validate_clip(j,record,window,source,folder,queries):
    if not queries or not window['sample_ids']:assert record is None;return
    msgs,paths,head,enc=encode_prefix(j,window,source,folder)
    assert record['head']==head and record['paths']==paths and record['input_tokens']==enc['input_ids'][0].tolist() and record['image_grid']==enc['image_grid_thw'].tolist()
    assert record['prefix_tokens']==len(record['input_tokens']) and record['rope_delta']==image_rope_delta(j,enc)
    assert record['prefix_forwards']==record['prefix_vision']==1 and record['prefix_seconds']>=0 and len(record['generations'])==len(queries)
    for query,g in zip(queries,record['generations']):
        text,ids=query_suffix(j,msgs,head,query);assert g['query']==query and g['prompt']==text and g['input_tokens']==record['input_tokens']+ids
        assert g['prefill_cache_tokens']==len(g['input_tokens']) and g['rope_delta']==record['rope_delta'] and g['max_tokens']==SPEC['relevance_generation_tokens']
        stream=Stream(j,g['max_tokens'],tokens=g['tokens']);complete=True;value=None
        try:value=relevance_writer(stream)
        except Capped:complete=False
        assert stream.tokens==g['tokens'] and stream.events==g['events'] and value==g['selection'] and g['truncated']==(not complete)
        assert g['positions']==list(range(g['prefill_cache_tokens']+g['rope_delta'],g['prefill_cache_tokens']+g['rope_delta']+len(g['tokens'])))
        assert g['actual_forwards']==1+len(g['tokens']) and g['actual_vision_forwards']==0 and g['seconds']>=0 and j.tok.decode(g['tokens'],skip_special_tokens=True).strip()==g['text']
