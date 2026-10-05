"""Same frozen Qwen text hidden-state embeddings; no second encoder or labels."""
import time
import torch
from retrieval import SPEC


def encode_content(j,text):
    assert isinstance(text,str) and text.strip()
    messages=[j.turn('system',SPEC['embedding_system']),j.turn('user',text)]
    rendered=j.render(messages,False)
    # Locate user content through the full exact role prefix, not a substring
    # lookup that could accidentally find the same words in the system prompt.
    empty=j.render([j.turn('system',SPEC['embedding_system']),j.turn('user','')],False)
    tail='<|im_end|>\n';assert empty.endswith(tail)
    prefix=empty[:-len(tail)];assert rendered==prefix+text+tail
    beginning=len(prefix);end=beginning+len(text)
    encoded=j.encode(rendered,[])
    offsets=j.tok(rendered,add_special_tokens=False,return_offsets_mapping=True)
    assert encoded['input_ids'][0].tolist()==offsets['input_ids']
    special=set(j.tok.all_special_ids)
    indices=[i for i,((a,b),token) in enumerate(zip(offsets['offset_mapping'],offsets['input_ids']))
        if b>a and a<end and b>beginning and token not in special]
    assert indices,'no actual user-content tokens to pool'
    return encoded,dict(prompt=rendered,text=text,body_character_range=[beginning,end],
        input_tokens=offsets['input_ids'],pooled_token_indices=indices)


@torch.no_grad()
def embed(j,text):
    if j.device.type=='cuda':torch.cuda.synchronize()
    start=time.perf_counter();before=j.forward_calls
    encoded,evidence=encode_content(j,text);old=j.model.model.rope_deltas
    j.model.model.rope_deltas=None
    try:
        output=j.model.model(**j.model_inputs(encoded),use_cache=False)
        hidden=output.last_hidden_state[0,evidence['pooled_token_indices']].float()
        mean=hidden.mean(0);norm=mean.norm();available=bool(torch.isfinite(mean).all() and norm>SPEC['embedding_epsilon'])
        vector=(mean/norm).cpu().tolist() if available else None
        del output,hidden
    finally:j.model.model.rope_deltas=old
    if j.device.type=='cuda':torch.cuda.synchronize()
    assert j.forward_calls-before==1
    return dict(**evidence,vector=vector,available=available,actual_forwards=1,seconds=time.perf_counter()-start)


def validate_embedding(j,e,text):
    _,expected=encode_content(j,text)
    assert all(e[k]==v for k,v in expected.items())
    assert e['actual_forwards']==1 and type(e['available']) is bool and e['seconds']>=0
    if e['available']:
        vector=torch.tensor(e['vector'],dtype=torch.float32)
        assert torch.isfinite(vector).all() and abs(float(vector.norm())-1)<1e-5
    else:assert e['vector'] is None
