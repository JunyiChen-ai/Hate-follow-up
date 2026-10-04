"""Native Qwen moderation/stance cache with independent local branches."""
import inspect
import torch
from .mllm_judge import VIDEO_QUESTION


def positions(j,ids,grids):
    kw=dict(input_ids=ids,image_grid_thw=grids,attention_mask=torch.ones_like(ids))
    if 'mm_token_type_ids' in inspect.signature(j.model.model.get_rope_index).parameters:
        kw['mm_token_type_ids']=(ids==j.image_token_id).to(torch.int32)
    return j.model.model.get_rope_index(**kw)


@torch.no_grad()
def build(j,frames,segments):
    j.model.model.rope_deltas=None
    msgs,files=j.prefix_messages(frames,segments);text,enc=j.encode_prefix(msgs,files)
    cache=j.prefix_cache(enc);qid,qtext=j.branch_ids(msgs,VIDEO_QUESTION)
    z=j.cached_margin(cache,qid,in_place=True);stance='Yes' if z>0 else 'No'
    aid,atext=j.answer_ids(msgs,VIDEO_QUESTION,stance);j.extend_cache(cache,aid)
    history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',stance)]
    ids=torch.cat((enc['input_ids'].to(j.device),torch.tensor([qid+aid],device=j.device)),1)
    p,delta=positions(j,ids,enc['image_grid_thw'].to(j.device))
    assert ids.shape[1]==cache.get_seq_length() and torch.equal(delta,j.model.model.rope_deltas)
    return cache,dict(msgs=msgs,files=files,history=history,head=text+qtext+atext,
        positions=p,rope=delta.clone(),global_margin=z,stance=stance,
        stance_cache_tokens=cache.get_seq_length(),stance_cache_logical_start=int(p.max())+1,
        prefix_tokens=enc['input_ids'].shape[1],image_counts=list(j.img_tokens))


@torch.no_grad()
def margin(j,cache,ctx,question):
    n=cache.get_seq_length();j.model.model.rope_deltas=ctx['rope'].clone()
    ids,_=j.branch_ids(ctx['msgs'],question,ctx['history'],head_text=ctx['head'])
    try:return j.cached_margin(cache,ids,in_place=True)
    finally:cache.crop(n);j.model.model.rope_deltas=ctx['rope'].clone()
