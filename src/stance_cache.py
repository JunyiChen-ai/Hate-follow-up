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


# ---- 2026-11-01: one free-form turn written onto the shared context (experiments/20261101_video_plan).
# build() and margin() above are unchanged; the window reads of the adopted reader never call these two.
@torch.no_grad()
def generate_turn(j,cache,ctx,question,max_new_tokens):
    """Greedy answer to `question` asked after the shared context (prefix + verdict turn). Generated in place on the
    cache, which is cropped back afterwards. Returns (text, generated token count, hit max_new_tokens)."""
    n=cache.get_seq_length();assert n==ctx['stance_cache_tokens']
    j.model.model.rope_deltas=ctx['rope'].clone()
    ids,_=j.branch_ids(ctx['msgs'],question,ctx['history'],head_text=ctx['head'])
    try:text,gen=j.cached_generate(cache,ids,max_new_tokens,in_place=True)
    finally:cache.crop(n);j.model.model.rope_deltas=ctx['rope'].clone()
    return text,len(gen),len(gen)>=max_new_tokens


@torch.no_grad()
def append_turn(j,cache,ctx,question,answer):
    """Write `question` and `answer` (as the model's own turn) onto the shared context, in place. Returns a new ctx
    whose history / head / cache length / logical start include the turn; later branches (margin, frames_margin)
    use that ctx unchanged. The answer text is tokenised by the chat template, the same way for any source text."""
    n=cache.get_seq_length();assert n==ctx['stance_cache_tokens']
    j.model.model.rope_deltas=ctx['rope'].clone()
    qid,qtext=j.branch_ids(ctx['msgs'],question,ctx['history'],head_text=ctx['head'])
    aid,atext=j.answer_ids(ctx['msgs'],question,answer,ctx['history'])
    assert atext.startswith(answer) and not j.loose_stance_seam,'template re-rendered the answer turn'
    j.extend_cache(cache,qid+aid)
    history=ctx['history']+[{'role':'user','content':[{'type':'text','text':question}]},j.turn('assistant',answer)]
    head=ctx['head']+qtext+atext
    assert j.render(j.conv(ctx['msgs'],'x',history),True).startswith(head)
    delta=int(ctx['rope'].reshape(-1)[0]);m=cache.get_seq_length()
    assert m==n+len(qid)+len(aid) and n+delta==ctx['stance_cache_logical_start']
    return dict(ctx,history=history,head=head,stance_cache_tokens=m,stance_cache_logical_start=m+delta,turn_tokens=len(qid)+len(aid))
