"""Direct causal query access to specified current native image-source keys."""
from contextlib import contextmanager
import torch


@contextmanager
def observe_prefix(j):
    original=j.encode_prefix; observed={}
    def encode(*args,**kwargs):
        text,encoded=original(*args,**kwargs)
        assert not observed
        observed.update(input_ids=encoded['input_ids'][0].tolist(),image_counts=list(j.img_tokens))
        return text,encoded
    j.encode_prefix=encode
    try:yield observed
    finally:j.encode_prefix=original


def image_groups(input_ids,image_token_id,counts):
    keys=[i for i,t in enumerate(input_ids) if t==image_token_id]
    assert sum(counts)==len(keys) and all(type(c) is int and c>0 for c in counts)
    result=[];offset=0
    for count in counts:
        group=keys[offset:offset+count]
        assert group==list(range(group[0],group[0]+count))
        result.append(group);offset+=count
    return result


def binding(input_ids,image_token_id,counts,frames):
    groups=image_groups(input_ids,image_token_id,counts)
    assert frames==sorted(set(frames)) and all(type(f) is int and 0<=f<len(groups) for f in frames)
    kept=[k for f in frames for k in groups[f]]
    dropped=[k for f,g in enumerate(groups) if f not in frames for k in g]
    return dict(frames=frames,kept_visual_keys=kept,dropped_visual_keys=dropped,prefix_tokens=len(input_ids))


@torch.no_grad()
def margin(j,cache,ctx,question,source):
    n=cache.get_seq_length();assert n==ctx['stance_cache_tokens'] and source['prefix_tokens']==ctx['prefix_tokens']
    ids,_=j.branch_ids(ctx['msgs'],question,ctx['history'],head_text=ctx['head']); q=len(ids)
    mask=torch.arange(n+q,device=j.device)[None,:]<=torch.arange(n,n+q,device=j.device)[:,None]
    mask[:,source['dropped_visual_keys']]=False
    j.model.model.rope_deltas=ctx['rope'].clone()
    # A four-dimensional causal mask cannot supply Qwen's two-dimensional
    # token positions. Keep the original cached logical text continuation.
    positions=(torch.arange(n,n+q,device=j.device)+int(ctx['rope'][0,0])).view(1,1,q).expand(3,1,q)
    try:
        out=j.model.model(input_ids=torch.tensor([ids],device=j.device),past_key_values=cache,
            attention_mask=mask[None,None],position_ids=positions,use_cache=True)
        hidden=out.last_hidden_state[0,-1].clone();del out
        return j.margins_fp32(hidden[None])[0]
    finally:
        cache.crop(n);j.model.model.rope_deltas=ctx['rope'].clone()
