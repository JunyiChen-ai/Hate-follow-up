"""Independent Qwen3 cached branch with new actual source images and native stance."""
from PIL import Image
import torch
from src.stance_cache import positions


def encode_branch(j,ctx,question,source_content,image_paths):
    assert j.family=='qwen3_vl' and image_paths
    assert sum(c['type']=='image' for c in source_content)==len(image_paths)
    message=dict(role='user',content=[*source_content,dict(type='text',text=question)])
    full=j.render(ctx['msgs']+ctx['history']+[message],True)
    assert full.startswith(ctx['head']),'source image branch changed the cached conversation'
    suffix=full[len(ctx['head']):]
    images=[Image.open(path).convert('RGB') for path in image_paths]
    try:encoded=j.encode(suffix,images)
    finally:
        for image in images:image.close()
    ids=encoded['input_ids'].to(j.device);grids=encoded['image_grid_thw'].to(j.device)
    assert ids.shape[0]==1 and len(grids)==len(image_paths)
    merge=j.processor.image_processor.merge_size
    counts=[int(t*h*w)//merge**2 for t,h,w in grids.tolist()]
    assert sum(counts)==int((ids==j.image_token_id).sum())
    p,_=positions(j,ids,grids)
    p=p+ctx['stance_cache_logical_start']
    return encoded,p,dict(suffix_text=suffix,suffix_ids=ids[0].tolist(),image_grid=grids.tolist(),
        image_counts=counts,positions=p[:,0].tolist(),logical_start=ctx['stance_cache_logical_start'])


@torch.no_grad()
def margin(j,cache,ctx,question,source_content,image_paths):
    n=cache.get_seq_length();assert n==ctx['stance_cache_tokens']
    encoded,p,evidence=encode_branch(j,ctx,question,source_content,image_paths)
    kw=j.model_inputs(encoded)
    # The processor's mask describes only this suffix. Use native cache-aware
    # causality; image/text positions are explicitly bound to the old prefix.
    kw.pop('attention_mask',None)
    j.model.model.rope_deltas=ctx['rope'].clone()
    try:
        out=j.model.model(**kw,position_ids=p,past_key_values=cache,use_cache=True)
        hidden=out.last_hidden_state[0,-1].clone();del out
        assert cache.get_seq_length()==n+len(evidence['suffix_ids'])
        z=j.margins_fp32(hidden[None])[0]
        return z,evidence
    finally:
        cache.crop(n);j.model.model.rope_deltas=ctx['rope'].clone()
