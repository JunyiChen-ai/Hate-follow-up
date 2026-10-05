"""Explicit source-bound sparse Qwen3 multi-image prefixes, without fake grids."""
from contextlib import contextmanager
import torch
from src.stance_cache import positions
from src.mllm_judge import VIDEO_QUESTION


@contextmanager
def capture_native(j):
    model=j.model.model
    original_features=model.get_image_features; original_prefix=j.prefix_cache; captured={}
    def features(*args,**kwargs):
        out=original_features(*args,**kwargs)
        assert 'features' not in captured
        captured['features']=torch.cat(tuple(out.pooler_output),0).detach()
        captured['deepstack']=[v.detach() for v in out.deepstack_features]
        return out
    def prefix(enc):
        assert 'encoded' not in captured
        captured['encoded']=enc
        return original_prefix(enc)
    model.get_image_features=features; j.prefix_cache=prefix
    try: yield captured
    finally: model.get_image_features=original_features; j.prefix_cache=original_prefix


def aggregate(features,members):
    """The same complete original source components apply to every visual layer."""
    assert features.ndim==2 and members and all(m for m in members)
    return torch.stack([features[m].float().mean(0).to(features.dtype) for m in members])


def pack(j,captured,roots,visual_features,deepstack_features):
    """Chronological roots with their unchanged native three-axis positions."""
    original=captured['encoded']; ids=original['input_ids'].to(j.device)
    grids=original['image_grid_thw'].to(j.device)
    pos,_=positions(j,ids,grids)
    visual=(ids[0]==j.image_token_id).nonzero().flatten()
    assert roots==sorted(set(roots)) and roots and 0<=roots[0]<=roots[-1]<len(visual)
    root_tensor=torch.tensor(roots,device=j.device)
    assert len(visual_features)==len(roots) and len(deepstack_features)==len(captured['deepstack'])
    sequence_keep=ids[0]!=j.image_token_id
    sequence_keep[visual[root_tensor]]=True
    sequence_indices=sequence_keep.nonzero().flatten()
    packed_ids=ids[:,sequence_indices]
    embeds=j.model.model.get_input_embeddings()(packed_ids)
    mask=packed_ids==j.image_token_id
    assert int(mask.sum())==len(roots)
    embeds[mask]=visual_features.to(embeds.device,embeds.dtype)
    packed=dict(inputs_embeds=embeds,position_ids=pos[:,:,sequence_indices],
        attention_mask=torch.ones_like(packed_ids),visual_pos_masks=mask,
        deepstack_visual_embeds=[v.to(embeds.device,embeds.dtype) for v in deepstack_features])
    assert all(len(v)==len(roots) for v in packed['deepstack_visual_embeds'])
    evidence=dict(original_prefix_ids=ids[0].tolist(),sequence_indices=sequence_indices.tolist(),
        packed_ids=packed_ids[0].tolist(),packed_positions=packed['position_ids'][:,0].tolist(),
        source_indices=roots,grid_thw=grids.tolist(),compressed_prefix_tokens=len(sequence_indices),
        uncompressed_prefix_tokens=ids.shape[1],
        visual_source_positions_exact=torch.equal(packed['position_ids'][:,:,mask[0]],pos[:,:,visual[root_tensor]]),
        text_and_boundaries_preserved=bool(sequence_keep[ids[0]!=j.image_token_id].all()))
    assert evidence['visual_source_positions_exact'] and evidence['text_and_boundaries_preserved']
    return packed,evidence


@torch.no_grad()
def prefill(j,msgs,text,packed,native_ctx):
    out=j.model.model.language_model(**packed,use_cache=True)
    cache=out.past_key_values; del out
    length=cache.get_seq_length(); assert length==packed['inputs_embeds'].shape[1]
    logical=int(packed['position_ids'].max())+1
    delta=torch.tensor([[logical-length]],dtype=torch.long,device=j.device)
    j.model.model.rope_deltas=delta.clone()
    qid,qtext=j.branch_ids(msgs,VIDEO_QUESTION,head_text=text); j.extend_cache(cache,qid)
    aid,atext=j.answer_ids(msgs,VIDEO_QUESTION,native_ctx['stance']); j.extend_cache(cache,aid)
    assert cache.get_seq_length()==length+len(qid)+len(aid) and torch.equal(j.model.model.rope_deltas,delta)
    history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',native_ctx['stance'])]
    return cache,dict(msgs=msgs,history=history,head=text+qtext+atext,rope=delta,
        global_margin=native_ctx['global_margin'],stance=native_ctx['stance'],prefix_tokens=length,
        stance_cache_tokens=cache.get_seq_length(),stance_cache_logical_start=logical+len(qid)+len(aid))
