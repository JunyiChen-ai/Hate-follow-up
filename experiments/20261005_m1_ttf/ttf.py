"""Paper TTF selection and explicit source-bound Qwen3 multi-image prefill."""
import copy
import json
from contextlib import contextmanager
from pathlib import Path
import sys
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.stance_cache import positions
from src.mllm_judge import VIDEO_QUESTION

SPEC=json.loads((Path(__file__).parent/'spec.json').read_text())


def select(features,grids,merge):
    """CPU fp32 deterministic matching; feature source indices never renumbered."""
    assert features.device.type=='cpu' and features.ndim==2
    gg=grids.tolist();assert len(gg)>=2 and all(g==gg[0] for g in gg) and gg[0][0]==1
    F=len(gg);H,W=gg[0][1]//merge,gg[0][2]//merge;N=H*W
    assert features.shape[0]==F*N and torch.isfinite(features.float()).all()
    x=features.float().reshape(F,N,-1)
    mean=x.mean(1);global_mean=x.mean((0,1))
    cos=torch.nn.functional.cosine_similarity(mean,global_mean[None],dim=1,eps=SPEC['norm_epsilon'])
    anchor=int(cos.argmax());unit=torch.nn.functional.normalize(x,dim=-1,eps=SPEC['norm_epsilon'])
    yy=torch.arange(N)//W;xx=torch.arange(N)%W
    neighbors=[]
    for dy in range(-SPEC['radius'],SPEC['radius']+1):
        for dx in range(-SPEC['radius'],SPEC['radius']+1):
            neighbors.append((yy+dy).clamp(0,H-1)*W+(xx+dx).clamp(0,W-1))
    neighbors=torch.stack(neighbors,1)
    duplicate=torch.zeros_like(neighbors,dtype=torch.bool)
    for k in range(neighbors.shape[1]):
        if k:duplicate[:,k]=(neighbors[:,:k]==neighbors[:,k,None]).any(1)
    sims=(unit[:,:,None,:]*unit[anchor][neighbors][None]).sum(-1)
    sims.masked_fill_(duplicate[None],-torch.inf)
    best,offset=sims.max(-1);destination=neighbors[None].expand(F,-1,-1).gather(2,offset[:,:,None]).squeeze(2)
    keep=best<=SPEC['threshold'];keep[anchor]=True
    order=[anchor]+[f for f in range(F) if f!=anchor]
    retained=[f*N+i for f in order for i in range(N) if keep[f,i]]
    replacement=[f*N+i if keep[f,i] else anchor*N+int(destination[f,i]) for f in range(F) for i in range(N)]
    return dict(anchor=anchor,order=order,grid=[H,W],tokens_per_frame=N,
        frame_cosines=cos.tolist(),best_cosines=best.tolist(),destinations=destination.tolist(),
        keep=keep.tolist(),retained=retained,replacement=replacement,
        original_count=F*N,retained_count=len(retained),counts=[int(k.sum()) for k in keep])


@contextmanager
def capture_native(j):
    """Observe native tensors without modifying inputs, outputs, weights or cache."""
    model=j.model.model;original_features=model.get_image_features;original_prefix=j.prefix_cache;captured={}
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
    model.get_image_features=features;j.prefix_cache=prefix
    try:yield captured
    finally:model.get_image_features=original_features;j.prefix_cache=original_prefix


def layout(j,frames,segments,captured,plan,native_order=False):
    """Dense source coordinates -> sparse rows; all text/image boundaries retained."""
    original=captured['encoded'];F=len(frames);N=plan['tokens_per_frame']
    order=list(range(F)) if native_order else plan['order']
    assert sorted(order)==list(range(F))
    msgs,files=j.prefix_messages([frames[f] for f in order],segments)
    text,encoded=j.encode_prefix(msgs,files)
    assert encoded['image_grid_thw'].tolist()==original['image_grid_thw'][order].tolist()
    old_ids=original['input_ids'].to(j.device);ids=encoded['input_ids'].to(j.device)
    old_positions,_=positions(j,old_ids,original['image_grid_thw'].to(j.device))
    pos,_=positions(j,ids,encoded['image_grid_thw'].to(j.device))
    old_visual=(old_ids[0]==j.image_token_id).nonzero().flatten()
    visual=(ids[0]==j.image_token_id).nonzero().flatten()
    assert old_visual.numel()==visual.numel()==F*N
    source_order=torch.tensor([f*N+i for f in order for i in range(N)],device=j.device)
    pos[:,:,visual]=old_positions[:,:,old_visual[source_order]]
    keep=torch.tensor([plan['keep'][f][i] for f in order for i in range(N)],device=j.device)
    sequence_keep=torch.ones(ids.shape[1],dtype=torch.bool,device=j.device);sequence_keep[visual]=keep
    sequence_indices=sequence_keep.nonzero().flatten()
    source_indices=source_order[keep]
    assert source_indices.tolist()==([f*N+i for f in order for i in range(N) if plan['keep'][f][i]])
    embeds=j.model.model.get_input_embeddings()(ids)
    embeds[:,visual]=captured['features'][source_order].to(embeds.dtype)
    packed=dict(inputs_embeds=embeds[:,sequence_indices],position_ids=pos[:,:,sequence_indices],
        attention_mask=torch.ones((1,len(sequence_indices)),dtype=ids.dtype,device=j.device),
        visual_pos_masks=(ids[:,sequence_indices]==j.image_token_id),
        deepstack_visual_embeds=[v[source_indices].to(embeds.device,embeds.dtype) for v in captured['deepstack']])
    assert int(packed['visual_pos_masks'].sum())==len(source_indices)
    for v in packed['deepstack_visual_embeds']:assert len(v)==len(source_indices)
    evidence=dict(original_prefix_ids=old_ids[0].tolist(),ordered_prefix_ids=ids[0].tolist(),
        ordered_full_positions=pos[:,0].tolist(),sequence_indices=sequence_indices.tolist(),
        packed_ids=ids[0,sequence_indices].tolist(),packed_positions=packed['position_ids'][:,0].tolist(),
        source_indices=source_indices.tolist(),grid_thw=encoded['image_grid_thw'].tolist(),
        compressed_prefix_tokens=len(sequence_indices),uncompressed_prefix_tokens=ids.shape[1],
        visual_source_positions_exact=torch.equal(packed['position_ids'][:,:,packed['visual_pos_masks'][0]],old_positions[:,:,old_visual[source_indices]]),
        text_and_boundaries_preserved=bool(sequence_keep[ids[0]!=j.image_token_id].all()),
        deepstack_gather_exact=all(torch.equal(v,src[source_indices]) for v,src in zip(packed['deepstack_visual_embeds'],captured['deepstack'])))
    assert evidence['visual_source_positions_exact'] and evidence['text_and_boundaries_preserved'] and evidence['deepstack_gather_exact']
    return msgs,text,packed,evidence


@torch.no_grad()
def prefill(j,msgs,text,packed,native_ctx):
    out=j.model.model.language_model(**packed,use_cache=True)
    cache=out.past_key_values;del out
    L=cache.get_seq_length();assert L==packed['inputs_embeds'].shape[1]
    logical=int(packed['position_ids'].max())+1
    delta=torch.tensor([[logical-L]],dtype=torch.long,device=j.device)
    j.model.model.rope_deltas=delta.clone()
    qid,qtext=j.branch_ids(msgs,VIDEO_QUESTION,head_text=text)
    j.extend_cache(cache,qid)
    aid,atext=j.answer_ids(msgs,VIDEO_QUESTION,native_ctx['stance'])
    j.extend_cache(cache,aid)
    assert cache.get_seq_length()==L+len(qid)+len(aid) and torch.equal(j.model.model.rope_deltas,delta)
    history=[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',native_ctx['stance'])]
    ctx=dict(msgs=msgs,history=history,head=text+qtext+atext,rope=delta,
        global_margin=native_ctx['global_margin'],stance=native_ctx['stance'],prefix_tokens=L,
        stance_cache_tokens=cache.get_seq_length(),stance_cache_logical_start=logical+len(qid)+len(aid))
    return cache,ctx


def dense_plan(plan):
    d=copy.deepcopy(plan);F=len(d['order']);N=d['tokens_per_frame']
    d['keep']=[[True]*N for _ in range(F)];d['counts']=[N]*F
    d['retained_count']=d['original_count'];d['retained']=[f*N+i for f in d['order'] for i in range(N)]
    d['replacement']=list(range(F*N))
    return d
