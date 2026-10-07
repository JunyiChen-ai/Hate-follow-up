"""One fresh uncompressed LOCAL suffix with source-bound quantized context."""
from PIL import Image
import torch
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb,repeat_kv
from inputs import ROOT,SPEC
from src.stance_cache import positions
from src.pre_rotary_memory import translate,rotate,mask,attention_scope,mean_query,select
from src.expanded_image_offsets import expand


def encode_local_input(j,ctx,frames,local,question):
    content=[dict(type='text',text=SPEC['reader_role_text'])]
    for i in local:
        content+=[dict(type='text',text=f"Actual LOCAL source at {frames[i]['time']:.3f} seconds."),dict(type='image',image=frames[i]['path'])]
    content.append(dict(type='text',text=question));message=dict(role='user',content=content)
    full=j.render(ctx['msgs']+ctx['history']+[message],True);assert full.startswith(ctx['head']);text=full[len(ctx['head']):]
    images=[]
    try:
        for i in local:
            with Image.open(ROOT/frames[i]['path']) as original:images.append(original.convert('RGB'))
        encoded=j.encode(text,images)
    finally:
        for image in images:image.close()
    ids=encoded['input_ids'].to(j.device);grid=encoded['image_grid_thw'].to(j.device)
    assert len(grid)==len(local);relative,_=positions(j,ids,grid)
    counts=[int(torch.tensor(g).prod())//j.processor.image_processor.merge_size**2 for g in grid.tolist()]
    assert int((ids==j.image_token_id).sum())==sum(counts)
    tok=j.tok(text,add_special_tokens=False,return_offsets_mapping=True)
    offsets=expand(tok['input_ids'],tok['offset_mapping'],j.image_token_id,counts,ids[0].tolist())
    # The final literal question occurrence is selected even if source role text
    # happens to contain a repeated substring. Role/chat/image rows are excluded.
    begin=text.rfind(question);assert begin>=0;end=begin+len(question)
    rows=[i for i,(a,b) in enumerate(offsets) if a<end and b>begin]
    assert rows and all(ids[0,i].item() not in j.tok.all_special_ids for i in rows)
    evidence=dict(message=message,suffix_text=text,suffix_ids=ids[0].tolist(),grid=grid.tolist(),positions=relative[:,0].tolist(),
        LOCAL_ids=list(local),image_counts=counts,question_start=begin,question_end=end,question_rows=rows,
        raw_token_ids=tok['input_ids'],raw_offsets=[list(x) for x in tok['offset_mapping']])
    return encoded,relative,rows,evidence


def encode_local(j,ctx,memory,frames,local,question):
    encoded,relative,rows,evidence=encode_local_input(j,ctx,frames,local,question)
    ids=encoded['input_ids'].to(j.device);visual=ids==j.image_token_id;features=[];deep=[]
    for i,count in zip(local,evidence['image_counts']):
        f,ds=memory.local_features(i);assert count==len(f) and all(v.shape==f.shape for v in ds)
        features.append(f);deep.append(ds)
    embedding=j.model.model.get_input_embeddings()(ids);embedding[visual]=torch.cat(features).to(j.device,j.dtype)
    layers=len(deep[0]);assert all(len(v)==layers for v in deep)
    injected=[torch.cat([v[l] for v in deep]).to(j.device,j.dtype) for l in range(layers)]
    return dict(inputs_embeds=embedding,visual_pos_masks=visual,deepstack_visual_embeds=injected),relative,rows,evidence


def factory(j,cache,ctx,memory,local,relative,question_rows,traces,pick=None):
    representatives=memory.representatives();indices=[b['source_index'] for b in memory.blocks]
    rotary=j.model.model.language_model.rotary_emb
    def make(layer):
        def attention(att,hidden_states,position_embeddings,attention_mask,past_key_values=None,**kwargs):
            assert past_key_values is cache and hidden_states.shape[0]==1
            shape=(*hidden_states.shape[:-1],-1,att.head_dim)
            q=att.q_norm(att.q_proj(hidden_states).view(shape)).transpose(1,2)
            k=att.k_norm(att.k_proj(hidden_states).view(shape)).transpose(1,2);v=att.v_proj(hidden_states).view(shape).transpose(1,2)
            pooled=mean_query(q,question_rows,k.shape[1]).detach().cpu()
            # pick (mechanism controls only) replaces per-layer question matching; None is R1.
            if pick is None:remote,scores=select(pooled,representatives[layer],indices,set(local),SPEC['remote_frames_per_layer'])
            else:remote=list(pick(layer))
            ordered=sorted(remote,key=lambda i:indices[i]);packed,end=translate([memory.blocks[i]['positions'] for i in ordered],ctx['stance_cache_logical_start'])
            context_k=[];context_v=[]
            for i,p in zip(ordered,packed):
                sk,sv,_=memory.layer(i,layer,hidden_states.device);context_k.append(rotate(sk,p,rotary));context_v.append(sv)
            current=relative+end;cos,sin=rotary(q,current);q,k=apply_rotary_pos_emb(q,k,cos,sin)
            prefix=cache.layers[layer];assert prefix.keys.shape[2]==ctx['stance_cache_tokens']
            allk=torch.cat([prefix.keys,*context_k,k],2);allv=torch.cat([prefix.values,*context_v,v],2)
            out=scaled_dot_product_attention(q,repeat_kv(allk,att.num_key_value_groups),repeat_kv(allv,att.num_key_value_groups),
                attn_mask=mask(hidden_states.shape[1],allk.shape[2]-hidden_states.shape[1],hidden_states.device),
                dropout_p=0.,is_causal=False,scale=att.scaling)
            assert layer not in traces
            traces[layer]=dict(remote_ids=remote,packed_remote_ids=ordered,query_vector=pooled,source_tokens=sum(memory.blocks[i]['shape'][-2] for i in ordered),
                source_translations=[int(a.min())-int(b.min()) for a,b in zip(packed,[memory.blocks[i]['positions'] for i in ordered])],
                suffix_logical_start=end,source_times=[memory.blocks[i]['actual_time'] for i in ordered],prefix_tokens=ctx['stance_cache_tokens'],suffix_tokens=hidden_states.shape[1])
            out=out.transpose(1,2).reshape(*hidden_states.shape[:-1],-1).contiguous()
            return att.o_proj(out),None
        return attention
    return make


@torch.no_grad()
def visual_margin(j,cache,ctx,memory,frames,local,question,pick=None):
    assert local and cache.get_seq_length()==ctx['stance_cache_tokens']
    kwargs,relative,rows,evidence=encode_local(j,ctx,memory,frames,local,question);trace={}
    old=j.model.model.rope_deltas.clone();first=j.forward_calls;vision=j.vision_calls
    try:
        with attention_scope(j,factory(j,cache,ctx,memory,local,relative,rows,trace,pick)):
            out=j.model.model.language_model(**kwargs,position_ids=relative+ctx['stance_cache_logical_start'],past_key_values=cache,use_cache=True)
            hidden=out.last_hidden_state[0,-1].clone();del out
        assert len(trace)==len(cache.layers) and cache.get_seq_length()==ctx['stance_cache_tokens']
        assert j.forward_calls-first==1 and j.vision_calls==vision
        return j.margins_fp32(hidden[None])[0],dict(input=evidence,layers=trace)
    finally:j.model.model.rope_deltas=old
