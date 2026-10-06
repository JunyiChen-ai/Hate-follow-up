"""One unscored local probe, coherent multigrain context and a fresh V margin."""
from PIL import Image
import torch
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb,repeat_kv
from inputs import ROOT,SPEC
from src.cached_image_features import reuse
from src.pre_rotary_memory import translate,rotate,mask,mean_query,attention_scope
from src.expanded_image_offsets import expand
from coherence import retrieve


def local_geometry(j,ctx,frames,question,role):
    content=[dict(type='text',text=role)]
    for f in frames:content+=[dict(type='text',text=f"Actual LOCAL frame at {f['time']:.3f} seconds."),dict(type='image',image=f['path'])]
    content.append(dict(type='text',text=question));message=dict(role='user',content=content)
    full=j.render(ctx['msgs']+ctx['history']+[message],True);assert full.startswith(ctx['head']);text=full[len(ctx['head']):];images=[]
    try:
        for f in frames:
            with Image.open(ROOT/f['path']) as image:images.append(image.convert('RGB'))
        enc=j.encode(text,images)
    finally:
        for image in images:image.close()
    from src.stance_cache import positions
    p,_=positions(j,enc['input_ids'].to(j.device),enc['image_grid_thw'].to(j.device))
    counts=[int(g.prod())//j.processor.image_processor.merge_size**2 for g in enc['image_grid_thw']]
    raw=j.tok(text,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'][0].tolist()
    offsets=expand(raw['input_ids'],raw['offset_mapping'],j.image_token_id,counts,ids);begin=text.rfind(question);assert begin>=0
    rows=[i for i,(a,b) in enumerate(offsets) if a<begin+len(question) and b>begin]
    assert rows and all(ids[i] not in j.tok.all_special_ids for i in rows)
    evidence=dict(message=message,suffix_text=text,suffix_ids=ids,grid=enc['image_grid_thw'].tolist(),positions=p[:,0].tolist(),question_rows=rows,
        raw_ids=raw['input_ids'],raw_offsets=[list(x) for x in raw['offset_mapping']])
    return enc,p,rows,evidence


def local_input(j,ctx,frames,cached,question,role):
    enc,p,rows,evidence=local_geometry(j,ctx,frames,question,role)
    features=[cached[f['index']]['feature'] for f in frames];deep=[cached[f['index']]['deepstack'] for f in frames]
    assert enc['image_grid_thw'].tolist()==[cached[f['index']]['grid'] for f in frames]
    packed,_=reuse(j,enc,features,deep)
    return packed,p,rows,evidence


@torch.no_grad()
def probe(j,native,ctx,memory,window,frames,cached,body):
    question=SPEC['neutral_query_text'].format(body=body);packed,relative,rows,evidence=local_input(j,ctx,frames,cached,question,'Unscored neutral retrieval query. Source observations retain actual times.')
    pooled={};last=j.model.model.language_model.layers[-1].self_attn
    def capture(module,args,out):pooled['Q']=mean_query(out.transpose(1,2),rows,last.config.num_key_value_heads).detach().cpu()
    hook=last.q_norm.register_forward_hook(capture);n=native.get_seq_length();saved=j.model.model.rope_deltas.clone();first=j.forward_calls;vision=j.vision_calls
    try:
        out=j.model.model.language_model(**packed,position_ids=relative+ctx['stance_cache_logical_start'],past_key_values=native,use_cache=True);del out
        assert j.forward_calls-first==1 and j.vision_calls==vision and pooled
    finally:
        hook.remove();native.crop(n);j.model.model.rope_deltas=saved
    reps={};order={};excluded={};mapping={}
    for grain in SPEC['granularities']:
        ids=[b['id'] for b in memory.blocks if b['grain']==grain];mapping[grain]=ids
        reps[grain]=torch.stack([memory.reps[i] for i in ids]) if ids else torch.empty(0,len(pooled['Q']))
        order[grain]=[(memory.blocks[i]['window'],-1 if memory.blocks[i]['quadrant'] is None else memory.blocks[i]['quadrant']) for i in ids]
        excluded[grain]={k for k,i in enumerate(ids) if memory.blocks[i]['window']==window}
    result=retrieve(pooled['Q'],reps,order,excluded)
    selected=[mapping[g][i] for g in SPEC['granularities'] for i in result['selected'][g]]
    selected.sort(key=lambda i:(memory.blocks[i]['window'],SPEC['granularities'].index(memory.blocks[i]['grain']),-1 if memory.blocks[i]['quadrant'] is None else memory.blocks[i]['quadrant']))
    return selected,dict(input=evidence,query=pooled['Q'],grain_to_block_ids=mapping,retrieval=result,selected_block_ids=selected,actual_LM=1,actual_vision=0)


def factory(j,native,ctx,memory,selected,relative):
    ordered,end=translate([memory.blocks[i]['positions'] for i in selected],ctx['stance_cache_logical_start']);rotary=j.model.model.language_model.rotary_emb
    def make(layer):
        def attention(att,hidden_states,position_embeddings,attention_mask,past_key_values=None,**kwargs):
            assert past_key_values is native
            shape=(*hidden_states.shape[:-1],-1,att.head_dim)
            q=att.q_norm(att.q_proj(hidden_states).view(shape)).transpose(1,2);k=att.k_norm(att.k_proj(hidden_states).view(shape)).transpose(1,2)
            v=att.v_proj(hidden_states).view(shape).transpose(1,2);cos,sin=rotary(q,relative+end);q,k=apply_rotary_pos_emb(q,k,cos,sin)
            kk=[native.layers[layer].keys];vv=[native.layers[layer].values]
            for i,p in zip(selected,ordered):
                sk,sv,_=memory.layer(i,layer,hidden_states.device);kk.append(rotate(sk,p,rotary));vv.append(sv)
            kk.append(k);vv.append(v);allk=torch.cat(kk,2);allv=torch.cat(vv,2)
            out=scaled_dot_product_attention(q,repeat_kv(allk,att.num_key_value_groups),repeat_kv(allv,att.num_key_value_groups),
                attn_mask=mask(hidden_states.shape[1],allk.shape[2]-hidden_states.shape[1],hidden_states.device),dropout_p=0.,is_causal=False,scale=att.scaling)
            return att.o_proj(out.transpose(1,2).reshape(*hidden_states.shape[:-1],-1).contiguous()),None
        return attention
    return make,end


@torch.no_grad()
def visual_margin(j,native,ctx,memory,frames,cached,selected,question):
    packed,relative,_,evidence=local_input(j,ctx,frames,cached,question,SPEC['reader_role_text']);make,end=factory(j,native,ctx,memory,selected,relative)
    saved=j.model.model.rope_deltas.clone();n=native.get_seq_length();first=j.forward_calls;vision=j.vision_calls
    try:
        with attention_scope(j,make):
            out=j.model.model.language_model(**packed,position_ids=relative+ctx['stance_cache_logical_start'],past_key_values=native,use_cache=True)
            hidden=out.last_hidden_state[0,-1].clone();del out
        assert native.get_seq_length()==n and j.forward_calls-first==1 and j.vision_calls==vision
        return j.margins_fp32(hidden[None])[0],dict(input=evidence,selected_blocks=selected,suffix_logical_start=end,
            source_tokens=sum(memory.blocks[i]['shape'][-2] for i in selected),actual_LM=1,actual_vision=0)
    finally:j.model.model.rope_deltas=saved
