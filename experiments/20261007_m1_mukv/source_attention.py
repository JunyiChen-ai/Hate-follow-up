"""Complete earlier-window same-grain context loaded one language layer at a time."""
import torch
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb,repeat_kv
from inputs import SPEC
from src.pre_rotary_memory import translate,rotate,mask


def layout(ctx,memory,history,relative):
    packed,end=translate([memory.blocks[i]['positions'] for i in history],ctx['stance_cache_logical_start'])
    return packed,relative+end,end


def factory(j,native,ctx,memory,history,relative,visual_rows,captured):
    past_positions,current,end=layout(ctx,memory,history,relative);rotary=j.model.model.language_model.rotary_emb
    def make(layer):
        def attention(att,hidden_states,position_embeddings,attention_mask,past_key_values=None,**kwargs):
            assert past_key_values is native and hidden_states.shape[0]==1
            shape=(*hidden_states.shape[:-1],-1,att.head_dim)
            q=att.q_norm(att.q_proj(hidden_states).view(shape)).transpose(1,2)
            k=att.k_norm(att.k_proj(hidden_states).view(shape)).transpose(1,2);v=att.v_proj(hidden_states).view(shape).transpose(1,2)
            assert layer not in captured['keys'];captured['keys'][layer]=k.detach().clone();captured['values'][layer]=v.detach().clone()
            cos,sin=rotary(q,current);qr,kr=apply_rotary_pos_emb(q,k,cos,sin)
            prefix=native.layers[layer];kk=[prefix.keys];vv=[prefix.values]
            for i,p in zip(history,past_positions):
                hk,hv,_=memory.layer(i,layer,hidden_states.device);kk.append(rotate(hk,p,rotary));vv.append(hv)
            kk.append(kr);vv.append(v);allk=torch.cat(kk,2);allv=torch.cat(vv,2);past=allk.shape[2]-hidden_states.shape[1]
            expandedk=repeat_kv(allk,att.num_key_value_groups);expandedv=repeat_kv(allv,att.num_key_value_groups)
            out=scaled_dot_product_attention(qr,expandedk,expandedv,attn_mask=mask(hidden_states.shape[1],past,hidden_states.device),
                dropout_p=0.,is_causal=False,scale=att.scaling)
            if layer==len(native.layers)-1:
                # This additional observation is paid work, not a free attention
                # output. Normalization includes full history/native/current.
                total=torch.zeros(len(visual_rows),device=hidden_states.device,dtype=torch.float32)
                columns=torch.tensor(visual_rows,device=hidden_states.device)+past
                for offset in range(0,len(visual_rows),SPEC['source_attention_query_chunk']):
                    rows=torch.tensor(visual_rows[offset:offset+SPEC['source_attention_query_chunk']],device=hidden_states.device)
                    score=(qr[:,:,rows,:].float()@expandedk.float().transpose(-1,-2))*att.scaling
                    causal=torch.arange(allk.shape[2],device=rows.device)[None,:]<=(past+rows[:,None])
                    weights=torch.softmax(score.masked_fill(~causal[None,None],-torch.inf),-1)
                    total+=weights[:,:,:,columns].sum((0,1,2))/(qr.shape[1]*len(visual_rows))
                captured['attention']=total.detach().cpu()
                captured['last_rotated_keys']=kr[0,:,visual_rows,:].detach().cpu()
            out=out.transpose(1,2).reshape(*hidden_states.shape[:-1],-1).contiguous()
            return att.o_proj(out),None
        return attention
    return make,current,end
