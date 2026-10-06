"""Explicit Qwen3 pre-rotary source observation/layout for supplementary memories.

Generic operators promoted from the reviewed ReKV implementation on2026-10-07.
Active ReKV retains its frozen local operators; this module carries no method
constants, dataset paths, scoring/evaluation logic or model revision digests.
"""
from contextlib import contextmanager
import types
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import rotate_half


def translate(blocks,start):
    cursor=int(start);assert cursor>=0;packed=[]
    for old in blocks:
        assert old.ndim==3 and old.shape[:2]==(3,1) and old.shape[2]>0 and old.dtype==torch.long
        shift=cursor-int(old.min());new=old+shift;packed.append(new);cursor=int(new.max())+1
    return packed,cursor


def rotate(key,position,rotary):
    assert key.ndim==4 and key.shape[0]==1 and position.shape==(3,1,key.shape[2])
    cos,sin=rotary(key,position.to(key.device))
    return key*cos.unsqueeze(1)+rotate_half(key)*sin.unsqueeze(1)


def mask(query_length,past_length,device):
    assert query_length>0 and past_length>=0
    rows=torch.arange(query_length,device=device)[:,None]
    columns=torch.arange(query_length+past_length,device=device)[None,:]
    return (columns<=past_length+rows)[None,None]


@contextmanager
def observe(j):
    keys,values,hooks={},{},[]
    try:
        for index,layer in enumerate(j.model.model.language_model.layers):
            def capture_key(module,args,out,index=index):
                assert index not in keys;keys[index]=out.transpose(1,2).detach().clone()
            def capture_value(module,args,out,index=index,att=layer.self_attn):
                assert index not in values
                values[index]=out.reshape(*out.shape[:-1],-1,att.head_dim).transpose(1,2).detach().clone()
            hooks.append(layer.self_attn.k_norm.register_forward_hook(capture_key))
            hooks.append(layer.self_attn.v_proj.register_forward_hook(capture_value))
        yield keys,values
        assert len(keys)==len(values)==len(j.model.model.language_model.layers)
    finally:
        for hook in hooks:hook.remove()


@contextmanager
def attention_scope(j,factory):
    saved=[]
    try:
        for layer in j.model.model.language_model.layers:
            att=layer.self_attn;saved.append((att,att.__dict__.get('forward')))
            att.forward=types.MethodType(factory(att.layer_idx),att)
        yield
    finally:
        for att,old in saved:
            if old is None:att.__dict__.pop('forward',None)
            else:att.forward=old


def mean_query(query,rows,kv_heads):
    assert query.shape[0]==1 and rows and query.shape[1]%kv_heads==0
    assert len(set(rows))==len(rows) and all(0<=r<query.shape[2] for r in rows)
    grouped=query.float().reshape(1,kv_heads,query.shape[1]//kv_heads,query.shape[2],query.shape[-1])
    return grouped[:,:,:,rows].mean(2).mean(2).reshape(-1)


def select(query,representatives,source_indices,excluded,count):
    assert query.device.type==representatives.device.type=='cpu'
    assert representatives.shape==(len(source_indices),len(query)) and torch.isfinite(representatives).all() and torch.isfinite(query).all()
    q=query.float();r=representatives.float();den=r.norm(dim=-1)*q.norm()
    safe=torch.where(den>0,den,torch.ones_like(den))
    scores=torch.where(den>0,(r@q)/safe,torch.zeros_like(den))
    assert count>=0 and all(a<b for a,b in zip(source_indices,source_indices[1:]))
    candidates=[i for i in range(len(source_indices)) if i not in excluded]
    chosen=sorted(candidates,key=lambda i:(-float(scores[i]),source_indices[i]))[:count]
    return chosen,scores
