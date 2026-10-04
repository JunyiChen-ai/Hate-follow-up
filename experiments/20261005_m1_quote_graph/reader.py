"""Single asymmetric source-context/local/query suffix forward."""
import numpy as np
import torch
from graph import compile_segments


def compile_branch(j,ctx,windows,packet):
    head,contexts,local,tail=compile_segments(windows,packet)
    m0,m1='QGCONTEXTSLOT','QGQUERYSLOT'
    assert all(m0 not in s and m1 not in s for s in [head,*contexts,local,tail])
    rendered=j.render(ctx['msgs']+ctx['history']+[j.turn('user',head+m0+local+m1+tail)],True)
    assert rendered.startswith(ctx['head'])
    h,l,t=rendered[len(ctx['head']):].replace(m0,m1,1).split(m1)
    assert l==local
    parts=[h,*contexts,l,t];ids=[];ranges=[]
    for text in parts:
        start=len(ids);ids.extend(j.tok.encode(text,add_special_tokens=False));ranges.append([start,len(ids)])
    return ids,dict(parts=parts,ranges=ranges,context_count=len(contexts),packet=packet,
        physical_tokens=len(ids),head_text=h,local_text=l,tail_text=t)


def attention_bias(ranges,prefix):
    length=ranges[-1][1];bias=np.full((length,prefix+length),-np.inf,dtype=np.float32)
    bias[:,:prefix]=0.;h0,h1=ranges[0];assert h0==0
    for q in range(h1):bias[q,prefix:prefix+q+1]=0.
    # Every separately encoded context record is independent of other records.
    for a,b in ranges[1:-2]:
        bias[a:b,prefix:prefix+h1]=0.
        for q in range(a,b):bias[q,prefix+a:prefix+q+1]=0.
    a,b=ranges[-2]
    for q in range(a,b):bias[q,prefix:prefix+q+1]=0.
    a,b=ranges[-1];local_start=ranges[-2][0]
    for q in range(a,b):
        bias[q,prefix:prefix+h1]=0.
        bias[q,prefix+local_start:prefix+q+1]=0.
    assert all(np.isfinite(bias[q,prefix+q]) for q in range(length))
    return bias


@torch.no_grad()
def structural(j,cache,ctx,windows,packet):
    ids,trace=compile_branch(j,ctx,windows,packet)
    n=cache.get_seq_length();start=n+int(ctx['rope'][0,0]);assert start==int(ctx['positions'].max())+1
    positions=list(range(start,start+len(ids)));bias=attention_bias(trace['ranges'],n)
    j.model.model.rope_deltas=ctx['rope'].clone()
    try:
        out=j.model.model(input_ids=torch.tensor([ids],device=j.device),past_key_values=cache,use_cache=True,
            position_ids=torch.tensor(positions,device=j.device)[None,None,:].expand(3,1,-1),
            attention_mask=torch.as_tensor(bias,device=j.device,dtype=j.model.dtype)[None,None,:,:])
        hidden=out.last_hidden_state[0,-1];del out;z=j.margins_fp32(hidden[None])[0]
    finally:cache.crop(n);j.model.model.rope_deltas=ctx['rope'].clone()
    trace.update(ids=ids,positions=positions,prefix_tokens=n,prefix_logical_start=start,
        margin=z,attention_dtype=str(j.model.dtype),mask_forward=True)
    return z,trace
