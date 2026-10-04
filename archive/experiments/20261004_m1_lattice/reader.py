"""Single structural suffix forward on an unchanged native stance cache."""
import torch
from lattice import tokenize_graph,graph_bias,graph_positions,SCAFFOLD,TAIL


def compile_branch(j,ctx,w,nwindows,arm='full'):
    graph=tokenize_graph(w['confusion'],j.tok)
    scaffold=SCAFFOLD.format(i=w['i']+1,n=nwindows,a=w['start'],b=w['end'])
    marker='WCNQZ'
    assert marker not in scaffold+TAIL
    full=j.render(ctx['msgs']+ctx['history']+[j.turn('user',scaffold+marker+TAIL)],True)
    assert full.startswith(ctx['head'])
    head,tail=full[len(ctx['head']):].split(marker)
    h=j.tok.encode(head,add_special_tokens=False);t=j.tok.encode(tail,add_special_tokens=False)
    return h+graph['ids']+t,graph,len(h),len(t),dict(scaffold=scaffold,
        head_text=head,tail_text=tail,head_tokens=h,tail_tokens=t,graph=graph,arm=arm)


@torch.no_grad()
def structural(j,cache,ctx,w,nwindows,verify=False):
    ids,graph,h,t,trace=compile_branch(j,ctx,w,nwindows)
    n=cache.get_seq_length();start=n+int(ctx['rope'][0,0])
    assert start==int(ctx['positions'].max())+1
    logical=graph_positions(graph,h,t,start)
    bias=graph_bias(graph,h,t,n)
    kwargs=dict(input_ids=torch.tensor([ids],device=j.device),past_key_values=cache,use_cache=True,
        position_ids=torch.tensor(logical,device=j.device)[None,None,:].expand(3,1,-1),
        # CUDA SDPA requires additive bias to match the query/embedding dtype.
        attention_mask=torch.as_tensor(bias,device=j.device,dtype=j.model.dtype)[None,None,:,:])
    j.model.model.rope_deltas=torch.tensor([[logical[-1]+1-(n+len(ids))]],device=j.device)
    try:
        out=j.model.model(**kwargs);hidden=out.last_hidden_state[0,-1];del out
        z=j.margins_fp32(hidden[None])[0]
    finally:cache.crop(n);j.model.model.rope_deltas=ctx['rope'].clone()
    trace.update(physical_tokens=len(ids),logical_positions=logical,margin=z,
        prefix_tokens=n,prefix_logical_start=start,
        graph_tokens=len(graph['ids']),slot_count=len(graph['slots']),
        prefix_positions_unchanged=True,mask_forward=True,clone_checked=False)
    if verify:
        import copy
        clone=copy.deepcopy(cache)
        try:
            replay,replay_trace=structural(j,clone,ctx,w,nwindows,False)
            assert replay==z,'independent cloned-cache structural replay differs'
        finally:del clone
        trace['clone_checked']=True
    return z,trace
