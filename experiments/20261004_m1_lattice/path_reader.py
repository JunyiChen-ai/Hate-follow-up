"""One complete-path lattice suffix on the unchanged native stance cache."""
import torch
from lattice import TAIL
from path_graph import VERSION,SCAFFOLD,tokenize_paths,bias,positions


def compile_branch(j,ctx,w,nwindows):
    graph=tokenize_paths(w['beams'],j.tok);scaffold=SCAFFOLD.format(i=w['i']+1,n=nwindows,a=w['start'],b=w['end'])
    marker='FULLPATHLATTICESLOT';assert marker not in scaffold+TAIL
    rendered=j.render(ctx['msgs']+ctx['history']+[j.turn('user',scaffold+marker+TAIL)],True)
    assert rendered.startswith(ctx['head']);head,tail=rendered[len(ctx['head']):].split(marker)
    h=j.tok.encode(head,add_special_tokens=False);t=j.tok.encode(tail,add_special_tokens=False)
    return h+graph['ids']+t,dict(version=VERSION,scaffold=scaffold,head_text=head,tail_text=tail,
        head_tokens=h,tail_tokens=t,graph=graph,physical_tokens=len(h)+len(graph['ids'])+len(t))


@torch.no_grad()
def structural(j,cache,ctx,w,nwindows):
    ids,trace=compile_branch(j,ctx,w,nwindows);g=trace['graph'];h=len(trace['head_tokens']);t=len(trace['tail_tokens'])
    n=cache.get_seq_length();start=n+int(ctx['rope'][0,0]);assert start==int(ctx['positions'].max())+1
    logical=positions(g,h,t,start);mask=bias(g,h,t,n)
    j.model.model.rope_deltas=torch.tensor([[logical[-1]+1-(n+len(ids))]],device=j.device)
    try:
        out=j.model.model(input_ids=torch.tensor([ids],device=j.device),past_key_values=cache,use_cache=True,
            position_ids=torch.tensor(logical,device=j.device)[None,None,:].expand(3,1,-1),
            attention_mask=torch.as_tensor(mask,device=j.device,dtype=j.model.dtype)[None,None,:,:])
        hidden=out.last_hidden_state[0,-1];del out;z=j.margins_fp32(hidden[None])[0]
    finally:cache.crop(n);j.model.model.rope_deltas=ctx['rope'].clone()
    trace.update(ids=ids,logical_positions=logical,margin=z,prefix_tokens=n,prefix_logical_start=start,
        attention_dtype=str(j.model.dtype),mask_forward=True)
    return z,trace
