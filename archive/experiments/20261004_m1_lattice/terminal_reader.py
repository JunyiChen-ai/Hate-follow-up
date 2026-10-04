"""R3 isolated path encoder and posterior-weighted end-state query."""
import numpy as np
import torch
from lattice import TAIL
from path_graph import SCAFFOLD
from terminal_graph import VERSION,tokenize_paths,bias,positions


def compile_branch(j,ctx,w,nwindows):
    graph=tokenize_paths(w['beams'],j.tok);scaffold=SCAFFOLD.format(i=w['i']+1,n=nwindows,a=w['start'],b=w['end'])
    marker='TERMINALLATTICESLOT';assert marker not in scaffold+TAIL
    rendered=j.render(ctx['msgs']+ctx['history']+[j.turn('user',scaffold+marker+TAIL)],True)
    assert rendered.startswith(ctx['head']);head,tail=rendered[len(ctx['head']):].split(marker)
    h=j.tok.encode(head,add_special_tokens=False);t=j.tok.encode(tail,add_special_tokens=False)
    return h+graph['ids']+t,dict(version=VERSION,scaffold=scaffold,head_text=head,tail_text=tail,
        head_tokens=h,tail_tokens=t,graph=graph,physical_tokens=len(h)+len(graph['ids'])+len(t))


@torch.no_grad()
def _forward(j,cache,ctx,ids,trace,mask,logical):
    n=cache.get_seq_length();start=n+int(ctx['rope'][0,0]);assert start==int(ctx['positions'].max())+1
    j.model.model.rope_deltas=torch.tensor([[logical[-1]+1-(n+len(ids))]],device=j.device)
    try:
        out=j.model.model(input_ids=torch.tensor([ids],device=j.device),past_key_values=cache,use_cache=True,
            position_ids=torch.tensor(logical,device=j.device)[None,None,:].expand(3,1,-1),
            attention_mask=torch.as_tensor(mask,device=j.device,dtype=j.model.dtype)[None,None,:,:])
        hidden=out.last_hidden_state[0,-1];del out;z=j.margins_fp32(hidden[None])[0]
    finally:cache.crop(n);j.model.model.rope_deltas=ctx['rope'].clone()
    trace.update(ids=ids,logical_positions=logical,margin=z,prefix_tokens=n,prefix_logical_start=start,
        attention_dtype=str(j.model.dtype),mask_forward=True,terminal_key_mask_forward=True)
    return z,trace


@torch.no_grad()
def structural(j,cache,ctx,w,nwindows):
    ids,t=compile_branch(j,ctx,w,nwindows);g=t['graph'];h=len(t['head_tokens']);tail=len(t['tail_tokens'])
    n=cache.get_seq_length();start=n+int(ctx['rope'][0,0])
    return _forward(j,cache,ctx,ids,t,bias(g,h,tail,n),positions(g,h,tail,start))


@torch.no_grad()
def unit_reference(j,cache,ctx,w,nwindows):
    """Independent one-path matrix: same end-state contract, not ordinary causal."""
    ids,t=compile_branch(j,ctx,w,nwindows);graph=t['graph'];h=len(t['head_tokens']);g=len(graph['ids'])
    assert len(graph['paths'])==1 and np.isclose(graph['paths'][0]['mass'],1.,atol=1e-6)
    assert graph['terminal_indices']==([g-1] if g else [])
    n=cache.get_seq_length();length=len(ids);mask=np.full((length,n+length),-np.inf,dtype=np.float32)
    for q in range(length):
        mask[q,:n]=0.
        if q<h+g:mask[q,n:n+q+1]=0.
        else:
            mask[q,n:n+h]=0.
            if g:mask[q,n+h+g-1]=0.
            mask[q,n+h+g:n+q+1]=0.
    start=n+int(ctx['rope'][0,0]);logical=list(range(start,start+length))
    return _forward(j,cache,ctx,ids,t,mask,logical)
