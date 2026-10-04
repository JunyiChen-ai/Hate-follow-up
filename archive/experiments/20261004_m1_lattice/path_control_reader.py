"""Matched intact-path controls; no labels or earlier predictions as inputs."""
import copy
import numpy as np
import torch
from lattice import TAIL
from path_graph import tokenize_paths,bias,positions
from path_reader import compile_branch,structural
from control_reader import ARMS,donor_map

CONTROL_VERSION='R2 intact-path matched controls, sources2026-10-05'


def graph_positions(g,h,t,start,arm='full'):
    return list(range(start,start+h+len(g['ids'])+t)) if arm=='flat' else positions(g,h,t,start)


def compile_control(j,ctx,window,nwindows,arm,donor=None):
    assert arm in ARMS
    source=donor if arm=='wrong_audio_window' else window
    assert source is not None and source['available'] and window['available']
    ids,trace=compile_branch(j,ctx,source,nwindows);trace=copy.deepcopy(trace);g=copy.deepcopy(trace['graph'])
    head,tail=trace['head_tokens'],trace['tail_tokens']
    coverage=dict(source_binding_changed=False,mass_paths_changed=0,bf16_logmass_paths_changed=0,
        mass_paths_singleton=0,mass_paths_zero=0)
    if arm=='onebest':
        g=tokenize_paths([dict(text=window['confusion']['onebest'],score=0.) for _ in range(5)],j.tok)
    elif arm=='wrong_mass':
        pp=g['paths'];mm=[p['mass'] for p in pp]
        if len(pp)<=1:coverage['mass_paths_singleton']=len(pp)
        elif any(m==0 for m in mm):coverage['mass_paths_zero']=sum(m==0 for m in mm)
        else:
            shift=len(pp)//2;rotated=mm[shift:]+mm[:shift]
            coverage['mass_paths_changed']=sum(a!=b for a,b in zip(mm,rotated))
            a=torch.tensor(np.log(np.asarray(mm,dtype=np.float32)),dtype=torch.bfloat16)
            b=torch.tensor(np.log(np.asarray(rotated,dtype=np.float32)),dtype=torch.bfloat16)
            coverage['bf16_logmass_paths_changed']=int((a!=b).sum())
            for p,m in zip(pp,rotated):p['mass']=m
            for node in g['nodes']:node['mass']=pp[node['path']]['mass']
    elif arm=='wrong_audio_window' and donor['i']!=window['i']:
        coverage['source_binding_changed']=True
        instruction=(f"Judge destination window {window['i']+1} of {nwindows}, from {window['start']:.1f}s "
            f"to {window['end']:.1f}s of this video. The transcription above comes from source "
            f"window {donor['i']+1} of {nwindows}, from {donor['start']:.1f}s to {donor['end']:.1f}s.\n")
        assert trace['tail_text'].count(TAIL)==1
        trace['tail_text']=trace['tail_text'].replace(TAIL,'\n\n'+instruction+TAIL)
        tail=j.tok.encode(trace['tail_text'],add_special_tokens=False)
    ids=head+g['ids']+tail
    trace.update(arm=arm,control_version=CONTROL_VERSION,graph=g,head_tokens=head,tail_tokens=tail,
        physical_tokens=len(ids),destination_window=window['i'],source_window=source['i'],
        source_nominal_interval=[source['start'],source['end']],source_actual_interval=source['crop']['actual_interval'],
        coverage=coverage)
    return ids,g,len(head),len(tail),trace


@torch.no_grad()
def control_margin(j,cache,ctx,window,nwindows,arm,donor=None):
    ids,g,h,t,trace=compile_control(j,ctx,window,nwindows,arm,donor)
    n=cache.get_seq_length();start=n+int(ctx['rope'][0,0]);assert start==int(ctx['positions'].max())+1
    position_arm='flat' if arm in ('flat','onebest') else 'full'
    bias_arm=arm if arm in ('flat','binary') else 'full';logical=graph_positions(g,h,t,start,position_arm)
    if arm=='full':
        z,production=structural(j,cache,ctx,window,nwindows)
        assert all(production[k]==trace[k] for k in ('graph','head_tokens','tail_tokens','physical_tokens'))
    else:
        j.model.model.rope_deltas=ctx['rope'].clone()
        try:
            if arm=='onebest':z=j.cached_margin(cache,ids,in_place=True)
            else:
                mask=bias(g,h,t,n)
                if arm=='binary':mask[np.isfinite(mask)]=0.
                elif arm=='flat':mask[:,n:]=np.where(np.tri(len(ids),dtype=bool),0.,-np.inf)
                j.model.model.rope_deltas=torch.tensor([[logical[-1]+1-(n+len(ids))]],device=j.device)
                out=j.model.model(input_ids=torch.tensor([ids],device=j.device),past_key_values=cache,use_cache=True,
                    position_ids=torch.tensor(logical,device=j.device)[None,None,:].expand(3,1,-1),
                    attention_mask=torch.as_tensor(mask,device=j.device,dtype=j.model.dtype)[None,None,:,:])
                hidden=out.last_hidden_state[0,-1];del out;z=j.margins_fp32(hidden[None])[0]
        finally:cache.crop(n);j.model.model.rope_deltas=ctx['rope'].clone()
    assert cache.get_seq_length()==n and torch.equal(j.model.model.rope_deltas,ctx['rope'])
    trace.update(margin=z,physical_tokens=len(ids),logical_positions=logical,prefix_tokens=n,prefix_logical_start=start,
        graph_tokens=len(g['ids']),path_count=len(g['paths']),position_arm=position_arm,bias_arm=bias_arm,
        ordinary_serial=arm=='onebest',attention_dtype=str(j.model.dtype))
    return z,trace
