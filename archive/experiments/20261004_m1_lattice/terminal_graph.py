"""R3: one conditional utterance end-state per complete acoustic path."""
import numpy as np
from path_graph import tokenize_paths as complete_paths,bias as complete_bias,positions

VERSION='R3 complete-hypothesis terminal-state lattice, sources2026-10-05'
TERMINAL='\n'


def tokenize_paths(beams,tokenizer):
    old=complete_paths(beams,tokenizer);terminal=tokenizer.encode(TERMINAL,add_special_tokens=False)
    assert len(terminal)==1,'declared literal newline terminal must be one Qwen token'
    paths=old['paths'];ids=[];nodes=[];width=0;ends=[]
    for pi,p in enumerate(paths):
        p['lexical_tokens']=p['tokens'][:]
        p['tokens']=p['lexical_tokens']+terminal if p['lexical_tokens'] else []
        width=max(width,len(p['tokens']))
        for k,token in enumerate(p['tokens']):
            end=k==len(p['tokens'])-1
            if end:ends.append(len(ids))
            nodes.append(dict(path=pi,subword=k,logical=k,mass=p['mass'],physical=len(ids),terminal=end));ids.append(token)
    return dict(ids=ids,nodes=nodes,paths=paths,path_length=width,beam_weights=old['beam_weights'],
        terminal_literal=TERMINAL,terminal_token_ids=terminal,terminal_indices=ends)


def bias(graph,h,t,n):
    out=complete_bias(graph,h,t,n);g=len(graph['ids'])
    if g:
        out[h+g:,n+h:n+h+g]=-np.inf
        for end in graph['terminal_indices']:
            out[h+g:,n+h+end]=np.float32(np.log(np.float32(graph['nodes'][end]['mass'])))
    return out
