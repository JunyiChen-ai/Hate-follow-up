#!/usr/bin/env python3
"""Hand end-state attention oracles and fixed tokenizer boundary check; no GT."""
import copy
import numpy as np
from terminal_graph import tokenize_paths,bias,positions
from path_graph import tokenize_paths as full_paths


class Characters:
    def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]


def main():
    texts=['ab','c','ab','','de'];beams=[dict(text=s,score=0.) for s in texts]
    g=tokenize_paths(beams,Characters());old=full_paths(beams,Characters())
    assert [p['text'] for p in g['paths']]==['ab','c','','de']
    assert [p['mass'] for p in g['paths']]==[p['mass'] for p in old['paths']]
    assert g['ids']==[ord(c) for s in (' ab\n',' c\n',' de\n') for c in s]
    assert g['terminal_indices']==[3,6,10] and g['path_length']==4
    assert g['paths'][2]['tokens']==[] and g['paths'][2]['lexical_tokens']==[]
    n,h,t,start=2,2,3,11;actual=bias(g,h,t,n);expected=np.full_like(actual,-np.inf)
    for q in range(len(expected)):
        expected[q,:n]=0.
        if q<h:expected[q,n:n+q+1]=0.
        elif q<h+len(g['ids']):
            expected[q,n:n+h]=0.;owner=g['nodes'][q-h]['path']
            for k,node in enumerate(g['nodes']):
                if node['path']==owner and k<=q-h:expected[q,n+h+k]=0.
        else:
            expected[q,n:n+h]=0.
            for k,node in enumerate(g['nodes']):
                if node['terminal']:expected[q,n+h+k]=np.float32(np.log(np.float32(node['mass'])))
            expected[q,n+h+len(g['ids']):n+q+1]=0.
    assert np.array_equal(actual,expected)
    assert positions(g,h,t,start)==[11,12,13,14,15,16,13,14,15,13,14,15,16,17,18,19]
    unit=tokenize_paths([dict(text='utterance',score=0.)]*5,Characters())
    ub=bias(unit,h,t,n);ll=h+len(unit['ids'])+t;uu=np.full_like(ub,-np.inf)
    for q in range(ll):
        uu[q,:n]=0.
        if q<h+len(unit['ids']):uu[q,n:n+q+1]=0.
        else:
            uu[q,n:n+h]=0.;uu[q,n+h+len(unit['ids'])-1]=0.;uu[q,n+h+len(unit['ids']):n+q+1]=0.
    assert np.array_equal(ub,uu) and positions(unit,h,t,start)==list(range(start,start+ll))
    assert not np.array_equal(ub[:,n:],np.where(np.tri(ll,dtype=bool),0.,-np.inf))
    empty=tokenize_paths([dict(text='',score=0.)]*5,Characters())
    assert empty['ids']==[] and empty['terminal_indices']==[]
    assert np.array_equal(bias(empty,h,t,n)[:,n:],np.where(np.tri(h+t,dtype=bool),0.,-np.inf))
    under=tokenize_paths([dict(text='good',score=0.)]+[dict(text='zero',score=-1000.)]*4,Characters())
    assert under['paths'][1]['text']=='zero' and under['paths'][1]['tokens']==[] and len(under['terminal_indices'])==1
    from transformers import AutoTokenizer
    from src.mllm_judge import MODEL
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    assert len(tok.encode('\n',add_special_tokens=False))==1
    real=tokenize_paths(beams,tok);assert len(real['ids'])==sum(len(p['lexical_tokens']) for p in real['paths'])+3
    print('PASS hand every-row endpoint-only visibility, own utterance/history, mass/epsilon/underflow/positions')
    print('PASS one-path independently constructed end-state matrix and deliberately non-causal-all-word contract')
    print('PASS actual frozen Qwen newline is one token; no GPU/GT/real predictions')


if __name__=='__main__':main()
