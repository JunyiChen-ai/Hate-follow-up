#!/usr/bin/env python3
"""Hand oracles for complete-path correlation, epsilon and weighted accessibility."""
import numpy as np
from path_graph import tokenize_paths,bias,positions


class Characters:
    def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]


def main():
    texts=['ab','cd','ab','','ef'];g=tokenize_paths([dict(text=s,score=0.) for s in texts],Characters())
    assert [p['text'] for p in g['paths']]==['ab','cd','','ef']
    assert np.allclose([p['mass'] for p in g['paths']],[.4,.2,.2,.2])
    assert [p['beams'] for p in g['paths']]==[[0,2],[1],[3],[4]]
    assert g['paths'][2]['tokens']==[] and g['path_length']==3
    assert g['ids']==[ord(c) for s in (' ab',' cd',' ef') for c in s]
    n,h,t,start=2,2,3,11;actual=bias(g,h,t,n);length=h+len(g['ids'])+t
    expected=np.full_like(actual,-np.inf)
    for q in range(length):
        expected[q,:n]=0.
        if q<h:
            for k in range(q+1):expected[q,n+k]=0.
        elif q<h+len(g['ids']):
            for k in range(h):expected[q,n+k]=0.
            owner=g['nodes'][q-h]['path']
            for k,v in enumerate(g['nodes']):
                if v['path']==owner and k<=q-h:expected[q,n+h+k]=0.
        else:
            for k in range(h):expected[q,n+k]=0.
            for k,v in enumerate(g['nodes']):expected[q,n+h+k]=np.float32(np.log(np.float32(v['mass'])))
            for k in range(h+len(g['ids']),q+1):expected[q,n+k]=0.
    assert np.array_equal(actual,expected)
    assert positions(g,h,t,start)==[11,12]+[13,14,15]*3+[16,17,18]
    unit=tokenize_paths([dict(text='intact sentence.',score=0.)]*5,Characters())
    b=bias(unit,h,t,n);ll=h+len(unit['ids'])+t
    assert np.array_equal(b[:,:n],np.zeros((ll,n)))
    assert np.array_equal(b[:,n:],np.where(np.tri(ll,dtype=bool),0.,-np.inf))
    assert positions(unit,h,t,start)==list(range(start,start+ll))
    empty=tokenize_paths([dict(text='',score=0.)]*5,Characters());assert empty['ids']==[]
    assert np.array_equal(bias(empty,h,t,n)[:,n:],np.where(np.tri(h+t,dtype=bool),0.,-np.inf))
    print('PASS hand duplicate/epsilon/source-path identity and every-row accessibility/logmass oracle')
    print('PASS logical shared path starts, complete probability-one and empty-path causal degeneration; no GPU or GT')


if __name__=='__main__':main()
