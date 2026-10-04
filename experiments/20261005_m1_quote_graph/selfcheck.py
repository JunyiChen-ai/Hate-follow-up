#!/usr/bin/env python3
"""Hand graph/source and access-matrix checks, no dataset labels or predictions."""
import json
import copy
import numpy as np
from graph import sid,uid,parse_chunk,packet,chunk_indices
from reader import attention_bias


def main():
    windows=[dict(i=i,start=8*i,end=8*(i+1),body='abcdefghij') for i in range(8)]
    nodes=[]
    for w,a,b,kind in [(0,0,1,'mention'),(4,0,1,'mention'),(4,2,4,'quote'),
                        (4,5,7,'negation'),(6,0,3,'correction'),(7,0,3,'endorsement')]:
        n=dict(window=w,start=a,end=b,kind=kind);n['id']=sid(n);nodes.append(n)
    m0,m4,q4,unrelated,c6,c7=[n['id'] for n in nodes]
    graph=dict(nodes=nodes,edges=[dict(type=t,**{'from':a,'to':b}) for t,a,b in
        [('same_referent',m0,m4),('attributed_to',q4,m4),('corrects',c6,q4),('endorses',c7,c6)]])
    p=packet(graph,windows,0);selected={n['id'] for n in p['selected']}
    assert selected=={m4,q4,c6},selected
    assert p['distances']=={m4:0,q4:1,c6:2}
    assert unrelated not in selected and c7 not in selected
    assert chunk_indices(8)==[list(range(8))] and chunk_indices(9)==[list(range(8)),[6,7,8]]
    empty=dict(nodes=[],edges=[])
    valid=dict(nodes=[dict(id='x0',window=0,start=0,end=2,kind='mention')],
               edges=[dict(type='refers_to',**{'from':'x0','to':uid(0)})])
    parsed,error=parse_chunk(json.dumps(valid),False,windows,list(range(8)),empty)
    assert error is None and len(parsed['nodes'])==1 and len(parsed['edges'])==1
    variants=[]
    for field,value in [('window',8),('start',True),('end',99),('kind','unknown')]:
        v=copy.deepcopy(valid);v['nodes'][0][field]=value;variants.append(json.dumps(v))
    v=copy.deepcopy(valid);v['edges'].append(dict(type='attributed_to',**{'from':'x0','to':uid(0)}));variants.append(json.dumps(v))
    variants.append('{"nodes":[],"nodes":[],"edges":[]}');variants.append(json.dumps(valid)+' prose')
    for text in variants:
        result,error=parse_chunk(text,False,windows,list(range(8)),empty)
        assert error and result==empty,'salvaged part of invalid chunk'
    assert parse_chunk(json.dumps(valid),True,windows,list(range(8)),empty)[0]==empty
    print('PASS hand component/2-hop/directed-membership graph, exact spans, whole-chunk rejection')
    ranges=[[0,2],[2,5],[5,7],[7,10],[10,13]];prefix=4
    bias=attention_bias(ranges,prefix)
    for q in range(13):
        if q<2:allowed=set(range(q+1))
        elif q<5:allowed={0,1}|set(range(2,q+1))
        elif q<7:allowed={0,1}|set(range(5,q+1))
        elif q<10:allowed=set(range(q+1))
        else:allowed={0,1}|set(range(7,q+1))
        assert set(np.flatnonzero(np.isfinite(bias[q,prefix:])))==allowed
        assert np.all(bias[q,:prefix]==0)
    no_context=attention_bias([[0,2],[2,5],[5,8]],prefix)
    expected=np.concatenate([np.zeros((8,prefix)),np.where(np.tri(8,dtype=bool),0.,-np.inf)],axis=1)
    np.testing.assert_array_equal(no_context,expected)
    print('PASS independently enumerated context/local/query accessibility and contextless causal degeneration')


if __name__=='__main__':main()
