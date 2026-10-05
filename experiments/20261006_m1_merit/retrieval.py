"""Single-video MaxSim and actual temporal-neighborhood source selection."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
SPEC=json.loads((Path(__file__).parent/'spec.json').read_text())


def retrieve(keys,available,query,current,seen=()):
    keys=np.asarray(keys,dtype=np.float32);available=np.asarray(available,dtype=bool)
    assert keys.ndim==3 and keys.shape[:2]==available.shape and keys.shape[1]==4
    assert 0<=current<len(keys) and np.isfinite(keys).all()
    if query is None:return dict(anchors=[],scores=[],best_keys=[],candidates=[])
    query=np.asarray(query,dtype=np.float32);assert query.shape==(keys.shape[-1],) and np.isfinite(query).all()
    similarities=np.sum(keys*query[None,None,:],axis=-1,dtype=np.float32)
    similarities[~available]=-np.inf;best=np.argmax(similarities,axis=1);scores=np.max(similarities,axis=1)
    banned={current,*seen};valid=[i for i in range(len(keys)) if i not in banned and np.isfinite(scores[i])]
    order=sorted(valid,key=lambda i:(-float(scores[i]),i));anchors=order[:SPEC['retrieval_anchors']]
    candidates=[]
    for rank,anchor in enumerate(anchors):
        for index in range(max(0,anchor-SPEC['neighbor_radius']),min(len(keys),anchor+SPEC['neighbor_radius']+1)):
            if index not in banned:candidates.append((rank,abs(index-anchor),index))
    candidates.sort();chosen=[]
    for _,_,index in candidates:
        if index not in chosen:chosen.append(index)
        if len(chosen)>=SPEC['candidates_per_round']:break
    return dict(anchors=anchors,scores=[float(scores[i]) for i in order],ranked_indices=order,
        best_keys=[SPEC['key_order'][int(best[i])] for i in order],candidates=sorted(chosen))


def compile_filter(result,candidates):
    assert result['status'] in ('SUPPORTED','INSUFFICIENT','UNKNOWN')
    ids=result['ids'];assert all(type(i) is int and i in candidates for i in ids) and len(set(ids))==len(ids)
    assert len(ids)<=SPEC['max_remote_windows']
    return [] if result['status']=='UNKNOWN' else sorted(ids)


def union_sources(previous,new):
    result=list(previous)
    for index in new:
        if index not in result and len(result)<SPEC['max_remote_windows']:result.append(index)
    assert len(result)<=SPEC['max_remote_windows'] and len(set(result))==len(result)
    return result
