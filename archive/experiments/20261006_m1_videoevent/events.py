"""Deterministic input-event selection; no prediction scores or labels."""
import json
import math
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
SPEC=json.loads((Path(__file__).parent/'spec.json').read_text())


def candidates(current,count):return list(range(max(0,current-SPEC['candidate_radius']),min(count,current+SPEC['candidate_radius']+1)))


def select(ids,values,current):
    assert ids==sorted(set(ids)) and len(ids)==len(values) and current in ids
    assert all(v is None or type(v) is int and 1<=v<=10 for v in values)
    smooth=[]
    for k,value in enumerate(values):
        available=[v for v in values[max(0,k-1):min(len(values),k+2)] if v is not None]
        smooth.append(None if value is None else sum(available)/len(available))
    valid=[v for v in smooth if v is not None];threshold=sum(valid)/len(valid) if valid else None;groups=[]
    for i,v in zip(ids,smooth):
        if v is None or v<threshold:continue
        if groups and i==groups[-1][-1]+1:groups[-1].append(i)
        else:groups.append([i])
    scores=dict(zip(ids,smooth));chosen=min(groups,key=lambda g:(-max(scores[i] for i in g),g[0])) if groups else []
    kept=sorted(sorted(chosen,key=lambda i:(abs(i-current),i))[:SPEC['max_event_windows']])
    representative=min(kept,key=lambda i:(-scores[i],i)) if kept else None
    return dict(candidate_ids=ids,relevance=values,smoothed=smooth,threshold=threshold,events=groups,selected_windows=kept,representative=representative,source_ids=[i for i in kept if i!=current])
