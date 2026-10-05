"""Separable three-slot source assignment, no labels or output scores."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
SPEC=json.loads((Path(__file__).parent/'spec.json').read_text())


def assign(vectors,available,queries,current):
    vectors=np.asarray(vectors,dtype=np.float32);available=np.asarray(available,dtype=bool)
    assert vectors.ndim==2 and available.shape==(len(vectors),) and 0<=current<len(vectors)
    result={};objective=np.float32(0)
    for role in SPEC['slot_order']:
        query=queries[role]
        candidates=[current] if role=='current' else [i for i in range(len(vectors)) if (i<current if role=='prequel' else i>current)]
        candidates=[i for i in candidates if available[i]]
        scores=[]
        if query is not None:
            q=np.asarray(query,dtype=np.float32);assert q.shape==vectors.shape[1:] and np.isfinite(q).all()
            scores=[dict(id=i,cosine=float(np.dot(vectors[i],q))) for i in candidates]
        if role=='current':
            selected=current;score=scores[0]['cosine'] if scores else SPEC['none_score']
        else:
            eligible=[r for r in scores if r['cosine']>SPEC['none_score']]
            best=min(eligible,key=lambda r:(-r['cosine'],r['id'])) if eligible else None
            selected=best['id'] if best else None;score=best['cosine'] if best else SPEC['none_score']
        result[role]=dict(source_id=selected,score=score,candidates=scores)
        objective+=np.float32(score)
    return dict(bindings=result,objective=float(objective),source_ids=[result[r]['source_id'] for r in ('prequel','sequel') if result[r]['source_id'] is not None])
