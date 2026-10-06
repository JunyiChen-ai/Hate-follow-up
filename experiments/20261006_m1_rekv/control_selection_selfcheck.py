"""Independent small-set donor optimality, token and temporal-control checks."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from control_selection import wrong_donors,matched_preference,time_order,Selection


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(0);cases=0
    for n in range(2,9):
        for attempt in range(25):
            blocks=[dict(shape=[2,36,8,int(rng.integers(2,5)),128],source_index=i*16,actual_time=i*2.,ancestors=list(range(i)),image_rows=[0]) for i in range(n)]
            local=[0];remote=list(range(1,n));k=min(4,len(remote));original=list(rng.choice(remote,k,replace=False))
            actual=wrong_donors(blocks,local,original);budget=Counter(blocks[i]['shape'][3] for i in original)
            feasible=[ids for ids in combinations(remote,k) if Counter(blocks[i]['shape'][3] for i in ids)==budget]
            best=max(len(set(ids)-set(original)) for ids in feasible)
            assert len(set(actual)-set(original))==best and set(actual).isdisjoint(local)
            assert actual==wrong_donors(blocks,local,original)
            chronology=time_order(blocks,local,[0,2]);assert chronology==remote
            for preference in [chronology,list(reversed(remote))]:
                chosen=matched_preference(blocks,original,preference)
                assert Counter(blocks[i]['shape'][3] for i in chosen)==budget and len(set(chosen))==k
            cases+=1
    blocks=[dict(shape=[2,36,8,3+(i%2),128],source_index=i*16,actual_time=i*2.,ancestors=list(range(i)),image_rows=[0]) for i in range(8)]
    local=[2,3];bounds=[4,8];r0=[dict(remote_ids=[0,4,6,7]),dict(remote_ids=[1,4,5,7])]
    scores=np.array([9,8,7,6,5,4,3,2],np.float32)
    for arm in ['R0','L0','C0','L1','C1','L2','D0']:
        selector=Selection(arm,blocks,local,bounds,r0)
        first=selector.choose(0,r0[0]['remote_ids'],scores);second=selector.choose(1,r0[1]['remote_ids'],scores)
        assert set(first).isdisjoint(local) and set(second).isdisjoint(local)
        if arm=='L0':assert first==second==[]
        elif arm=='L1':assert first==second==r0[0]['remote_ids']
        elif arm in ('C1','L2','D0'):
            for chosen,ref in zip([first,second],r0):assert Counter(blocks[i]['shape'][3] for i in chosen)==Counter(blocks[i]['shape'][3] for i in ref['remote_ids'])
        try:selector.choose(1,r0[1]['remote_ids'],scores)
        except AssertionError:pass
        else:raise AssertionError('repeated actual layer accepted')
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,exhaustive_optimal_donor_cases=cases,exact_token_multisets=True,
        actual_time_ties=True,shared_layer0_state=True,repeated_layer_rejected=True,
        scope='CPU selection arithmetic only; no model/retrieval/GT/performance or executed ablations'),indent=2)+'\n')
    print('CONTROL_SELECTION_CPU_PASS',cases)


if __name__=='__main__':main()
