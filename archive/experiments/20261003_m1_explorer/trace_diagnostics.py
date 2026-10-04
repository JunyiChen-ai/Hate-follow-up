#!/usr/bin/env python3
"""Post-run, annotation-free acquisition audit. Does not change any prediction."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import socket
import numpy as np
from explorer import choose_frames


def run(root):
    durations={(r['dataset'],r['video_id']):r['duration'] for r in map(json.loads,(root/'base/predictions.jsonl').open())}
    rows=[]
    for path in sorted((root/'details').glob('*/*.json')):
        d=json.loads(path.read_text());source=d['source'];video=path.stem
        windows=Counter();rounds=0;different_order=different_set=0;encoded=0;new=0
        grids=set();priors=[]
        for t in d['traces']:
            windows[f'support_{int(t["nominal_support"]>0)}_rounds_{len(t["rounds"])}']+=1
            observed=list(d['original_frame_times']);prior=t['initial_prior'];used=set()
            weights=np.asarray(prior,float)
            priors.append(float(-(weights*np.log(np.maximum(weights,1e-300))).sum()/math.log(len(weights))))
            for step in t['rounds']:
                # Reconstruct only the pre-existing 4 fps candidate mapping from
                # saved actual PTS. No media decode or annotation access here.
                start=8*t['window'];end=min(start+8,float(durations[(path.parent.name,video)]))
                times=np.asarray([e['time'] for e in source['frames']])
                excluded=set(source['legacy_excluded_indices'])|used;indices=set()
                for k in range(max(0,math.ceil(4*start-.5)),math.ceil(4*end-.5)):
                    target=(k+.5)/4
                    if not start<=target<end:continue
                    i=int(np.searchsorted(times,target))
                    if i<len(times) and start<=times[i]<end and i not in excluded:indices.add(i)
                candidates=[source['frames'][i] for i in sorted(indices)]
                selected=[candidates[i]['index'] for i in choose_frames(candidates,observed,prior)]
                actual=[e['index'] for e in step['added']]
                assert selected==actual
                distance=[candidates[i]['index'] for i in choose_frames(candidates,observed,prior,attention=False)]
                different_order+=selected!=distance;different_set+=set(selected)!=set(distance);rounds+=1
                used.update(actual);new+=len(actual);encoded+=len(step['image_counts'])
                grids.update(tuple(g) for g in step['image_grid_thw'])
                prior=step['prior'];observed=d['original_frame_times']+[e['time'] for e in step['acquired']]
        rows.append({'dataset':path.parent.name,'video_id':video,'windows':dict(windows),
            'acquisition_reads':rounds,'new_frames':new,'image_encodes':encoded,
            'different_order_from_distance':different_order,'different_set_from_distance':different_set,
            'initial_frame_prior_normalized_entropy_mean':float(np.mean(priors)),
            'new_image_grids':sorted(grids),'same_grid_within_video':len(grids)<=1,
            'candidate_exhausted_windows':sum(bool(t.get('candidate_exhausted')) for t in d['traces'])})
    result={'host':socket.gethostname(),'no_GT':True,'scope':'conditional trace diagnostic, not a counterfactual performance control',
        'input':str(root),'per_video':rows,'per_dataset':{}}
    for ds in sorted({r['dataset'] for r in rows}):
        subset=[r for r in rows if r['dataset']==ds];counts=Counter()
        for r in subset:counts.update(r['windows'])
        result['per_dataset'][ds]={'videos':len(subset),'windows':dict(counts),
            **{k:sum(r[k] for r in subset) for k in ('acquisition_reads','new_frames','image_encodes',
                'different_order_from_distance','different_set_from_distance','candidate_exhausted_windows')},
            'all_videos_same_grid':all(r['same_grid_within_video'] for r in subset)}
    (root/'trace_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['per_dataset'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);a=parser.parse_args()
    run(a.run)
