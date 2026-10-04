#!/usr/bin/env python3
"""Actual cached input geometry/opportunity counts; no model/GT/predictions."""
import json
from pathlib import Path
import socket
import sys
from collections import Counter
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from extract import selected_rows,validate_cached,CACHE,DATASETS
from control_inputs import temporal_tree,packet_control
from tree import window_packet
from src.video_inputs import fixed_windows


def main():
    print('host '+socket.gethostname(),flush=True)
    out=ROOT/'runs/20261004_m1_tree/cpu_control_preflight';out.mkdir(parents=True,exist_ok=True)
    results=[]
    for row in selected_rows(False):
        m,f=validate_cached(CACHE/row['dataset']/row['video_id'],row)
        # Only a geometry placeholder: never persisted as captions or scored.
        temporal=temporal_tree(m['tree'],f,m['entries'],lambda i:dict(text='CPU GEOMETRY PLACEHOLDER',tokens=[]))
        nodes={n['id']:n for n in temporal['nodes']};old={n['id']:n for n in m['tree']['nodes']}
        assert nodes.keys()==old.keys()
        roots=[n for n in nodes.values() if n['parent'] is None]
        assert Counter(i for n in roots for i in n['members'])==Counter(range(len(m['entries'])))
        for key,n in nodes.items():
            assert (n['parent'],n['depth'],n['root_relevance'],len(n['members']))==(
                old[key]['parent'],old[key]['depth'],old[key]['root_relevance'],len(old[key]['members']))
            assert n['members']==list(range(n['members'][0],n['members'][-1]+1))
            kids=[c for c in nodes.values() if c['parent']==key]
            if kids:assert Counter(i for c in kids for i in c['members'])==Counter(n['members'])
        changed=Counter();images=Counter();extra=set()
        for original,(a,b) in zip(m['packets'],fixed_windows(float(row['duration']),8)):
            for arm in ('flat','wrong_links','no_depth','no_added_pixels','temporal'):
                p=window_packet(temporal,f,m['entries'],a,b) if arm=='temporal' else packet_control(arm,m['tree'],f,m['entries'],a,b)
                assert all(a<=m['entries'][i]['time']<b for i in p['pool_members'])
                if arm=='temporal':
                    changed[arm]+=any(p[k]!=original[k] for k in ('pool_members','leaf_ids','ancestor_ids'))
                else:changed[arm]+=p!=original
                images[arm]+=len(p['pool_members'])
                if arm in ('no_depth','temporal'):extra.update(p['pool_members'])
        missing=[i for i in extra if not (CACHE/row['dataset']/row['video_id']/'witnesses'/f'frame_{m["entries"][i]["index"]:08d}.png').is_file()]
        results.append(dict(dataset=row['dataset'],video_id=row['video_id'],nodes=len(nodes),
            new_temporal_caption_frames=len(temporal['caption_events']),reused_caption_frames=len(temporal['reused_main_captions']),
            changed_windows=dict(changed),local_images=dict(images),extra_local_witness_frames=len(missing)))
    summary=dict(host=socket.gethostname(),GT_read=False,model_run=False,coverage=len(results),
        scope='geometry/opportunity count only; placeholder texts discarded, no control predictions; temporal changed_windows counts selected indices/links only, not ungenerated captions',datasets={})
    for ds in DATASETS:
        rr=[r for r in results if r['dataset']==ds]
        summary['datasets'][ds]=dict(videos=len(rr),**{k:sum(r[k] for r in rr) for k in (
            'nodes','new_temporal_caption_frames','reused_caption_frames','extra_local_witness_frames')},
            changed_windows={a:sum(r['changed_windows'][a] for r in rr) for a in ('flat','wrong_links','no_depth','no_added_pixels','temporal')},
            local_images={a:sum(r['local_images'][a] for r in rr) for a in ('flat','wrong_links','no_depth','no_added_pixels','temporal')})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'per_video.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True);print('CONTROL_INPUT_PREFLIGHT_DONE',flush=True)


if __name__=='__main__':main()
