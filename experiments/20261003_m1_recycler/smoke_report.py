#!/usr/bin/env python3
"""Real-input smoke plumbing and cost only; no GT or performance metrics."""
import json
from pathlib import Path
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
run=ROOT/'runs/20261003_m1_recycler/r1_smoke'
def read(p):return {(r['dataset'],r['video_id']):r for r in map(json.loads,p.open())}
checks=read(run/'checks.jsonl');raw={a:read(run/a/'predictions.jsonl') for a in ('base','eager','recycle')}
old=read(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl')
assert len(checks)==5 and all(r.keys()==checks.keys() for r in raw.values())
count=0;rows=[]
for k,c in checks.items():
    b=raw['base'][k];h=old[k]
    assert b['extra']['z_video']==h['extra']['z_video'] and np.array_equal(b['score_curve'],h['score_curve'])
    ws=b['extra']['windows'];V=len(ws);B=sum(1+('z_speech' in w) for w in ws)
    assert c['actual_forwards']==6+2*B+2*V and c['verify']['native_windows_exact']
    for w,v in zip(ws,h['extra']['windows']):
        for name in ('start','end','z_visual','z_speech'):
            assert w.get(name)==v.get(name);count+=name.startswith('z_') and name in w
    changes={};cost={}
    for a,r in raw.items():
        v=r[k];assert v['extra']['z_video']==b['extra']['z_video'] and v['extra']['stance']==b['extra']['stance']
        assert len(v['score_curve'])==len(b['score_curve']) and np.isfinite(v['score_curve']).all() and v['native_rate']==4
        assert len(v['extra']['windows'])==V and v['calls']==3+B
        for bw,vw in zip(ws,v['extra']['windows']):
            assert (bw['start'],bw['end'],bw.get('z_speech'))==(vw['start'],vw['end'],vw.get('z_speech'))
        changes[a]=max(abs(vw['z_visual']-bw['z_visual']) for bw,vw in zip(ws,v['extra']['windows']))
        cost[a]=v['extra']['standalone_seconds']
    stats={}
    for a in ('eager','recycle'):
        z=np.asarray([d[a]['statistics'] for d in c['attention']]);assert z.shape==(V,35,6) and np.isfinite(z).all()
        assert (z[:,:,5]<2e-6).all()
        if a=='eager':assert (z[:,:,4]==0).all()
        stats[a]={'mean_selected_fraction':float(z[:,:,3].mean()),'mean_transferred_mass':float(z[:,:,4].mean()),
            'max_sum_error':float(z[:,:,5].max()),'layers_with_transfer':np.flatnonzero(z[:,:,4].max(0)>0).tolist()}
    with np.load(run/'masks'/k[0]/(k[1]+'.npz')) as t:
        assert t['channels'].shape==(35,2) and t['cached_sinks'].shape==(35,c['cached_tokens'])
        sink_counts=t['cached_sinks'].sum(1).tolist()
    rows.append({'dataset':k[0],'video_id':k[1],'prefix_tokens':c['prefix_tokens'],'peak_GiB':c['peak_GiB'],
        'max_abs_visual_change':changes,'standalone_seconds':cost,'attention':stats,'cached_sink_counts':sink_counts,
        'paired_seconds':sum(c['seconds'].values())})
estimates={}
for ds,n in [('HateMM',215),('HateClipSeg',118)]:
    sample=[r for r in rows if r['dataset']==ds and r['video_id']!='hate_video_114'];assert len(sample)==2
    estimates[ds]={a:n*np.mean([r['standalone_seconds'][a] for r in sample]) for a in raw}
    estimates[ds]['paired']=n*np.mean([r['paired_seconds'] for r in sample])
result={'no_GT':True,'videos':5,'native_globals_exact':5,'native_branches_exact':count,'rows':rows,
    'extrapolated_seconds_from_first2_each_not_stress':estimates}
(run/'plumbing_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
