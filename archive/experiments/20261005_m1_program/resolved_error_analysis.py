"""Post-run R3 GT/source diagnostics only, never imported by a scoring path."""
import json
import os
import socket
from collections import Counter
import numpy as np
from handle_analyze import ROOT,read
from handle_inputs import CACHE,DATASETS
from handle_measure import question,available_field


def main():
    rawroot=ROOT/'runs/20261005_m1_program/r3_handles_full_main'
    rr={k:read(rawroot/k/'predictions.jsonl') for k in ('base','optimized')}
    per=json.loads((ROOT/'runs/20261005_m1_program/r3_handles_full_main_analysis/per_video.json').read_text())
    result=dict(host=socket.gethostname(),GT_read=True,development_selected=True,
        read_paths=[str(rawroot.relative_to(ROOT))+'/{base,optimized}/predictions.jsonl',
            'runs/20261005_m1_program/r3_handles_full_main_analysis/per_video.json',str(CACHE.relative_to(ROOT))+'/<dataset>/*.json',
            'data/gt_4fps/{HateMM,HateClipSeg}.npz'],datasets={})
    for ds in DATASETS:
        gt=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True)
        ys={str(v):np.asarray(gt['y4'][i]) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        counts=Counter(); source_rows=[]
        for key,base in rr['base'].items():
            if key[0]!=ds:continue
            new=rr['optimized'][key]; metadata=json.loads((CACHE/ds/(key[1]+'.json')).read_text()); y=ys[key[1]]
            for b,n,w in zip(base['extra']['windows'],new['extra']['windows'],metadata['windows']):
                _,rec=question(w['execution'],b['i'],len(metadata['windows']),b['start'],b['end'],'','visual',3)
                actions=[e['value'] for e in rec['evidence'] if e['value']['kind']=='action']
                complete=any(all(available_field(a[f]) for f in ('actor','action','target')) for a in actions)
                action=any(available_field(a['action']) for a in actions)
                counts['windows']+=1;counts['action_resolved']+=action;counts['complete_tuple']+=complete
                counts['resolved_action_incomplete_tuple']+=action and not complete
                if action:
                    sl=y[int(b['start']*4):min(len(y),int(np.ceil(b['end']*4)))]
                    source_rows.append(dict(video_id=key[1],i=b['i'],start=b['start'],end=b['end'],GT_positive_fraction=float(sl.mean()) if len(sl) else None,
                        complete_tuple=complete,actions=actions,native_V=b['z_visual'],new_V=n['z_visual'],native_max=b['z'],new_max=n['z'],speech=b.get('z_speech')))
        videos=[r for r in per if r['dataset']==ds]
        worst=sorted(videos,key=lambda r:r['optimized']['final']-r['base']['final'])[:8]
        result['datasets'][ds]=dict(source_counts=dict(counts),worst=[dict(**r,windows=[s for s in source_rows if s['video_id']==r['video_id']]) for r in worst],
            largest_false_increases=sorted([s for s in source_rows if s['GT_positive_fraction']==0],key=lambda s:s['new_max']-s['native_max'],reverse=True)[:12],
            largest_positive_decreases=sorted([s for s in source_rows if s['GT_positive_fraction'] is not None and s['GT_positive_fraction']>0],key=lambda s:s['new_max']-s['native_max'])[:12])
    out=ROOT/'runs/20261005_m1_program/r3_error_analysis';out.mkdir(parents=True,exist_ok=True)
    (out/'run.pid').write_text(str(os.getpid()));(out/'source_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    for ds,d in result['datasets'].items():
        print(ds,d['source_counts'])
        for r in d['worst'][:3]:print(r['video_id'],r['optimized']['final']-r['base']['final'],[(w['i'],w['complete_tuple'],w['GT_positive_fraction'],w['native_V'],w['new_V']) for w in r['windows'][:6]])
    print('ERROR_ANALYSIS_DONE',flush=True)


if __name__=='__main__':main()
