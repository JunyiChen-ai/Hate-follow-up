#!/usr/bin/env python3
"""Postscore descriptive error analysis; never imported by scoring."""
import json
import os
import socket
import time
import numpy as np
from extract import ROOT,CACHE,DATASETS
from analyze import read


def main():
    run=ROOT/'runs/20261004_m1_lattice/r1_full_main'
    report=ROOT/'runs/20261004_m1_lattice/r1_full_main_analysis'
    out=ROOT/'runs/20261004_m1_lattice/r1_error_analysis';out.mkdir(parents=True,exist_ok=True)
    print(socket.gethostname(),flush=True);(out/'run.pid').write_text(str(os.getpid()))
    raw={a:read(run/a/'predictions.jsonl') for a in ('base','optimized')}
    final={a:read(run.parent/'r1_full_main_decoded'/a/'predictions.jsonl') for a in raw}
    paired={(r['dataset'],r['video_id']):r for r in json.loads((report/'per_video.json').read_text())}
    sources=[str(p.relative_to(ROOT)) for p in [report/'summary.json',report/'per_video.json']]
    sources += [str((run/a/'predictions.jsonl').relative_to(ROOT)) for a in raw]
    sources += [str((run.parent/'r1_full_main_decoded'/a/'predictions.jsonl').relative_to(ROOT)) for a in raw]
    rows=[];summary={};examples=[]
    for ds in DATASETS:
        path=ROOT/f'data/gt_4fps/{ds}.npz';sources.append(str(path.relative_to(ROOT)))
        with np.load(path,allow_pickle=True) as gt:
            ys={str(v):np.asarray(gt['y4'][i]) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        sums={kind:dict(n=0,base=0.,new=0.,base_yes=0,new_yes=0) for kind in ('speech_positive','speech_negative','final_positive','final_negative')}
        for key,b in raw['base'].items():
            if key[0]!=ds:continue
            c=raw['optimized'][key];y=ys[key[1]];n=min(len(y),len(b['score_curve']))
            y=y[:n];idx=np.clip(((np.arange(n)+.5)/4//8).astype(int),0,len(b['extra']['windows'])-1)
            m=json.loads((CACHE/ds/(key[1]+'.json')).read_text())
            r=dict(dataset=ds,video_id=key[1],gt_video_has_positive=bool(y.any()),native_global=b['extra']['z_video'],
                within_delta=(paired[key]['optimized']['final']-paired[key]['base']['final']) if key in paired else None,
                final_mean_delta=float(np.mean(np.asarray(final['optimized'][key]['score_curve'])[:n]-np.asarray(final['base'][key]['score_curve'])[:n])),
                windows=[])
            for a,d,w in zip(b['extra']['windows'],c['extra']['windows'],m['windows']):
                mask=idx==a['i'];yy=y[mask];nw=int(mask.sum())
                item=dict(i=a['i'],frames=nw,gt_positive_fraction=float(yy.mean()) if nw else None,
                    base_speech=a.get('z_speech'),new_speech=d.get('z_speech'),
                    truncated_beams=sum(t['truncated'] for t in w['beams']) if w['available'] else 0,
                    alternative_slots=sum(len(s['alternatives'])>1 for s in w['confusion']['slots']) if w['available'] else 0,
                    slots=len(w['confusion']['slots']) if w['available'] else 0)
                r['windows'].append(item)
                if 'z_speech' in a and 'z_speech' in d:
                    for positive in (False,True):
                        z=sums['speech_positive' if positive else 'speech_negative'];cnt=int(np.sum(yy==positive))
                        z['n']+=cnt;z['base']+=cnt*a['z_speech'];z['new']+=cnt*d['z_speech']
                        z['base_yes']+=cnt*(a['z_speech']>0);z['new_yes']+=cnt*(d['z_speech']>0)
            for positive in (False,True):
                use=y==positive;z=sums['final_positive' if positive else 'final_negative'];z['n']+=int(use.sum())
                z['base']+=float(np.asarray(final['base'][key]['score_curve'])[:n][use].sum())
                z['new']+=float(np.asarray(final['optimized'][key]['score_curve'])[:n][use].sum())
            rows.append(r)
        summary[ds]={k:dict(n=v['n'],base_mean=v['base']/v['n'],new_mean=v['new']/v['n'],
            delta_mean=(v['new']-v['base'])/v['n'],base_yes_rate=v['base_yes']/v['n'] if k.startswith('speech') else None,
            new_yes_rate=v['new_yes']/v['n'] if k.startswith('speech') else None) for k,v in sums.items()}
        rr=[r for r in rows if r['dataset']==ds]
        groups=dict(negative_video_increase=sorted([r for r in rr if not r['gt_video_has_positive']],key=lambda r:-r['final_mean_delta'])[:3],
            within_decrease=sorted([r for r in rr if r['within_delta'] is not None],key=lambda r:r['within_delta'])[:3],
            within_increase=sorted([r for r in rr if r['within_delta'] is not None],key=lambda r:-r['within_delta'])[:3])
        for kind,chosen in groups.items():
            for r in chosen:
                metadata=json.loads((CACHE/ds/(r['video_id']+'.json')).read_text())
                candidates=[w for w in r['windows'] if w['base_speech'] is not None and w['new_speech'] is not None]
                chosen_windows=sorted(candidates,key=lambda w:-(w['new_speech']-w['base_speech']))[:2]
                examples.append(dict(group=kind,video=r,source=str((CACHE/ds/(r['video_id']+'.json')).relative_to(ROOT)),
                    speech_examples=[dict(diagnostic=w,input=metadata['windows'][w['i']]) for w in chosen_windows]))
    payload=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),scope='development-selected postscore only; no scoring/fitting/threshold input',
        GT_read=True,sources=sources,input_root=str(CACHE.relative_to(ROOT)),datasets=summary)
    (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    (out/'per_video.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(payload,indent=2));print('ERROR_ANALYSIS_DONE')


if __name__=='__main__':main()
