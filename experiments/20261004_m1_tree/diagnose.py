#!/usr/bin/env python3
"""Postscore error analysis only; never called by acquisition or reader."""
import json
import socket
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


def read(path):
    out={}
    for line in path.open():
        r=json.loads(line);key=r['dataset'],r['video_id'];assert key not in out;out[key]=r
    return out


def main():
    out=ROOT/'runs/20261004_m1_tree/r1_error_analysis';out.mkdir(parents=True,exist_ok=True)
    root=ROOT/'runs/20261004_m1_tree/r1_full_main';decoded=root.parent/'r1_full_main_decoded'
    raw={a:read(root/a/'predictions.jsonl') for a in ('base','optimized')}
    final={a:read(decoded/a/'predictions.jsonl') for a in raw}
    per={ (r['dataset'],r['video_id']):r for r in json.loads((root.parent/'r1_full_main_analysis/per_video.json').read_text())}
    results=[];summary=dict(host=socket.gethostname(),GT_read=True,scope='development-selected postscore descriptive diagnostic; no new predictions or gate',
        sources=['runs/20261004_m1_tree/r1_full_main/{base,optimized}/predictions.jsonl',
            'runs/20261004_m1_tree/r1_full_main_decoded/{base,optimized}/predictions.jsonl',
            'runs/20261004_m1_tree/r1_full_main_analysis/per_video.json','data/gt_4fps/{HateMM,HateClipSeg}.npz'],datasets={})
    for ds in ('HateMM','HateClipSeg'):
        g=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True)
        ys={str(v):np.asarray(g['y4'][i]) for i,v in enumerate(g['video_ids']) if str(g['split'][i])=='test'}
        rows=[]
        for key in [k for k in raw['base'] if k[0]==ds]:
            y=ys[key[1]];r=dict(dataset=ds,video_id=key[1],positive_frames=int(y.sum()),frames=len(y),positive_fraction=float(y.mean()),base={},optimized={})
            for arm in raw:
                x=raw[arm][key];ws=x['extra']['windows'];mean=float(np.mean([w['z'] for w in ws]));zv=x['extra']['z_video']
                r[arm]=dict(global_margin=zv,stance=x['extra']['stance'],mean_raw_max=mean,
                    K=zv+mean,final_video_key=float(np.mean(final[arm][key]['score_curve'])))
            r['global_delta']=r['optimized']['global_margin']-r['base']['global_margin']
            r['mean_raw_max_delta']=r['optimized']['mean_raw_max']-r['base']['mean_raw_max']
            r['K_delta']=r['optimized']['K']-r['base']['K']
            if key in per:r['within_delta']={k:per[key]['optimized'][k]-per[key]['base'][k] for k in ('final','raw_max')}
            rows.append(r)
        for arm in raw:
            ranks=rankdata([r[arm]['final_video_key'] for r in rows],method='average')/len(rows)
            for r,rank in zip(rows,ranks):r[arm]['video_key_rank']=float(rank)
        for r in rows:r['rank_delta']=r['optimized']['video_key_rank']-r['base']['video_key_rank']
        groupstats={}
        for group,pick in [('any_positive',lambda r:r['positive_frames']>0),('all_negative',lambda r:r['positive_frames']==0)]:
            rr=[r for r in rows if pick(r)];weights=np.asarray([r['positive_frames'] if group=='any_positive' else r['frames'] for r in rr])
            groupstats[group]=dict(n=len(rr),frames=sum(r['frames'] for r in rr),
                **{k:dict(mean=float(np.mean([r[k] for r in rr])),weighted_mean=float(np.average([r[k] for r in rr],weights=weights)),median=float(np.median([r[k] for r in rr])))
                    for k in ('global_delta','mean_raw_max_delta','K_delta','rank_delta')},
                stance_flips=sum(r['base']['stance']!=r['optimized']['stance'] for r in rr))
        pos=[r for r in rows if r['positive_frames']>0]
        summary['datasets'][ds]=dict(groups=groupstats,
            largest_positive_mass_rank_losses=[r['video_id'] for r in sorted(pos,key=lambda r:r['rank_delta']*r['positive_frames'])[:10]],
            positive_mass_downweighted_by_global=sum(r['positive_frames'] for r in pos if r['global_delta']<0),
            total_positive_frames=sum(r['positive_frames'] for r in pos))
        results.extend(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'per_video.json').write_text(json.dumps(results,indent=2)+'\n')
    (out/'run.log').write_text('host '+summary['host']+'\npostscore GT descriptive diagnostic; no prediction produced\nDIAGNOSTIC_DONE\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
