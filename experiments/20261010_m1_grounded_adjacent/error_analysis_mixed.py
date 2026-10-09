"""Candidate 41 error analysis, part 2 (rule 10, development; 2026-10-10): restrict the direction × probe × label
table to the videos the within metric uses (both labels present), count helpfulness by the visual direction (the
r6 decoder reads z_visual and z_speech as separate conditions), and compare decoded per-video within AUC between
accept_all, grounded, random and the two oracles. Files read: main records, data/gt_4fps, decoded predictions of
error_analysis/, r6_bma and the 2026-10-10 selection_analysis oracle."""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parent))
from error_analysis import load,DATASETS,EXP,ROOT,OUT


def mixed_ids():
    out={}
    for ds in DATASETS:
        z=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True);out[ds]={v for v,y in zip(z['video_ids'],z['y4']) if 0<y.sum()<len(y)}
    return out


def table_mixed(videos,mixed):
    T={}
    for ds in DATASETS:
        agg={}
        for vid in videos:
            if vid['ds']!=ds or vid['base']['video_id'] not in mixed[ds]:continue
            for w in vid['win']:
                if not w['covered']:continue
                k=('up' if w['up'] else 'down')+' '+w['probe'];a=agg.setdefault(k,dict(n=0,pos=0,help_visual=0,abs_dz=0.,abs_dz_harm=0.,abs_dz_help=0.))
                hv=(w['dz']>0)==bool(w['label']);a['n']+=1;a['pos']+=w['label'];a['help_visual']+=hv;a['abs_dz']+=abs(w['dz']);a['abs_dz_help' if hv else 'abs_dz_harm']+=abs(w['dz'])
        T[ds]={k:dict(n=a['n'],positive_share=a['pos']/a['n'],help_rate_visual=a['help_visual']/a['n'],sum_abs_dz_helpful=round(a['abs_dz_help']),sum_abs_dz_harmful=round(a['abs_dz_harm'])) for k,a in sorted(agg.items())}
    return T


def per_video(path,gt,mixed):
    out={ds:{} for ds in DATASETS}
    for line in Path(path).open():
        r=json.loads(line);ds=r['dataset'];v=r['video_id']
        if ds not in out or v not in mixed[ds]:continue
        y=gt[ds][v];s=np.asarray(r['score_curve'],dtype=float);n=min(len(y),len(s))
        out[ds][v]=float(roc_auc_score(y[:n],s[:n]))
    return out


def main():
    videos=load();mixed=mixed_ids();gt={}
    for ds in DATASETS:
        z=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True);gt[ds]={v:y for v,y in zip(z['video_ids'],z['y4'])}
    T=table_mixed(videos,mixed)
    runs={'r6':ROOT/'runs/20260926_twolevel/r6_bma/predictions.jsonl','oracle_visual_2026-10-10':ROOT/'runs/20261007_m1_streamingtom/selection_analysis/decoded/adjacent_native__oracle_help/predictions.jsonl'}
    for n in ('accept_all','grounded','random_down_rate_0','random_down_rate_1','down_only','up_only','oracle_help'):runs[n]=OUT/'decoded'/n/'predictions.jsonl'
    PV={n:per_video(p,gt,mixed) for n,p in runs.items()}
    comp={}
    for ds in DATASETS:
        ids=sorted(PV['accept_all'][ds]);comp[ds]=dict(n_videos=len(ids),mean={n:float(np.mean([PV[n][ds][v] for v in ids])) for n in PV})
        for a,b in (('grounded','accept_all'),('accept_all','r6'),('grounded','r6'),('oracle_help','accept_all'),('oracle_visual_2026-10-10','accept_all')):
            d=np.array([PV[a][ds][v]-PV[b][ds][v] for v in ids])
            comp[ds][f'{a}_minus_{b}']=dict(mean=float(d.mean()),improved=int((d>.001).sum()),worsened=int((d<-.001).sum()),unchanged=int((abs(d)<=.001).sum()),
                largest=[(ids[i],round(float(d[i]),4)) for i in np.argsort(-abs(d))[:6]])
    res=dict(scope='development-selected test error analysis (rule 10); within-eligible videos only',mixed_videos={ds:len(mixed[ds]) for ds in DATASETS},table_mixed=T,per_video_within=comp)
    (OUT/'within_eligible.json').write_text(json.dumps(res,indent=2)+'\n')
    for ds in DATASETS:
        print(ds,'mixed videos',len(mixed[ds]))
        for k,v in T[ds].items():print(f"  {k:12s} n={v['n']:5d} pos={v['positive_share']:.2f} help_visual={v['help_rate_visual']:.3f} |dz| helpful={v['sum_abs_dz_helpful']:5d} harmful={v['sum_abs_dz_harmful']:5d}")
        c=comp[ds];print('  mean within:',' '.join(f"{n}={x:.4f}" for n,x in c['mean'].items()))
        for k,v in c.items():
            if k.endswith(')') or k in ('n_videos','mean'):continue
            print(f"  {k}: mean {v['mean']:+.4f} improved {v['improved']} worsened {v['worsened']} unchanged {v['unchanged']} largest {v['largest'][:4]}")
    print('C41_ERROR_ANALYSIS_MIXED_DONE')


if __name__=='__main__':main()
