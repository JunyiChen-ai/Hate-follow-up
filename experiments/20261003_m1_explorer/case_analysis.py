#!/usr/bin/env python3
"""Post-scoring, development-selected case diagnostics; no method fitting."""
import argparse
import json
from pathlib import Path
import socket
import sys
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.eval.evaluate import within_video_macro
from analyze import boot,read
KEY='within_video_macro_ROC_AUC'


def score(y,s):
    return within_video_macro({'v':y},{'v':s})[KEY]


def main(version='r1'):
    parent=ROOT/'runs/20261003_m1_explorer';run=parent/(version+'_main')
    raw={a:read(run/a/'predictions.jsonl') for a in ('base','explore')}
    final={a:read(parent/(version+'_main_decoded')/a/'predictions.jsonl') for a in raw}
    assert all(v.keys()==raw['base'].keys() for v in [*raw.values(),*final.values()])
    report={'host':socket.gethostname(),'scope':'development-selected post-scoring diagnostics; subgroup associations are not causal tests',
        'GT_sources':[f'data/gt_4fps/{ds}.npz' for ds in ('HateMM','HateClipSeg')],
        'global_correct_definition':'native Yes/No compared with any positive frame in the canonical shared evaluation extent; HCS positives include offensive content',
        'per_dataset':{},'per_video':[]}
    for ds in ('HateMM','HateClipSeg'):
        with np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True) as gt:
            truth={str(v):np.asarray(gt['y4'][i]) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        rows=[]
        for k,b in raw['base'].items():
            if k[0]!=ds:continue
            e=raw['explore'][k];n=min(len(truth[k[1]]),len(b['score_curve']));y=truth[k[1]][:n]
            trace=json.loads((run/'details'/ds/(k[1]+'.json')).read_text())['traces']
            ws=b['extra']['windows'];t=(np.arange(n)+.5)/4
            idx=np.clip(np.searchsorted([w['start'] for w in ws],t,side='right')-1,0,len(ws)-1)
            masks={'all':np.ones(n,bool),
                'no_nominal_support':np.asarray([r['nominal_support']==0 for r in trace])[idx],
                'nominal_support':np.asarray([r['nominal_support']>0 for r in trace])[idx],
                'acquired':np.asarray([bool(r['rounds']) for r in trace])[idx],
                'not_acquired':np.asarray([not r['rounds'] for r in trace])[idx]}
            r={'dataset':ds,'video_id':k[1],'global_correct':bool((b['extra']['z_video']>0)==bool(y.any())),
                'n_windows':len(ws),'n_acquired_windows':sum(bool(x['rounds']) for x in trace),'groups':{},'windows':[]}
            for group,mask in masks.items():
                vals={}
                for arm in raw:
                    for name,curve in [('visual',np.asarray([w['z_visual'] for w in raw[arm][k]['extra']['windows']])[idx]),
                        ('raw_max',np.asarray(raw[arm][k]['score_curve'])[:n]),('final',np.asarray(final[arm][k]['score_curve'])[:n])]:
                        vals[arm+'_'+name]=score(y[mask],curve[mask]) if mask.any() else None
                r['groups'][group]=vals
            for i,(bw,ew,ct) in enumerate(zip(ws,e['extra']['windows'],trace)):
                m=idx==i
                r['windows'].append({'i':i,'start':bw['start'],'end':bw['end'],
                    'positive_fraction':float(y[m].mean()) if m.any() else None,
                    'nominal_support':ct['nominal_support'],'rounds':len(ct['rounds']),
                    'base_visual':bw['z_visual'],'explore_visual':ew['z_visual'],
                    'base_speech':bw.get('z_speech'),'base_max':bw['z'],'explore_max':ew['z']})
            rows.append(r)
        summaries={}
        for group in ('all','no_nominal_support','nominal_support','acquired','not_acquired'):
            summaries[group]={}
            for branch in ('visual','raw_max','final'):
                pairs=[(r['groups'][group]['base_'+branch],r['groups'][group]['explore_'+branch]) for r in rows]
                pairs=[(b,e) for b,e in pairs if b is not None and e is not None]
                summaries[group][branch]={'n':len(pairs),
                    'base_mean':float(np.mean([b for b,e in pairs])) if pairs else None,
                    'explore_mean':float(np.mean([e for b,e in pairs])) if pairs else None,
                    'paired_delta':boot([e-b for b,e in pairs]) if pairs else None}
        for correct in (True,False):
            subset=[r for r in rows if r['global_correct']==correct and r['groups']['all']['base_final'] is not None]
            summaries['global_correct_'+str(correct)]={branch:boot([r['groups']['all']['explore_'+branch]-r['groups']['all']['base_'+branch] for r in subset]) if subset else None for branch in ('visual','raw_max','final')}
        eligible=[r for r in rows if r['groups']['all']['base_final'] is not None]
        ranked=sorted(eligible,key=lambda r:r['groups']['all']['explore_final']-r['groups']['all']['base_final'])
        summaries['largest_final_losses']=[r['video_id'] for r in ranked[:5]]
        summaries['largest_final_gains']=[r['video_id'] for r in ranked[-5:][::-1]]
        report['per_dataset'][ds]=summaries;report['per_video']+=rows
    out=parent/(version+'_main_analysis')/'case_analysis.json';out.write_text(json.dumps(report,indent=2)+'\n')
    print('CASE_ANALYSIS_DONE',out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=('r1','r2','r3'),default='r1')
    main(parser.parse_args().version)
