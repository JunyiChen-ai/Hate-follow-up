#!/usr/bin/env python3
"""Audit control records, then use only the canonical evaluator and unchanged r6."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from analyze import ROOT,DATASETS,METRICS,read,metrics,boot
from src.eval.evaluate import within_video_macro


def prepare(arm,smoke):
    parent=ROOT/'runs/20261003_m1_explorer';main=parent/('r1_smoke' if smoke else 'r1_main')
    root=parent/('controls_smoke' if smoke else 'controls')/arm
    base=read(main/'base/predictions.jsonl');pred=read(root/'predictions.jsonl')
    assert base.keys()==pred.keys() and len(base)==(5 if smoke else 333)
    cfg=json.loads((root/'config.json').read_text())
    assert cfg['GT_in_reader'] is False and cfg['entropy_regating'] is False
    assert cfg['arm']==arm and cfg['smoke']==smoke and cfg['model']=='Qwen/Qwen3-VL-8B-Instruct'
    rows=[]
    for key,b in base.items():
        p=pred[key];ds,vid=key
        for name in ('z_video','stance','n_branches'):assert b['extra'][name]==p['extra'][name]
        assert b['duration']==p['duration'] and b['native_rate']==p['native_rate']==4
        assert len(b['score_curve'])==len(p['score_curve']) and np.isfinite(p['score_curve']).all()
        dt=json.loads((root/'details'/ds/(vid+'.json')).read_text());mt=json.loads((main/'details'/ds/(vid+'.json')).read_text())
        ws=p['extra']['windows'];assert len(ws)==len(b['extra']['windows'])==len(dt['traces'])==len(mt['traces'])
        sizes_match=tokens_match=True;matched_windows=0;image_encodes=0;rounds=0
        for i,(bw,w,t,m) in enumerate(zip(b['extra']['windows'],ws,dt['traces'],mt['traces'])):
            assert t['window']==i and t['base_visual']==bw['z_visual']
            for name in ('i','start','end','z_speech'):assert bw.get(name)==w.get(name)
            assert w['z']==max(w['z_visual'],w.get('z_speech',-np.inf))
            sizes=[len(r['added']) for r in t['rounds']];expected=[len(r['added']) for r in m['rounds']]
            sizes_match &= sizes==expected
            if arm!='fixed4':assert sizes==expected
            if smoke:assert dt['native_replay_exact'] is True
            assert w['z_visual']==(t['rounds'][-1]['z'] if t['rounds'] else bw['z_visual'])
            if arm=='mismatch':matched_windows+=bool(t['matched'])
            used=set()
            for k,r in enumerate(t['rounds']):
                assert 1<=len(r['added'])<=2 and len(t['rounds'])<=2
                for e in r['added']:
                    assert w['start']<=e['time']<w['end'] and e['index'] not in used
                    used.add(e['index'])
                assert len(r['acquired'])==len(used)<=4 and {e['index'] for e in r['acquired']}==used
                assert [e['time'] for e in r['acquired']]==sorted(e['time'] for e in r['acquired'])
                assert len(r['image_counts'])==len(r['image_grid_thw'])==len(r['paths'])==len(used)
                assert all((ROOT/path).is_file() for path in r['paths'])
                image_encodes+=len(r['image_counts']);rounds+=1
                if arm!='fixed4':tokens_match &= r['image_counts']==m['rounds'][k]['image_counts']
        assert p['calls']==3+b['extra']['n_branches']+rounds
        assert dt['actual_forwards']==p['calls']+(len(ws) if smoke else 0) and dt['acquisition_reads']==rounds
        rows.append({'dataset':ds,'video_id':vid,'sizes_match_main':sizes_match,'tokens_match_main':tokens_match if arm!='fixed4' else None,
            'image_encodes':image_encodes,'acquisition_reads':rounds,'matched_windows':matched_windows if arm=='mismatch' else None,
            'total_windows':len(ws),'peak_GiB':dt['peak_GiB']})
    result={'no_GT':True,'arm':arm,'videos':len(rows),'all_sizes_match_main':all(r['sizes_match_main'] for r in rows),
        'all_tokens_match_main':all(r['tokens_match_main'] for r in rows) if arm!='fixed4' else None,'rows':rows}
    (root/'alignment.json').write_text(json.dumps(result,indent=2)+'\n');print('CONTROL_PREPARED',arm,len(rows))


def evaluate(arm):
    parent=ROOT/'runs/20261003_m1_explorer';root=parent/'controls'/arm
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(root/'predictions.jsonl'),
        '--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(root/'metrics.json')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(root),
        '--datasets',*DATASETS,'--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length',
        '--min-windows','2','--bma-grid','6','--arm','m2','--out-root',str(parent/'controls_decoded'),'--tag',arm],cwd=ROOT,check=True)


def report(arm):
    parent=ROOT/'runs/20261003_m1_explorer';root=parent/'controls'/arm
    mp=read(parent/'r1_main_decoded/explore/predictions.jsonl');cp=read(parent/'controls_decoded'/arm/'predictions.jsonl')
    mm=metrics(parent/'r1_main_decoded/explore/metrics.json');cm=metrics(parent/'controls_decoded'/arm/'metrics.json')
    assert mp.keys()==cp.keys()
    result={'scope':'development-selected complete-corpus control, r6 independently refit; no automatic causal conclusion',
        'arm':arm,'datasets':{},'metric_source':str((parent/'controls_decoded'/arm/'metrics.json').relative_to(ROOT))}
    for ds in DATASETS:
        with np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True) as gt:
            ys={str(v):np.asarray(gt['y4'][i]) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        diffs=[]
        for key,p in mp.items():
            if key[0]!=ds:continue
            values=[within_video_macro({key[1]:ys[key[1]]},{key[1]:np.asarray(r[key]['score_curve'])})[METRICS[-1]] for r in (mp,cp)]
            if values[0] is not None:assert values[1] is not None;diffs.append(values[0]-values[1])
        result['datasets'][ds]={'control':{m:cm[ds][m] for m in METRICS},
            'main_minus_control':{m:mm[ds][m]-cm[ds][m] for m in METRICS},'within_paired':boot(diffs)}
    result['common_metrics_with_removal_drop_at_least_01']=[m for m in METRICS if all(result['datasets'][ds]['main_minus_control'][m]>=.01 for ds in DATASETS)]
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('CONTROL_ANALYSIS_DONE')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True)
    ap.add_argument('--arm',choices=('uniform','distance','fixed4','mismatch'),required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.smoke and a.stage!='prepare':ap.error('smoke is no-GT plumbing only')
    if a.stage=='prepare':prepare(a.arm,a.smoke)
    elif a.stage=='evaluate':evaluate(a.arm)
    else:report(a.arm)
