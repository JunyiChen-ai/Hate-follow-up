#!/usr/bin/env python3
"""Audit control records, then use only the canonical evaluator and unchanged r6."""
import argparse
from collections import Counter
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
        groups=Counter(sum(len(r['added']) for r in t['rounds']) for t in mt['traces'])
        for i,(bw,w,t,m) in enumerate(zip(b['extra']['windows'],ws,dt['traces'],mt['traces'])):
            assert t['window']==i and t['base_visual']==bw['z_visual']
            for name in ('i','start','end','z_speech'):assert bw.get(name)==w.get(name)
            assert w['z']==max(w['z_visual'],w.get('z_speech',-np.inf))
            sizes=[len(r['added']) for r in t['rounds']];expected=[len(r['added']) for r in m['rounds']]
            sizes_match &= sizes==expected
            if arm!='fixed4':assert sizes==expected
            if smoke:assert dt['native_replay_exact'] is True
            assert w['z_visual']==(t['rounds'][-1]['z'] if t['rounds'] else bw['z_visual'])
            if arm=='mismatch':
                count=sum(expected);is_matched=count>0 and groups[count]>=2
                assert t['matched']==is_matched
                matched_windows+=is_matched
                if not is_matched:assert w['z_visual']==m['final_z']
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
    raw_main=read(parent/'r1_main/explore/predictions.jsonl');raw_control=read(root/'predictions.jsonl')
    assert mp.keys()==cp.keys()==raw_main.keys()==raw_control.keys()
    for key in mp:
        original=raw_main[key]
        for source in (mp,cp,raw_control):
            p=source[key]
            assert p['native_rate']==original['native_rate']==4
            assert p['duration']==original['duration']
            assert len(p['score_curve'])==len(original['score_curve']) and np.isfinite(p['score_curve']).all()
            assert p['extra']['z_video']==original['extra']['z_video']
            if 'stance' in p['extra']:assert p['extra']['stance']==original['extra']['stance']
    result={'scope':'development-selected complete-corpus control, r6 independently refit; no automatic causal conclusion',
        'arm':arm,'datasets':{},'metric_source':str((parent/'controls_decoded'/arm/'metrics.json').relative_to(ROOT))}
    for ds in DATASETS:
        with np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True) as gt:
            ys={str(v):np.asarray(gt['y4'][i]) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        diffs=[];group_values={name:{branch:[] for branch in ('visual','raw_max','final')} for name in (
            'matched_windows','unmatched_windows','videos_with_matched_windows','videos_without_matched_windows')}
        group_counts=Counter()
        for key,p in mp.items():
            if key[0]!=ds:continue
            values=[within_video_macro({key[1]:ys[key[1]]},{key[1]:np.asarray(r[key]['score_curve'])})[METRICS[-1]] for r in (mp,cp)]
            if values[0] is not None:assert values[1] is not None;diffs.append(values[0]-values[1])
            if arm=='mismatch':
                details=json.loads((root/'details'/ds/(key[1]+'.json')).read_text())
                flags=np.asarray([bool(t['matched']) for t in details['traces']]);windows=raw_main[key]['extra']['windows']
                assert len(flags)==len(windows)
                group_counts['matched_windows']+=int(flags.sum());group_counts['unmatched_windows']+=int((~flags).sum())
                video_group='videos_with_matched_windows' if flags.any() else 'videos_without_matched_windows'
                group_counts[video_group]+=1
                n=min(len(ys[key[1]]),len(p['score_curve']));y=ys[key[1]][:n]
                index=np.clip(np.searchsorted([w['start'] for w in windows],(np.arange(n)+.5)/4,side='right')-1,0,len(windows)-1)
                masks={'matched_windows':flags[index],'unmatched_windows':~flags[index],video_group:np.ones(n,bool)}
                for branch in ('visual','raw_max','final'):
                    if branch=='visual':
                        curves=[np.asarray([w['z_visual'] for w in source[key]['extra']['windows']])[index] for source in (raw_main,raw_control)]
                    else:curves=[np.asarray(source[key]['score_curve'])[:n] for source in ((raw_main,raw_control) if branch=='raw_max' else (mp,cp))]
                    for group,mask in masks.items():
                        v=[within_video_macro({'v':y[mask]},{'v':curve[mask]})[METRICS[-1]] for curve in curves]
                        if v[0] is not None:assert v[1] is not None;group_values[group][branch].append((v[0],v[1]))
        result['datasets'][ds]={'control':{m:cm[ds][m] for m in METRICS},
            'main_minus_control':{m:mm[ds][m]-cm[ds][m] for m in METRICS},'within_paired':boot(diffs),
            'standalone_seconds':sum(p['extra']['standalone_seconds'] for k,p in raw_control.items() if k[0]==ds),
            'mean_forwards':float(np.mean([p['calls'] for k,p in raw_control.items() if k[0]==ds]))}
        if arm=='mismatch':
            result['datasets'][ds]['mismatch_counts']={name:group_counts[name] for name in group_values}
            result['datasets'][ds]['mismatch_subgroups']={group:{branch:{'eligible_videos':len(pairs),
                'main_mean':float(np.mean([m for m,c in pairs])) if pairs else None,
                'control_mean':float(np.mean([c for m,c in pairs])) if pairs else None,
                'main_minus_control':boot([m-c for m,c in pairs]) if pairs else None}
                for branch,pairs in values.items()} for group,values in group_values.items()}
            result['datasets'][ds]['subgroup_note']='Window subsets recompute canonical within on their masked frames; video groups use whole curves. Descriptive, not replacement primary metrics. Unchanged raw windows can still change after whole-corpus r6 refit.'
    result['common_metrics_with_removal_drop_at_least_01']=[m for m in METRICS if all(result['datasets'][ds]['main_minus_control'][m]>=.01 for ds in DATASETS)]
    (root/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('CONTROL_ANALYSIS_DONE')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True)
    ap.add_argument('--arm',choices=('uniform','distance','fixed4','mismatch'),required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.smoke and a.stage!='prepare':ap.error('smoke is no-GT plumbing only')
    if a.stage=='prepare':prepare(a.arm,a.smoke)
    elif a.stage=='evaluate':evaluate(a.arm)
    else:report(a.arm)
