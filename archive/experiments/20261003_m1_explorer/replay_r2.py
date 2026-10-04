#!/usr/bin/env python3
"""Exact R2 branch replay from restored-cache R1 reads; no new model inference."""
import argparse
import copy
import json
import math
import socket
from pathlib import Path
import numpy as np
from analyze import ROOT,DATASETS,METRICS,read,metrics,boot,evaluate
from explorer import binary_entropy
from src.eval.evaluate import within_video_macro

PARENT=ROOT/'runs/20261003_m1_explorer'
OUT=PARENT/'r2_cache'
DECODED=PARENT/'r2_cache_decoded'
ANALYSIS=PARENT/'r2_cache_analysis'


def replay():
    source=PARENT/'r1_main';base=read(source/'base/predictions.jsonl');expanded=read(source/'explore/predictions.jsonl')
    manifest=read(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl')
    expected={k for k in manifest if k[0] in DATASETS};assert len(expected)==333 and base.keys()==expanded.keys()==expected
    cfg=json.loads((source/'config.json').read_text())
    assert cfg['entropy_thresholds']==[.1,.3] and cfg['frames_per_round']==2 and cfg['round_limit']==2 and cfg['GT_in_reader'] is False
    OUT.mkdir(parents=True,exist_ok=True)
    assert not any((OUT/a/'predictions.jsonl').exists() for a in ('base','explore'))
    config={'host':socket.gethostname(),'code':'experiments/20261003_m1_explorer/replay_r2.py;2026-10-03',
        'source':str(source.relative_to(ROOT)),'GT_in_replay':False,'actual_new_model_calls':0,
        'initial_entropy_threshold':.1,'support_override':False,'other_rules':'identical R1, independent restored context per window',
        'timing':'no measured R2 GPU time; calls describe deployment, not replay execution','selection':'development-selected R2, first revision'}
    (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    handles={};checks=[]
    for arm in ('base','explore'):
        dest=OUT/arm;dest.mkdir(exist_ok=True);(dest/'config.json').write_text(json.dumps({**config,'arm':arm},indent=2)+'\n')
        handles[arm]=(dest/'predictions.jsonl').open('w')
    for key,b in base.items():
        e=expanded[key];d=json.loads((source/'details'/key[0]/(key[1]+'.json')).read_text())
        assert b['extra']['z_video']==e['extra']['z_video'] and b['extra']['stance']==e['extra']['stance']
        assert len(b['extra']['windows'])==len(e['extra']['windows'])==len(d['traces'])
        result=copy.deepcopy(e);out_windows=[];kept=skipped=removed_windows=0
        for bw,ew,t in zip(b['extra']['windows'],e['extra']['windows'],d['traces']):
            assert bw['z_visual']==t['initial_z'] and ew['z_visual']==t['final_z']
            assert bw.get('z_speech')==ew.get('z_speech')
            h=binary_entropy(bw['z_visual']);assert abs(h-t['initial_entropy'])<1e-12
            retain=h>=.1
            if retain:kept+=len(t['rounds'])
            else:
                skipped+=len(t['rounds']);removed_windows+=bool(t['rounds'])
                if t['rounds']:assert t['nominal_support']==0
            w=copy.deepcopy(ew if retain else bw);out_windows.append(w)
        result['extra']['windows']=out_windows;result['method']='m1_explorer_r2_replay'
        duration=float(manifest[key]['duration']);assert duration==b['duration']==e['duration']
        idx=np.clip(((np.arange(math.ceil(4*duration))+.5)/4//8).astype(int),0,len(out_windows)-1)
        result['score_curve']=np.asarray([w['z'] for w in out_windows])[idx].tolist()
        assert len(result['score_curve'])==len(b['score_curve']) and np.isfinite(result['score_curve']).all()
        B=b['extra']['n_branches'];assert b['calls']==3+B and e['calls']==3+B+kept+skipped
        result['calls']=3+B+kept
        for arm,record in (('base',copy.deepcopy(b)),('explore',result)):
            record['code_path']=str(Path(__file__).relative_to(ROOT))
            record['extra'].pop('standalone_seconds',None)
            record['extra']['timing_status']='replayed model outputs; actual R2 GPU time not measured'
            record['extra']['source_prediction']=str((source/('base' if arm=='base' else 'explore')/'predictions.jsonl').relative_to(ROOT))
            handles[arm].write(json.dumps(record)+'\n')
        checks.append({'dataset':key[0],'video_id':key[1],'kept_acquisition_reads':kept,
            'skipped_acquisition_reads':skipped,'removed_windows':removed_windows,'deployment_forwards':result['calls']})
    for handle in handles.values():handle.close()
    (OUT/'replay_checks.json').write_text(json.dumps({'no_GT':True,'videos':len(checks),'actual_new_model_calls':0,'rows':checks},indent=2)+'\n')
    print('REPLAY_DONE',len(checks),flush=True)


def report():
    mm={a:metrics(DECODED/a/'metrics.json') for a in ('base','explore')}
    raw={a:metrics(OUT/a/'metrics.json') for a in mm};pred={a:read(DECODED/a/'predictions.jsonl') for a in mm}
    original=read(OUT/'base/predictions.jsonl');current=metrics(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json')
    for a in mm:
        assert pred[a].keys()==original.keys()
        for key,b in original.items():
            r=pred[a][key];assert r['native_rate']==b['native_rate']==4 and r['duration']==b['duration']
            assert len(r['score_curve'])==len(b['score_curve']) and np.isfinite(r['score_curve']).all()
            assert r['extra']['z_video']==b['extra']['z_video']
    result={'scope':'development-selected exact cache replay, not fresh GPU confirmation','revision':1,'mechanism_supported':False,'datasets':{}}
    checks=json.loads((OUT/'replay_checks.json').read_text())['rows']
    for ds in DATASETS:
        for metric in METRICS:assert mm['base'][ds][metric]==current[ds][metric]
        with np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True) as gt:
            truth={str(v):np.asarray(gt['y4'][i]) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        differences=[]
        for key in original:
            if key[0]!=ds:continue
            values=[within_video_macro({key[1]:truth[key[1]]},{key[1]:np.asarray(pred[a][key]['score_curve'])})[METRICS[-1]] for a in ('base','explore')]
            if values[0] is not None:assert values[1] is not None;differences.append(values[1]-values[0])
        rows=[r for r in checks if r['dataset']==ds]
        result['datasets'][ds]={'final':{a:{m:mm[a][ds][m] for m in METRICS} for a in mm},
            'raw':{a:{m:raw[a][ds][m] for m in METRICS} for a in mm},
            'delta':{m:mm['explore'][ds][m]-mm['base'][ds][m] for m in METRICS},
            'within_paired':boot(differences),
            'deployment_forwards_mean':float(np.mean([r['deployment_forwards'] for r in rows])),
            'removed_windows':sum(r['removed_windows'] for r in rows),'skipped_acquisition_reads':sum(r['skipped_acquisition_reads'] for r in rows)}
    result['gates']={'any_qualifying_gain':any(d['delta'][m]>=.01 for d in result['datasets'].values() for m in METRICS),
        'performance_pass':all(d['delta'][METRICS[-1]]>=.01 and all(d['delta'][m]>=(-.01 if m==METRICS[-1] else -.005) for m in METRICS) for d in result['datasets'].values())}
    ANALYSIS.mkdir(exist_ok=True);(ANALYSIS/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('ANALYSIS_DONE',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('replay','evaluate','report'),required=True)
    ap.add_argument('--arm',choices=('base','explore'));a=ap.parse_args();print('host',socket.gethostname(),flush=True)
    if a.stage=='replay':replay()
    elif a.stage=='evaluate':
        if not a.arm:ap.error('evaluate requires arm')
        evaluate(OUT,DECODED,a.arm)
    else:report()
