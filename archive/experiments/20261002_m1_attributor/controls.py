#!/usr/bin/env python3
"""Declared CPU falsification arms, runnable only after a qualifying full result."""
import argparse
import copy
import json
import socket
import subprocess
from pathlib import Path
import sys
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from attributor import window_contributions
from analyze import boot, metrics, METRICS, DATASETS
from src.eval.evaluate import within_video_macro

ARMS=('endpoint','absolute','rotated','density','shift')


def read(path):
    out={}
    for r in map(json.loads,path.open()):
        k=r['dataset'],r['video_id'];assert k not in out and not r.get('error');out[k]=r
    return out


def transform(arm,token,base,attribute):
    windows=base['extra']['windows'];n=len(windows)
    if arm=='shift':
        offset=np.mean([w['z'] for w in attribute['extra']['windows']])-np.mean([w['z'] for w in windows])
        values=[{m:w[m]+offset for m in ('z_visual','z_speech') if m in w} for w in windows]
    else:
        attr=np.array(token['attribution'],copy=True)
        if arm=='endpoint':attr=np.array(token['endpoint'],copy=True)
        elif arm=='absolute':attr=np.abs(attr)
        elif arm=='density':attr=token['media'].astype(float)
        elif arm=='rotated':
            for kind in ('visual','speech'):
                ids=np.flatnonzero(token[kind]);attr[ids]=np.roll(attr[ids],len(ids)//2)
        else:raise ValueError(arm)
        regions={'local':[x for x in token['local']], 'visual':token['visual'],'speech':token['speech']}
        # Only availability matters here; original modality support is unchanged.
        texts=['present' if 'z_speech' in w else '' for w in windows]
        values,_,_=window_contributions(attr,regions,texts)
    r=copy.deepcopy(attribute);r['method']='m1_attributor_control_'+arm
    z=np.array([max(w.values()) for w in values]);assert np.isfinite(z).all()
    L=len(r['score_curve']);idx=np.clip(((np.arange(L)+.5)/4//8).astype(int),0,n-1)
    r['score_curve']=z[idx].tolist()
    r['extra']['windows']=[{'i':i,'start':w['start'],'end':w['end'],'z':float(z[i]),**values[i]} for i,w in enumerate(windows)]
    r['extra']['diagnostic_only']=True
    r['extra']['cost_note']='reuses completed attribution run; calls/time fields describe source extraction, not incremental CPU cost'
    return r


def assert_primary_gain(parent):
    summary=json.load((parent/'r1_main_fp32_mem_analysis/summary.json').open())
    assert summary['gates']['any_qualifying_gain'],'declared funnel: no controls after no qualifying primary gain'


def generate(source,out):
    base=read(source/'base/predictions.jsonl');attribute=read(source/'attribute/predictions.jsonl')
    assert base.keys()==attribute.keys() and len(base)==333
    handles={}
    for arm in ARMS:
        d=out/arm;d.mkdir(parents=True,exist_ok=True)
        (d/'config.json').write_text(json.dumps({'arm':arm,'source':str(source.relative_to(ROOT)),
          'code':'experiments/20261002_m1_attributor/controls.py, sources2026-10-03',
          'scope':'development-selected diagnostic; no GT in transformation','incremental_model_calls':0},indent=2)+'\n')
        handles[arm]=(d/'predictions.jsonl').open('w')
    for key,b in base.items():
        token=np.load(source/'tokens'/key[0]/(key[1]+'.npz'))
        for arm,h in handles.items():h.write(json.dumps(transform(arm,token,b,attribute[key]))+'\n')
    for h in handles.values():h.close()
    print('CONTROLS_GENERATED',len(base),list(ARMS))


def evaluate(out,decoded,arm):
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets',
      '--predictions',str(out/arm/'predictions.jsonl'),'--gt-dir','data/gt_4fps',
      '--datasets',*DATASETS,'--out',str(out/arm/'metrics.json')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py',
      '--run',str(out/arm),'--datasets',*DATASETS,'--noleak','--transform','nscore',
      '--key','calib','--duration','bma','--bma-prior','length','--min-windows','2',
      '--bma-grid','6','--arm','m2','--out-root',str(decoded),'--tag',arm],cwd=ROOT,check=True)


def report(parent,source,out,decoded):
    paths={a:out/a for a in ARMS}
    paths.update({a:source/a for a in ('base','attribute')})
    dpaths={a:decoded/a for a in ARMS}
    dpaths.update({a:parent/'r1_main_fp32_mem_decoded'/a for a in ('base','attribute')})
    raw={a:read(p/'predictions.jsonl') for a,p in paths.items()}
    pred={a:read(p/'predictions.jsonl') for a,p in dpaths.items()}
    mm={a:metrics(p/'metrics.json') for a,p in dpaths.items()}
    rm={a:metrics(p/'metrics.json') for a,p in paths.items()}
    expected=raw['base'].keys();assert len(expected)==333
    for arm in paths:
        assert raw[arm].keys()==pred[arm].keys()==expected
        for key,b in raw['base'].items():
            for r in (raw[arm][key],pred[arm][key]):
                assert r['native_rate']==4 and np.isfinite(r['score_curve']).all()
                assert len(r['score_curve'])==len(b['score_curve'])
                assert r['extra']['z_video']==b['extra']['z_video']
            assert [('z_speech' in w) for w in raw[arm][key]['extra']['windows']]==[('z_speech' in w) for w in b['extra']['windows']]
    result={'scope':'development-selected falsification controls; no GT used to generate scores',
      'incremental_model_calls':0,'metric_sources':{a:str((p/'metrics.json').relative_to(ROOT)) for a,p in dpaths.items()},
      'mechanism_supported':False,'remaining_checks':['FP32 native window control','declared cached-value deletion interventions'],
      'datasets':{}}
    lines=['dataset\tarm\tROC\tPR\twithin\traw_within\n'];allrows=[]
    for ds in DATASETS:
        g=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True)
        rows=[]
        for i,vid in enumerate(g['video_ids']):
            if str(g['split'][i])!='test':continue
            key=ds,str(vid);assert key in expected
            y=np.asarray(g['y4'][i]);r={'dataset':ds,'video_id':str(vid)}
            for arm in paths:
                r[arm]=within_video_macro({key[1]:y},{key[1]:np.asarray(pred[arm][key]['score_curve'])})[METRICS[-1]]
                r[arm+'_raw']=within_video_macro({key[1]:y},{key[1]:np.asarray(raw[arm][key]['score_curve'])})[METRICS[-1]]
            if r['base'] is not None:
                assert abs(r['shift_raw']-r['base_raw'])<1e-12,'shift altered raw temporal ordering'
                rows.append(r)
        allrows.extend(rows)
        result['datasets'][ds]={}
        for arm in paths:
            result['datasets'][ds][arm]={
              'final':{m:mm[arm][ds][m] for m in METRICS},'raw':{m:rm[arm][ds][m] for m in METRICS},
              'attribute_minus_control':{m:mm['attribute'][ds][m]-mm[arm][ds][m] for m in METRICS},
              'within_attribute_minus_control':boot([r['attribute']-r[arm] for r in rows]),
              'within_control_minus_base':boot([r[arm]-r['base'] for r in rows]),
              'raw_within_control_minus_base':boot([r[arm+'_raw']-r['base_raw'] for r in rows])}
            lines.append(ds+'\t'+arm+'\t'+'\t'.join(f'{mm[arm][ds][m]:.6f}' for m in METRICS)+f'\t{rm[arm][ds][METRICS[-1]]:.6f}\n')
    report_dir=parent/'r1_controls_analysis';report_dir.mkdir(exist_ok=True)
    (report_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (report_dir/'per_video.json').write_text(json.dumps(allrows,indent=2)+'\n')
    (report_dir/'table.tsv').write_text(''.join(lines));print(''.join(lines));print('CONTROLS_ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('generate','evaluate','report'),required=True)
    ap.add_argument('--arm',choices=ARMS);a=ap.parse_args()
    if a.stage=='evaluate' and not a.arm:ap.error('evaluate requires arm')
    parent=ROOT/'runs/20261002_m1_attributor';source=parent/'r1_main_fp32_mem';out=parent/'r1_controls'
    decoded=parent/'r1_controls_decoded'
    print('host',socket.gethostname(),flush=True);assert_primary_gain(parent)
    if a.stage=='generate':generate(source,out)
    elif a.stage=='evaluate':evaluate(out,decoded,a.arm)
    else:report(parent,source,out,decoded)

if __name__=='__main__':main()
