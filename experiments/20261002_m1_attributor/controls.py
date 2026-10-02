#!/usr/bin/env python3
"""Declared CPU falsification arms, runnable only after a qualifying full result."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from attributor import window_contributions

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


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--generate',action='store_true',required=True);a=ap.parse_args()
    parent=ROOT/'runs/20261002_m1_attributor';source=parent/'r1_main_fp32_mem';out=parent/'r1_controls'
    summary=json.load((parent/'r1_main_fp32_mem_analysis/summary.json').open())
    assert summary['gates']['any_qualifying_gain'],'declared funnel: no controls after no qualifying primary gain'
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

if __name__=='__main__':main()
