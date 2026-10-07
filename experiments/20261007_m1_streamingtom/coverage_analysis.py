"""Error analysis (development, test GT only through the sole evaluator): where does control #1 gain over r6?

Windows are split by r6's own input layout, never by labels: "empty" if none of the 20 shared prefix frames
(data/frames_k20 times, as read by r6) falls inside the window [start, end), "covered" otherwise. Two diagnostic
prediction sets take control #1's visual read only in one kind of window and r6's elsewhere; speech, G and stance are
r6's. Both go through the sole evaluator and the fixed r6 decoder. Diagnostics, not methods.
"""
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.video_inputs import frame_paths

EXP=ROOT/'runs/20261007_m1_streamingtom'
OUT=EXP/'coverage_analysis'
DATASETS=('HateMM','HateClipSeg')
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')


def records(path):
    return {(r['dataset'],r['video_id']):r for r in map(json.loads,path.open())}


def hybrid(base,new,take,name):
    ww=[]
    for b,n,t in zip(base['extra']['windows'],new['extra']['windows'],take):
        assert (b['i'],b['start'],b['end'],b.get('z_speech'))==(n['i'],n['start'],n['end'],n.get('z_speech'))
        v=n['z_visual'] if t else b['z_visual'];w={**b,'z_visual':v,'z':max(v,b['z_speech']) if 'z_speech' in b else v};ww.append(w)
    idx=np.clip(((np.arange(math.ceil(base['duration']*4))+.5)/4//8).astype(int),0,len(ww)-1)
    return {**base,'method':name,'score_curve':np.asarray([w['z'] for w in ww])[idx].tolist(),'extra':{**base['extra'],'windows':ww}}


def metric(path):
    return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    r6=records(EXP/'controls_dualpath_main/base/predictions.jsonl');c1=records(EXP/'controls_dualpath_main/no_remote/predictions.jsonl')
    glr=records(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');assert set(r6)==set(c1)==set(glr)
    arms={'swap_empty':[],'swap_covered':[]};stats={ds:dict(videos=0,windows=0,empty=0,videos_with_empty=0,abs_change_empty=[],abs_change_covered=[]) for ds in DATASETS}
    for key in sorted(r6):
        base,new=r6[key],c1[key];assert base['score_curve']==glr[key]['score_curve'] and base['extra']['windows']==glr[key]['extra']['windows']
        times=[t for t,_ in frame_paths(key[0],key[1],20)];assert times
        empty=[not any(w['start']<=t<w['end'] for t in times) for w in base['extra']['windows']]
        arms['swap_empty'].append(hybrid(base,new,empty,'diag_swap_empty'));arms['swap_covered'].append(hybrid(base,new,[not e for e in empty],'diag_swap_covered'))
        s=stats[key[0]];s['videos']+=1;s['windows']+=len(empty);s['empty']+=sum(empty);s['videos_with_empty']+=any(empty)
        for e,b,n in zip(empty,base['extra']['windows'],new['extra']['windows']):s['abs_change_empty' if e else 'abs_change_covered'].append(abs(n['z_visual']-b['z_visual']))
    for s in stats.values():
        for k in ('abs_change_empty','abs_change_covered'):s[k]=float(np.mean(s[k]))
    for name,rows in arms.items():
        folder=OUT/name;folder.mkdir(exist_ok=True)
        (folder/'config.json').write_text(json.dumps(dict(diagnostic=name,GT_read_in_construction=False,source=__file__),indent=2)+'\n')
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(folder/'predictions.jsonl'),
            '--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(folder/'metrics.json')],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(folder),'--datasets',*DATASETS,
            '--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length','--min-windows','2','--bma-grid','6','--arm','m2',
            '--out-root',str(OUT/'decoded'),'--tag',name],cwd=ROOT,check=True)
    m={'r6':metric(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json'),'control1':metric(EXP/'controls_dualpath_main_decoded/no_remote/metrics.json'),
        **{a:metric(OUT/'decoded'/a/'metrics.json') for a in arms}}
    result=dict(scope='development-selected error analysis; split from r6 frame layout only; test GT only inside the sole evaluator',stats=stats,
        metrics={a:{ds:{k:m[a][ds][k] for k in METRICS} for ds in DATASETS} for a in m},
        gain_vs_r6={a:{ds:{k:m[a][ds][k]-m['r6'][ds][k] for k in METRICS} for ds in DATASETS} for a in ('swap_empty','swap_covered','control1')})
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('COVERAGE_ANALYSIS_DONE')


if __name__=='__main__':main()
