#!/usr/bin/env python3
"""Declared R3 branch ablation; cached exact reads, no GT in composition."""
import copy
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import numpy as np
from analyze import evaluate,read,metrics,prepare,METRICS,DATASETS
from src.video_inputs import fixed_windows
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())

def compose(base,visual,arm):
    assert base['dataset']==visual['dataset'] and base['video_id']==visual['video_id']
    assert base['duration']==visual['duration'] and base['native_rate']==visual['native_rate']==4
    assert base['extra']['z_video']==visual['extra']['z_video'] and base['extra']['stance']==visual['extra']['stance']
    bw=base['extra']['windows'];vw=visual['extra']['windows'];assert len(bw)==len(vw)
    duration=base['duration'];assert duration>0
    expected_windows=fixed_windows(duration,8);assert len(bw)==len(expected_windows)
    rec=copy.deepcopy(visual);ww=[]
    for i,(b,v) in enumerate(zip(bw,vw)):
        assert (b['i'],b['start'],b['end'])==(v['i'],v['start'],v['end'])
        assert (b['i'],b['start'],b['end'])==(i,*expected_windows[i])
        assert ('z_speech' in b)==('z_speech' in v)
        w=copy.deepcopy(v)
        if 'z_speech' in b:w['z_speech']=b['z_speech']
        w['z']=max(w['z_visual'],w.get('z_speech',-np.inf));ww.append(w)
    rec['extra']['windows']=ww
    length=len(base['score_curve']);assert length==len(visual['score_curve'])==math.ceil(4*duration)
    times=(np.arange(length)+.5)/4
    # Same fixed 8s window mapping; the final cell midpoint may extend beyond duration.
    index=np.clip((times//8).astype(int),0,len(ww)-1)
    rec['score_curve']=np.asarray([w['z'] for w in ww])[index].tolist()
    assert np.isfinite(rec['score_curve']).all()
    B=sum(1+('z_speech' in w) for w in ww)
    rec['calls']=3+B if arm=='base' else 6+B
    rec['extra']['timing']='cached branch composition; standalone revised schedule not measured'
    for k in ('prefix_seconds','branch_seconds','prefix_seconds_note','branch_seconds_note'):rec['extra'].pop(k,None)
    rec['method']='m1_integrator_r3_'+arm;rec['code_path']=str(Path(__file__).relative_to(ROOT))
    if arm=='base':assert rec['score_curve']==base['score_curve']
    return rec


def validate_source(up,out):
    cfg=json.load((up/'config.json').open())
    expected={'run_name':'r2_main','smoke':False,'future_keys':'text','seed':0,
        'model':'Qwen/Qwen3-VL-8B-Instruct','GT_in_reader':False,'frames':20,'window_seconds':8,'fps':4,
        'prefix_mask':'j<=i OR (visual(i) AND NOT visual(j)); all language layers',
        'queries':'ordinary full causal access','global_key':'native original','forced_answer':'native original'}
    for key,value in expected.items():assert cfg[key]==value,(key,cfg[key])
    for arm in ('base','causal','future'):
        ac=json.load((up/arm/'config.json').open());assert ac=={**cfg,'arm':arm}
    # Reviewed R2 validation checks all333 IDs, original native reads, each token
    # mapping/mask mode/edge count, actual upstream forwards, and finite curves.
    prepare(up,out)


def main():
    parent=ROOT/'runs/20261003_m1_integrator';up=parent/'r2_main';out=parent/'r3_main'
    out.mkdir(parents=True,exist_ok=True);(out/'run.pid').write_text(str(os.getpid()))
    print('host',socket.gethostname(),flush=True)
    validate_source(up,out)
    raw={a:read(up/a/'predictions.jsonl') for a in ('base','causal','future')}
    manifest=read(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl')
    keys={k for k in manifest if k[0] in DATASETS};assert len(keys)==333 and all(v.keys()==keys for v in raw.values())
    config={'date':time.strftime('%Y-%m-%d'),'host':socket.gethostname(),'code':str(Path(__file__).relative_to(ROOT)),
        'source':str(up.relative_to(ROOT)),'GT_in_composition':False,'rule':'R2 visual + original native speech; native global/answer',
        'runtime':'cached only; new-video schedule must be measured before promotion','frames':20,'window_seconds':8,'fps':4}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    for a in raw:
        dest=out/a;dest.mkdir(exist_ok=True);(dest/'config.json').write_text(json.dumps({**config,'arm':a},indent=2)+'\n')
        with (dest/'predictions.jsonl').open('w') as f:
            for k,b in raw['base'].items():f.write(json.dumps(compose(b,raw[a][k],a))+'\n')
    print('COMPOSED_COMPLETE333',flush=True)
    decoded=parent/'r3_main_decoded'
    for a in raw:evaluate(out,decoded,a)
    lines=['dataset\tarm\tROC\tPR\twithin\n']
    for a in raw:
        for ds,m in metrics(decoded/a/'metrics.json').items():
            lines.append(ds+'\t'+a+'\t'+'\t'.join(f'{m[k]:.9f}' for k in METRICS)+'\n')
    (out/'table.tsv').write_text(''.join(lines));print(''.join(lines));print('ANALYSIS_DONE',flush=True)

if __name__=='__main__':main()
