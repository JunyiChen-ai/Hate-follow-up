#!/usr/bin/env python3
"""R2 source validation and shared canonical/fixed-r6 orchestration."""
import argparse
import json
import numpy as np
from extract import ROOT,CACHE,DATASETS,selected_rows
from path_measure import validate_bundle
from path_graph import VERSION
from analyze import read,evaluate,report
from src.mllm_renderer import cpu_renderer
from src.video_inputs import load_asr


def prepare(root,out,smoke):
    rows=selected_rows(smoke);expected={(r['dataset'],r['video_id']):r for r in rows}
    cfg=json.loads((root/'config.json').read_text());assert cfg['GT_in_reader'] is False and cfg['smoke']==smoke and cfg['reader_version']==VERSION
    predictions={a:read(root/a/'predictions.jsonl') for a in ('base','optimized')}
    assert all(r.keys()==expected.keys() for r in predictions.values())
    old=read(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');asr={ds:load_asr(ds) for ds in DATASETS}
    renderer=cpu_renderer();bundles={};changed=verified=0
    for key,row in expected.items():
        metadata=json.loads((CACHE/key[0]/(key[1]+'.json')).read_text())
        b=json.loads((root/'records'/key[0]/(key[1]+'.json')).read_text())
        validate_bundle(row,b,metadata,asr[key[0]].get(key[1],[]),renderer,smoke)
        assert all(b[a]==predictions[a][key] for a in predictions)
        assert b['base']['extra']['z_video']==old[key]['extra']['z_video']
        assert b['base']['extra']['windows']==old[key]['extra']['windows'] and np.array_equal(b['base']['score_curve'],old[key]['score_curve'])
        changed+=sum(a.get('z_speech')!=c.get('z_speech') for a,c in zip(b['base']['extra']['windows'],b['optimized']['extra']['windows']))
        verified+=sum('smoke_checks' in t for t in b['traces']);bundles[key]=b
    if smoke:assert verified==sum(any(t['available'] for t in b['traces']) for b in bundles.values())
    result=dict(coverage=len(rows),GT_read=False,native_exact=True,global_and_visual_exact=True,
        changed_speech_windows=changed,clone_and_unit_checks=verified,mechanism_supported=False,cost={})
    for ds in DATASETS:
        bb=[b for k,b in bundles.items() if k[0]==ds]
        cc=[b['checks'] for b in bb];tt=[t for b in bb for t in b['traces'] if t['available']]
        result['cost'][ds]=dict(standalone_seconds={a:sum(b[a]['extra']['standalone_seconds'] for b in bb) for a in predictions},
            peak_GiB=max(c['peak_GiB'] for c in cc),
            **{k:sum(c[k] for c in cc) for k in ('prefix_seconds','visual_seconds','reference_speech_seconds','new_speech_seconds',
                'preprocessing_seconds','actual_forwards','diagnostic_forwards','diagnostic_seconds')},
            input_actual_forwards={k:sum(c['input_actual_forwards'][k] for c in cc) for k in ('encoder','decoder')},
            recognized_windows=len(tt),graph_tokens=sum(len(t['graph']['ids']) for t in tt),
            complete_paths=sum(len(t['graph']['paths']) for t in tt),
            epsilon_paths=sum(not p['text'] for t in tt for p in t['graph']['paths']),
            zero_mass_paths=sum(p['mass']==0 for t in tt for p in t['graph']['paths']))
        if smoke:
            sample=[b for b in bb if b['base']['video_id']!='hate_video_114']
            result['cost'][ds]['rough_full_seconds']=(215 if ds=='HateMM' else 118)*np.mean([b['optimized']['extra']['standalone_seconds'] for b in sample])
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--name',choices=('base','optimized'));a=ap.parse_args()
    stem='r2_full_'+('smoke' if a.smoke else 'main');root=ROOT/'runs/20261004_m1_lattice'/stem
    out=root.parent/(stem+'_analysis');out.mkdir(parents=True,exist_ok=True);decoded=root.parent/(stem+'_decoded')
    if a.stage=='prepare':prepare(root,out,a.smoke)
    elif a.stage=='evaluate':assert not a.smoke and a.name;evaluate(root,decoded,a.name)
    else:assert not a.smoke;report(root,decoded,out)


if __name__=='__main__':main()
