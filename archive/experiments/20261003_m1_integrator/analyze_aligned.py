#!/usr/bin/env python3
"""R4 full coverage, token-map validation and shared canonical analysis."""
import argparse
import json
import math
from pathlib import Path
import socket
import numpy as np
from aligned import aligned_edges
from analyze import read,evaluate,report,DATASETS,ARMS
from src.video_inputs import fixed_windows
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


def prepare(root,out,smoke=False):
    cfg=json.load((root/'config.json').open())
    assert cfg['future_keys']=='aligned' and cfg['run_name']==('r4_smoke' if smoke else 'r4_main') and cfg['smoke']==smoke
    assert cfg['model']=='Qwen/Qwen3-VL-8B-Instruct' and cfg['frames']==20 and cfg['window_seconds']==8 and cfg['fps']==4
    assert cfg['speech']==cfg['global_key']==cfg['forced_answer']=='native original' and cfg['GT_in_reader'] is False
    for a in ARMS:assert json.load((root/a/'config.json').open())=={**cfg,'arm':a}
    raw={a:read(root/a/'predictions.jsonl') for a in ARMS};checks=read(root/'checks.jsonl')
    manifest=read(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl')
    if smoke:
        expected={k for ds in DATASETS for k in [k for k in manifest if k[0]==ds][:2]}|{('HateMM','hate_video_114')}
        assert len(expected)==5
    else:expected={k for k in manifest if k[0] in DATASETS};assert len(expected)==333
    assert all(v.keys()==expected for v in [*raw.values(),checks])
    old=read(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');count=0;rows=[]
    for k,b in raw['base'].items():
        h=old[k];c=checks[k];dur=float(manifest[k]['duration']);ws=b['extra']['windows'];wins=fixed_windows(dur,8)
        V=len(wins);B=sum(1+('z_speech' in w) for w in ws)
        assert b['extra']['z_video']==h['extra']['z_video'] and np.array_equal(b['score_curve'],h['score_curve'])
        assert len(ws)==len(h['extra']['windows'])==V
        assert c['actual_forwards']==9+B+2*V+(3+B if smoke else 0)
        assert c['native_branches']==B and c['original_inputs_unchanged']
        if smoke:assert c['verify']['restored_native_global_exact'] and c['verify']['restored_native_windows_exact']
        for w,hw in zip(ws,h['extra']['windows']):
            for name in ('start','end','z_visual','z_speech'):
                assert w.get(name)==hw.get(name);count+=name.startswith('z_') and name in w
        with np.load(root/'tokens'/k[0]/(k[1]+'.npz')) as t:
            assert t['visual'].dtype==t['speech'].dtype==t['local'].dtype==bool
            assert len(t['visual'])==len(t['input_ids'])==c['prefix_tokens'] and t['local'].shape==(V,c['prefix_tokens'])
            assert c['mapping']['future_keys']=='aligned' and c['mapping']['language_layers']==36
            assert c['mapping']['visual_tokens']==int(t['visual'].sum()) and c['mapping']['ASR_tokens']==int(t['speech'].sum())
            assert c['mapping']['read_windows']==V
            edges=aligned_edges(t['visual'],t['speech'],t['local'])
            added=int(np.triu(edges,k=1).sum());assert added==c['mapping']['added_edges']
        drift={};cost={}
        for a in ARMS:
            r=raw[a][k];assert r['duration']==dur and r['native_rate']==4
            assert len(r['score_curve'])==math.ceil(4*dur) and np.isfinite(r['score_curve']).all()
            assert r['extra']['z_video']==b['extra']['z_video'] and r['extra']['stance']==b['extra']['stance']
            assert r['calls']==(3+B if a=='base' else 6+B) and len(r['extra']['windows'])==V
            for i,(w,v,extent) in enumerate(zip(ws,r['extra']['windows'],wins)):
                assert (v['i'],v['start'],v['end'])==(i,*extent) and w.get('z_speech')==v.get('z_speech')
                assert np.isfinite(v['z_visual']) and v['z']==max(v['z_visual'],v.get('z_speech',-np.inf))
            drift[a]=max(abs(w['z_visual']-v['z_visual']) for w,v in zip(ws,r['extra']['windows']))
            cost[a]=r['extra']['prefix_seconds']+r['extra']['branch_seconds']
        rows.append({'dataset':k[0],'video_id':k[1],'max_visual_delta':drift,'standalone_seconds':cost,
            'paired_seconds':sum(c[n] for n in ('global_seconds','base_seconds','mapping_seconds','causal_seconds','future_seconds')),
            'prefix_tokens':c['prefix_tokens'],'added_edges':added,'peak_GiB':c['peak_GiB']})
    result={'no_GT':True,'videos':len(expected),'native_branches_exact':count,'native_globals_exact':True,'rows':rows}
    if smoke:
        result['estimated_seconds']={ds:{a:n*np.mean([r['standalone_seconds'][a] for r in rows if r['dataset']==ds and r['video_id']!='hate_video_114']) for a in ARMS} for ds,n in [('HateMM',215),('HateClipSeg',118)]}
        result['estimated_paired_seconds']={ds:n*np.mean([r['paired_seconds'] for r in rows if r['dataset']==ds and r['video_id']!='hate_video_114']) for ds,n in [('HateMM',215),('HateClipSeg',118)]}
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('PREPARED',len(expected),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);ap.add_argument('--arm',choices=ARMS);a=ap.parse_args()
    if a.smoke and a.stage!='prepare':ap.error('smoke is plumbing only')
    if a.stage=='evaluate' and not a.arm:ap.error('evaluate requires arm')
    parent=ROOT/'runs/20261003_m1_integrator';name='r4_smoke' if a.smoke else 'r4_main'
    root=parent/name;decoded=parent/(name+'_decoded');out=root if a.smoke else parent/(name+'_analysis')
    out.mkdir(parents=True,exist_ok=True);print('host',socket.gethostname(),flush=True)
    if a.stage=='prepare':prepare(root,out,a.smoke)
    elif a.stage=='evaluate':evaluate(root,decoded,a.arm)
    else:report(root,decoded,out)

if __name__=='__main__':main()
