#!/usr/bin/env python3
"""Source binding before annotation access; canonical evaluation after scoring."""
import argparse
import json
import subprocess
import sys
import numpy as np
from inputs import ROOT, CACHE, DATASETS, selected_rows
from measure import validate_bundle
from src.mllm_renderer import cpu_renderer
from src.video_inputs import load_asr
from src.eval.evaluate import within_video_macro

METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')


def read(path):
    rows={}
    for line in path.open():
        r=json.loads(line);key=r['dataset'],r['video_id'];assert key not in rows;rows[key]=r
    return rows


def metrics(path):return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']}


def prepare(root,out,smoke):
    rows=selected_rows(smoke);expected={(r['dataset'],r['video_id']):r for r in rows}
    cfg=json.loads((root/'config.json').read_text());assert cfg['GT_read'] is False and cfg['smoke']==smoke
    records={name:read(root/name/'predictions.jsonl') for name in ('base','optimized')}
    assert all(r.keys()==expected.keys() for r in records.values())
    old=read(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl')
    asr={ds:load_asr(ds) for ds in DATASETS};renderer=cpu_renderer();bundles={};inputs={}
    changed_visual=changed_speech=repeated=0
    for key,row in expected.items():
        m=json.loads((CACHE/key[0]/key[1]/'metadata.json').read_text())
        bundle=json.loads((root/'records'/key[0]/(key[1]+'.json')).read_text())
        validate_bundle(row,bundle,m,asr[key[0]].get(key[1],[]),renderer,smoke)
        assert all(bundle[name]==records[name][key] for name in records)
        base,new=bundle['base'],bundle['optimized']
        assert base['extra']['z_video']==old[key]['extra']['z_video'] and base['extra']['windows']==old[key]['extra']['windows']
        assert np.array_equal(base['score_curve'],old[key]['score_curve'])
        for a,b in zip(base['extra']['windows'],new['extra']['windows']):
            changed_visual+=a['z_visual']!=b['z_visual'];changed_speech+=a.get('z_speech')!=b.get('z_speech')
        repeated+=sum(t['branches'][kind].get('repeat_exact',False) for t in bundle['traces'] for kind in ('visual','speech'))
        bundles[key]=bundle;inputs[key]=m
    result=dict(coverage=len(rows),native_exact=True,GT_read=False,global_exact=False,changed_visual_windows=changed_visual,
        changed_speech_windows=changed_speech,actual_repeat_checks=repeated,cost={},source={},mechanism_supported=False)
    for ds in DATASETS:
        keys=[k for k in expected if k[0]==ds];bb=[bundles[k] for k in keys];mm=[inputs[k] for k in keys]
        result['source'][ds]=dict(leaf_calls=sum(len(m['leaves']) for m in mm),parent_calls=sum(len(m['parents']) for m in mm),
            repair_calls=sum(len(m['repairs']) for m in mm),repaired_status_changes=sum(old['record']['status']!=new['status'] for m in mm for old,new in zip(m['leaves'],m['final_leaves'])),
            leaf_token_caps=sum(l['generation']['truncated'] for m in mm for l in m['leaves']),
            parent_token_caps=sum(l['generation']['truncated'] for m in mm for l in m['parents'].values()),
            repair_token_caps=sum(l['generation']['truncated'] for m in mm for l in m['repairs']),
            final_present_leaves=sum(l['status']=='present' for m in mm for l in m['final_leaves']),
            final_unknown_leaves=sum(l['status']=='UNKNOWN' for m in mm for l in m['final_leaves']),
            final_unresolved_parents=sum(r.get('disagreement',False) for m in mm for r in m['final_tree']),
            missing_local_frame_windows=sum(len(m['source']['uncovered_windows']) for m in mm))
        result['cost'][ds]=dict(standalone_seconds={name:sum(b[name]['extra']['standalone_seconds'] for b in bb) for name in records},
            peak_GiB=max(b['checks']['peak_GiB'] for b in bb),
            decode_seconds=sum(m['source']['decode_seconds'] for m in mm),generation_seconds=sum(m['generation_seconds'] for m in mm),
            input_actual_vision_forwards=sum(m['actual_vision_forwards'] for m in mm),
            generation_tokens=sum(len(l['generation']['tokens']) for m in mm for l in m['leaves']+list(m['parents'].values())+m['repairs']),
            **{k:sum(b['checks'][k] for b in bb) for k in ('actual_forwards','actual_vision_forwards','input_actual_forwards',
                'prefix_seconds','new_prefix_seconds','input_binding_seconds','native_visual','native_speech','new_visual','new_speech','diagnostic','diagnostic_forwards','preprocessing_seconds')})
        if smoke:
            sample=[b for b in bb if b['base']['video_id']!='hate_video_114']
            result['cost'][ds]['rough_full_seconds']=(215 if ds=='HateMM' else 118)*np.mean([b['optimized']['extra']['standalone_seconds'] for b in sample])
    if smoke:assert repeated==sum(1+int(any(w['body'].strip() for w in inputs[k]['windows'])) for k in expected)
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

def evaluate(root,decoded,name):
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(root/name/'predictions.jsonl'),
        '--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(root/name/'metrics.json')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(root/name),
        '--datasets',*DATASETS,'--noleak','--transform','nscore','--key','calib','--duration','bma',
        '--bma-prior','length','--min-windows','2','--bma-grid','6','--arm','m2',
        '--out-root',str(decoded),'--tag',name],cwd=ROOT,check=True)


def bootstrap(values):
    x=np.asarray(values);rng=np.random.default_rng(0)
    if not len(x):return dict(n=0,mean=None,CI95=None)
    means=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return dict(n=len(x),mean=float(x.mean()),CI95=np.quantile(means,[.025,.975]).tolist())


def report(root,decoded,out):
    mm={a:metrics(decoded/a/'metrics.json') for a in ('base','optimized')}
    current=metrics(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json')
    raw={a:read(root/a/'predictions.jsonl') for a in mm};final={a:read(decoded/a/'predictions.jsonl') for a in mm}
    expected={(r['dataset'],r['video_id']) for r in selected_rows(False)}
    assert all(r.keys()==expected for r in [*raw.values(),*final.values()])
    result=dict(scope='development-selected; unchanged r6 independently applied',
        metric_sources={a:str((decoded/a/'metrics.json').relative_to(ROOT)) for a in mm},datasets={},mechanism_supported=False)
    per_video=[]
    for ds in DATASETS:
        assert all(mm['base'][ds][m]==current[ds][m] for m in METRICS),'native six metrics mismatch'
        gt=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True)
        ys={str(v):np.asarray(gt['y4'][i]) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        paired={kind:[] for kind in ('final','raw_max','raw_visual','raw_speech_shared')}
        for key in [k for k in raw['base'] if k[0]==ds]:
            y=ys[key[1]];r=dict(dataset=ds,video_id=key[1]);ww=raw['base'][key]['extra']['windows']
            idx=np.clip(((np.arange(len(raw['base'][key]['score_curve']))+.5)/4//8).astype(int),0,len(ww)-1)
            shared=np.asarray(['z_speech' in b and 'z_speech' in s for b,s in zip(ww,raw['optimized'][key]['extra']['windows'])])[idx]
            for a in mm:
                wins=raw[a][key]['extra']['windows']
                curves=dict(final=np.asarray(final[a][key]['score_curve']),raw_max=np.asarray(raw[a][key]['score_curve']),
                    raw_visual=np.asarray([w['z_visual'] for w in wins])[idx])
                r[a]={kind:within_video_macro({key[1]:y},{key[1]:curve})[METRICS[-1]] for kind,curve in curves.items()}
                speech=np.asarray([w.get('z_speech',0.) for w in wins])[idx];n=min(len(y),len(speech));use=shared[:n]
                r[a]['raw_speech_shared']=within_video_macro({key[1]:y[:n][use]},{key[1]:speech[:n][use]})[METRICS[-1]]
            if r['base']['final'] is not None:
                per_video.append(r)
                for kind in paired:
                    if r['base'][kind] is not None and r['optimized'][kind] is not None:paired[kind].append(r['optimized'][kind]-r['base'][kind])
        assert len(paired['final'])==(84 if ds=='HateMM' else 99)
        delta={m:mm['optimized'][ds][m]-current[ds][m] for m in METRICS}
        result['datasets'][ds]=dict(final={a:{m:mm[a][ds][m] for m in METRICS} for a in mm},delta=delta,
            n_eligible=len(paired['final']),within_paired={kind:bootstrap(v) for kind,v in paired.items()})
    common=[m for m in METRICS if all(result['datasets'][ds]['delta'][m]>=.01 for ds in DATASETS)]
    no_losses=all(result['datasets'][ds]['delta'][m]>=(-.01 if m==METRICS[-1] else -.005) for ds in DATASETS for m in METRICS)
    result['gates']=dict(common_gain_metrics=common,no_losses_beyond_noise=no_losses,performance_pass=bool(common and no_losses),
        any_qualifying_gain=any(result['datasets'][ds]['delta'][m]>=.01 for ds in DATASETS for m in METRICS))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(out/'per_video.json').write_text(json.dumps(per_video,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True);print('ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--name',choices=('base','optimized'));a=ap.parse_args()
    stem='r1_full_'+('smoke' if a.smoke else 'main');root=ROOT/'runs/20261005_m1_interval_witness'/stem
    out=root.parent/(stem+'_analysis');out.mkdir(parents=True,exist_ok=True);decoded=root.parent/(stem+'_decoded')
    if a.stage=='prepare':prepare(root,out,a.smoke)
    elif a.stage=='evaluate':assert not a.smoke and a.name;evaluate(root,decoded,a.name)
    else:assert not a.smoke;report(root,decoded,out)


if __name__=='__main__':main()
