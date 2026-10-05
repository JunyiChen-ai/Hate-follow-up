#!/usr/bin/env python3
"""No-GT source replay first, then only the canonical evaluators and fixed r6."""
import argparse
import json
import subprocess
import sys
from types import SimpleNamespace,MethodType
import numpy as np
import torch
from measure import ROOT,SPEC,DATASETS,selected_rows,validate_bundle
from src.video_inputs import frame_paths,load_asr
from src.mllm_judge import MODEL,VIDEO_QUESTION
from src.mllm_renderer import cpu_renderer
from src.stance_cache import positions
from src.eval.evaluate import within_video_macro
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')


def read(path):
    result={}
    for line in path.open():
        r=json.loads(line);key=r['dataset'],r['video_id'];assert key not in result;result[key]=r
    return result


def metrics(path):return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']}


def position_renderer():
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    j=cpu_renderer();p=SimpleNamespace(config=AutoConfig.from_pretrained(MODEL,local_files_only=True))
    p.get_rope_index=MethodType(Qwen3VLModel.get_rope_index,p)
    if hasattr(Qwen3VLModel,'get_vision_position_ids'):p.get_vision_position_ids=MethodType(Qwen3VLModel.get_vision_position_ids,p)
    j.model=SimpleNamespace(model=p)
    return j


def replay_layout(j,row,b,segments):
    frames=frame_paths(row['dataset'],row['video_id'],20);plan=b['plan'];e=b['layout']
    msgs,paths=j.prefix_messages(frames,segments);text,enc=j.encode_prefix(msgs,paths)
    ids=enc['input_ids'];p,delta=positions(j,ids,enc['image_grid_thw']);visual=(ids[0]==j.image_token_id).nonzero().flatten()
    assert b['native_ctx']['msgs']==b['new_ctx']['msgs']==msgs
    for name in ('native_ctx','new_ctx'):
        ctx=b[name];qids,qtext=j.branch_ids(msgs,VIDEO_QUESTION,head_text=text);aids,atext=j.answer_ids(msgs,VIDEO_QUESTION,ctx['stance'])
        assert ctx['head']==text+qtext+atext and ctx['stance']==('Yes' if ctx['global_margin']>0 else 'No')
        assert ctx['history']==[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',ctx['stance'])]
    assert e['original_prefix_ids']==ids[0].tolist() and e['grid_thw']==enc['image_grid_thw'].tolist()
    keep=ids[0]!=j.image_token_id;keep[visual[torch.tensor(plan['roots'])]]=True;indices=keep.nonzero().flatten()
    assert e['sequence_indices']==indices.tolist() and e['packed_ids']==ids[0,indices].tolist()
    assert e['packed_positions']==p[:,0,indices].tolist() and e['source_indices']==plan['roots']
    assert b['base']['extra']['prefix_tokens']==ids.shape[1]
    assert b['optimized']['extra']['prefix_tokens']==len(indices)
    assert e['uncompressed_prefix_tokens']==ids.shape[1] and e['compressed_prefix_tokens']==len(indices)
    for name,length,logical in [('base',ids.shape[1],int(p.max())+1),('optimized',len(indices),max(max(ax) for ax in e['packed_positions'])+1)]:
        r=b[name];ctx=b['native_ctx' if name=='base' else 'new_ctx']
        qid,qtext=j.branch_ids(msgs,VIDEO_QUESTION,head_text=text);aid,atext=j.answer_ids(msgs,VIDEO_QUESTION,ctx['stance'])
        assert r['extra']['stance_cache_tokens']==length+len(qid)+len(aid)
        assert r['extra']['stance_cache_logical_start']==logical+len(qid)+len(aid)
    if b['dense_check'] is not None:assert b['dense_check']['positions']==p[:,0].tolist()


def prepare(root,out,smoke):
    cfg=json.loads((root/'config.json').read_text());assert cfg['spec']==SPEC and cfg['GT_read'] is False and cfg['smoke']==smoke
    rows=selected_rows(smoke);expected={(r['dataset'],r['video_id']):r for r in rows}
    rr={name:read(root/name/'predictions.jsonl') for name in ('base','optimized')};assert all(r.keys()==expected.keys() for r in rr.values())
    old=read(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');j=position_renderer();asr={ds:load_asr(ds) for ds in DATASETS}
    bundles={};changed=0
    for key,row in expected.items():
        b=json.loads((root/'records'/key[0]/(key[1]+'.json')).read_text());segments=asr[key[0]].get(key[1],[])
        validate_bundle(row,b,segments,j,smoke);replay_layout(j,row,b,segments)
        assert all(b[name]==rr[name][key] for name in rr)
        base,new=b['base'],b['optimized'];assert base['extra']['z_video']==old[key]['extra']['z_video']
        assert base['extra']['windows']==old[key]['extra']['windows'] and np.array_equal(base['score_curve'],old[key]['score_curve'])
        assert all(a.get('z_speech')==z.get('z_speech') for a,z in zip(base['extra']['windows'],new['extra']['windows']))
        changed+=sum(a['z_visual']!=z['z_visual'] for a,z in zip(base['extra']['windows'],new['extra']['windows']))
        bundles[key]=b
    result=dict(coverage=len(rows),GT_read=False,native_allraw_exact=True,global_speech_exact=True,source_layout_replayed=True,
        changed_visual_windows=changed,cloned_cache_checks=sum(t['optimized']['visual'].get('repeat_exact',False) for b in bundles.values() for t in b['traces']),source={},cost={},mechanism_supported=False)
    for ds in DATASETS:
        bb=[b for key,b in bundles.items() if key[0]==ds];cc=[b['checks'] for b in bb]
        result['source'][ds]=dict(original_tokens=sum(b['plan']['original_count'] for b in bb),retained_tokens=sum(b['plan']['retained_count'] for b in bb),
            compressed_videos=sum(b['plan']['retained_count']<b['plan']['original_count'] for b in bb),merged_components_videos=sum(any(len(m)>1 for m in b['plan']['members']) for b in bb),
            zero_token_source_blocks=sum(n==0 for b in bb for n in b['plan']['counts']))
        assert result['source'][ds]['compressed_videos']>0,'token compression did not execute in this corpus'
        result['cost'][ds]=dict(standalone_seconds={name:sum(b[name]['extra']['standalone_seconds'] for b in bb) for name in rr},
            peak_GiB=max(c['peak_GiB'] for c in cc),actual_language_forwards=sum(c['actual_language_forwards'] for c in cc),
            actual_vision_forwards=sum(c['actual_vision_forwards'] for c in cc),diagnostic_forwards=sum(c['diagnostic_forwards'] for c in cc),
            times={k:sum(c['times'][k] for c in cc) for k in cc[0]['times']})
    assert changed>0
    if smoke:assert result['cloned_cache_checks']==5 and all(b['dense_check']['native_order_exact'] for b in bundles.values())
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def evaluate(root,decoded,name):
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(root/name/'predictions.jsonl'),
        '--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(root/name/'metrics.json')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(root/name),
        '--datasets',*DATASETS,'--noleak','--transform','nscore','--key','calib','--duration','bma',
        '--bma-prior','length','--min-windows','2','--bma-grid','6','--arm','m2','--out-root',str(decoded),'--tag',name],cwd=ROOT,check=True)


def report(root,decoded,out):
    mm={a:metrics(decoded/a/'metrics.json') for a in ('base','optimized')};current=metrics(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json')
    raw={a:read(root/a/'predictions.jsonl') for a in mm};final={a:read(decoded/a/'predictions.jsonl') for a in mm}
    result=dict(scope='development-selected; fixed r6; not mechanism evidence',metric_sources={a:str((decoded/a/'metrics.json').relative_to(ROOT)) for a in mm},datasets={},mechanism_supported=False)
    per_video=[]
    for ds in DATASETS:
        assert all(mm['base'][ds][m]==current[ds][m] for m in METRICS),'native six metrics mismatch'
        gt=np.load(ROOT/f'data/gt_4fps/{ds}.npz',allow_pickle=True)
        ys={str(v):np.asarray(gt['y4'][i]) for i,v in enumerate(gt['video_ids']) if str(gt['split'][i])=='test'}
        counts=0
        for key in [k for k in raw['base'] if k[0]==ds]:
            y=ys[key[1]];r=dict(dataset=ds,video_id=key[1])
            for a in mm:
                wins=raw[a][key]['extra']['windows'];idx=np.clip(((np.arange(len(raw[a][key]['score_curve']))+.5)/4//8).astype(int),0,len(wins)-1)
                curves=dict(final=np.asarray(final[a][key]['score_curve']),raw_max=np.asarray(raw[a][key]['score_curve']),raw_visual=np.asarray([w['z_visual'] for w in wins])[idx])
                r[a]={kind:within_video_macro({key[1]:y},{key[1]:curve})[METRICS[-1]] for kind,curve in curves.items()}
            if r['base']['final'] is not None:per_video.append(r);counts+=1
        assert counts==(84 if ds=='HateMM' else 99)
        result['datasets'][ds]=dict(final={a:{m:mm[a][ds][m] for m in METRICS} for a in mm},delta={m:mm['optimized'][ds][m]-current[ds][m] for m in METRICS},n_eligible=counts)
    common=[m for m in METRICS if all(result['datasets'][ds]['delta'][m]>=.01 for ds in DATASETS)]
    no_losses=all(result['datasets'][ds]['delta'][m]>=(-.01 if m==METRICS[-1] else -.005) for ds in DATASETS for m in METRICS)
    result['gates']=dict(common_gain_metrics=common,no_losses_beyond_noise=no_losses,performance_pass=bool(common and no_losses),any_qualifying_gain=any(result['datasets'][ds]['delta'][m]>=.01 for ds in DATASETS for m in METRICS))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');(out/'per_video.json').write_text(json.dumps(per_video,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True);print('ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--name',choices=('base','optimized'));a=ap.parse_args()
    stem='r1_full_'+('smoke' if a.smoke else 'main');root=ROOT/'runs/20261005_m1_ott'/stem;out=root.parent/(stem+'_analysis');out.mkdir(parents=True,exist_ok=True);decoded=root.parent/(stem+'_decoded')
    if a.stage=='prepare':prepare(root,out,a.smoke)
    elif a.stage=='evaluate':assert not a.smoke and a.name;evaluate(root,decoded,a.name)
    else:assert not a.smoke;report(root,decoded,out)


if __name__=='__main__':main()
