"""Strict source/current-input replay before canonical evaluation and fixed r6."""
import argparse
import json
import subprocess
import sys
import numpy as np
import torch
from measure import ROOT,SPEC,DATASETS,selected_rows,validate_bundle,validate_acquisition,CACHE
from src.mllm_renderer import cpu_position_renderer
from src.mllm_judge import VIDEO_QUESTION
from src.video_inputs import load_asr,frame_paths
from src.stance_cache import positions
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')


def read(path):
    rows={}
    for line in path.open():
        r=json.loads(line);key=r['dataset'],r['video_id'];assert key not in rows;rows[key]=r
    return rows


def metrics(path):return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']}


def prepare(root,out,smoke):
    cfg=json.loads((root/'config.json').read_text());assert cfg['GT_read'] is False and cfg['spec']==SPEC and cfg['smoke']==smoke
    rows=selected_rows(smoke);expected={(r['dataset'],r['video_id']):r for r in rows}
    rr={name:read(root/name/'predictions.jsonl') for name in ('base','optimized')};assert all(r.keys()==expected.keys() for r in rr.values())
    old=read(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');j=cpu_position_renderer();asr={ds:load_asr(ds) for ds in DATASETS};bundles={}
    for key,row in expected.items():
        segments=asr[key[0]].get(key[1],[]);m=json.loads((CACHE/key[0]/key[1]/'metadata.json').read_text());validate_acquisition(j,m,row,segments,CACHE/key[0]/key[1])
        b=json.loads((root/'records'/key[0]/(key[1]+'.json')).read_text());validate_bundle(row,b,segments,m,j,smoke)
        assert all(b[name]==rr[name][key] for name in rr);base,new=b['base'],b['optimized']
        assert base['extra']['z_video']==old[key]['extra']['z_video'] and base['extra']['windows']==old[key]['extra']['windows'] and np.array_equal(base['score_curve'],old[key]['score_curve'])
        msgs,files=j.prefix_messages(frame_paths(*key,SPEC['native_frames']),segments);text,enc=j.encode_prefix(msgs,files);ctx=b['native_ctx']
        qids,qtext=j.branch_ids(msgs,VIDEO_QUESTION,head_text=text);aids,atext=j.answer_ids(msgs,VIDEO_QUESTION,ctx['stance'])
        assert ctx['msgs']==msgs and ctx['head']==text+qtext+atext and ctx['stance']==('Yes' if ctx['global_margin']>0 else 'No')
        assert ctx['history']==[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},j.turn('assistant',ctx['stance'])]
        ids=torch.cat([enc['input_ids'],torch.tensor([qids+aids])],1);p,delta=positions(j,ids,enc['image_grid_thw'])
        assert base['extra']['prefix_tokens']==enc['input_ids'].shape[1] and base['extra']['stance_cache_tokens']==ids.shape[1]
        assert base['extra']['stance_cache_logical_start']==int(p.max())+1 and b['native_rope']==delta.tolist()
        bundles[key]=b
    result=dict(coverage=len(rows),GT_read=False,native_allraw_exact=True,global_exact=True,source_pixel_coordinate_input_replay=True,cost={},source={},mechanism_supported=False)
    for ds in DATASETS:
        bb=[b for key,b in bundles.items() if key[0]==ds];cc=[b['checks'] for b in bb]
        result['source'][ds]=dict(caption_calls=sum(c['caption_calls'] for c in cc),filter_calls=sum(c['filter_calls'] for c in cc),embedding_calls=sum(c['embedding_calls'] for c in cc),
            remote_windows=sum(c['remote_windows'] for c in cc),source_windows=sum(c['source_windows'] for c in cc),
            changed_visual_windows=sum(t['native_visual']!=t['new_visual'] for b in bb for t in b['traces']),changed_speech_windows=sum(t['native_speech']!=t['new_speech'] for b in bb for t in b['traces']),
            visual_clones=sum(t.get('clone_exact',False) for b in bb for t in b['traces']),speech_clones=sum(t.get('speech_clone_exact',False) for b in bb for t in b['traces']))
        assert result['source'][ds]['remote_windows']>0 and result['source'][ds]['changed_visual_windows']>0 and result['source'][ds]['changed_speech_windows']>0,'actual remote-source reader not exercised; preserve UNKNOWN/interface diagnosis'
        result['cost'][ds]=dict(standalone_seconds={name:sum(b[name]['extra']['standalone_seconds'] for b in bb) for name in rr},peak_GiB=max(c['peak_GiB'] for c in cc),
            source_forwards=sum(c['source_forwards'] for c in cc),source_vision=sum(c['source_vision'] for c in cc),reader_forwards=sum(c['actual_forwards'] for c in cc),reader_vision=sum(c['actual_vision'] for c in cc),
            source_seconds=sum(c['source_seconds'] for c in cc),times={k:sum(c['times'][k] for c in cc) for k in cc[0]['times']})
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('PREPARE_PASS',flush=True)


def evaluate(root,decoded,name):
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(root/name/'predictions.jsonl'),'--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(root/name/'metrics.json')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(root/name),'--datasets',*DATASETS,'--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length','--min-windows','2','--bma-grid','6','--arm','m2','--out-root',str(decoded),'--tag',name],cwd=ROOT,check=True)


def report(decoded,out):
    mm={name:metrics(decoded/name/'metrics.json') for name in ('base','optimized')};current=metrics(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json')
    result=dict(scope='development-selected; fixedr6; mechanism unproven',metric_sources={name:str((decoded/name/'metrics.json').relative_to(ROOT)) for name in mm},datasets={},mechanism_supported=False)
    for ds in DATASETS:
        assert all(mm['base'][ds][m]==current[ds][m] for m in METRICS)
        assert mm['base'][ds]['n_videos_defined']==mm['optimized'][ds]['n_videos_defined']==(84 if ds=='HateMM' else 99)
        result['datasets'][ds]=dict(final={name:{m:mm[name][ds][m] for m in METRICS} for name in mm},n_eligible=mm['base'][ds]['n_videos_defined'],delta={m:mm['optimized'][ds][m]-current[ds][m] for m in METRICS})
    common=[m for m in METRICS if all(result['datasets'][ds]['delta'][m]>=.01 for ds in DATASETS)]
    losses=all(result['datasets'][ds]['delta'][m]>=(-.01 if m==METRICS[-1] else -.005) for ds in DATASETS for m in METRICS)
    result['gates']=dict(common_gain_metrics=common,no_losses_beyond_noise=losses,performance_pass=bool(common and losses),any_qualifying_gain=any(result['datasets'][ds]['delta'][m]>=.01 for ds in DATASETS for m in METRICS))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--name',choices=('base','optimized'));a=ap.parse_args()
    stem='r1_full_'+('smoke' if a.smoke else 'main');root=ROOT/'runs/20261006_m1_merit'/stem;out=root.parent/(stem+'_analysis');out.mkdir(parents=True,exist_ok=True);decoded=root.parent/(stem+'_decoded')
    if a.stage=='prepare':prepare(root,out,a.smoke)
    elif a.stage=='evaluate':assert not a.smoke and a.name;evaluate(root,decoded,a.name)
    else:assert not a.smoke;report(decoded,out)


if __name__=='__main__':main()
