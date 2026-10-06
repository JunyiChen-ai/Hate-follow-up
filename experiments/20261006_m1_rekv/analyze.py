"""Strict source/native/proof binding followed by the single canonical evaluator."""
import argparse
import json
import subprocess
import sys
import time
import numpy as np
from inputs import ROOT,SPEC,CACHE,DATASETS,selected_rows,validate_input
from validate import validate_bundle
from src.mllm_renderer import cpu_position_renderer
from src.video_inputs import load_asr,frame_paths
from src.stance_cache import positions
from src.mllm_judge import VIDEO_QUESTION
import torch

METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')


def records(path):
    result={}
    for line in path.open():
        row=json.loads(line);key=row['dataset'],row['video_id'];assert key not in result;result[key]=row
    return result


def prepare(root,out,smoke):
    prepare_start=time.perf_counter()
    cfg=json.loads((root/'config.json').read_text());assert cfg['GT_read'] is False and cfg['spec']==SPEC and cfg['smoke']==smoke
    expected={(row['dataset'],row['video_id']):row for row in selected_rows(smoke)}
    rr={name:records(root/name/'predictions.jsonl') for name in ('base','optimized')}
    assert all(rows.keys()==expected.keys() for rows in rr.values())
    old=records(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl')
    renderer=cpu_position_renderer();asr={dataset:load_asr(dataset) for dataset in DATASETS};bundles={}
    for ordinal,(key,row) in enumerate(expected.items(),1):
        segments=asr[key[0]].get(key[1],[])
        meta=json.loads((CACHE/key[0]/key[1]/'metadata.json').read_text());validate_input(meta,row)
        bundle=json.loads((root/'records'/key[0]/(key[1]+'.json')).read_text());validate_bundle(renderer,row,segments,meta,bundle,smoke)
        assert all(bundle[name]==rr[name][key] for name in rr)
        base=bundle['base'];ctx=bundle['native_ctx']
        assert base['extra']['z_video']==old[key]['extra']['z_video'] and base['extra']['windows']==old[key]['extra']['windows']
        assert np.array_equal(base['score_curve'],old[key]['score_curve'])
        msgs,files=renderer.prefix_messages(frame_paths(*key,SPEC['native_requested_frames']),segments)
        text,encoded=renderer.encode_prefix(msgs,files)
        qids,qtext=renderer.branch_ids(msgs,VIDEO_QUESTION,head_text=text)
        aids,atext=renderer.answer_ids(msgs,VIDEO_QUESTION,ctx['stance'])
        assert ctx['msgs']==msgs and ctx['head']==text+qtext+atext
        assert ctx['history']==[{'role':'user','content':[{'type':'text','text':VIDEO_QUESTION}]},renderer.turn('assistant',ctx['stance'])]
        ids=torch.cat([encoded['input_ids'],torch.tensor([qids+aids])],1)
        p,delta=positions(renderer,ids,encoded['image_grid_thw'])
        assert ctx['stance_cache_tokens']==ids.shape[1] and ctx['stance_cache_logical_start']==int(p.max())+1
        assert bundle['native_rope']==delta.tolist()
        bundles[key]=bundle
        if ordinal%25==0:print('BOUND',ordinal,'/',len(expected),flush=True)
    result=dict(coverage=len(expected),GT_read=False,native_allraw_exact=True,global_speech_exact=True,
        source_actual_pixel_input_proof=True,source={},cost={},mechanism_supported=False)
    for dataset in DATASETS:
        bb=[bundle for key,bundle in bundles.items() if key[0]==dataset];cc=[bundle['checks'] for bundle in bb]
        result['source'][dataset]=dict(source_frames=sum(c['source_frames'] for c in cc),source_forwards=sum(c['source_forwards'] for c in cc),
            remote_layer_reads=sum(c['remote_layer_reads'] for c in cc),missing_local_windows=sum(c['missing_local_windows'] for c in cc),
            changed_visual_windows=sum(t['native_visual']!=t['new_visual'] for b in bb for t in b['traces']),
            clones=sum(t.get('clone_exact',False) for b in bb for t in b['traces']))
        assert result['source'][dataset]['source_frames']>0 and result['source'][dataset]['remote_layer_reads']>0
        assert result['source'][dataset]['changed_visual_windows']>0,'actual media retrieval has not entered newV; preserve execution diagnosis'
        result['cost'][dataset]=dict(standalone_seconds={name:sum(b[name]['extra']['standalone_seconds'] for b in bb) for name in rr},
            source_seconds=sum(c['source_seconds'] for c in cc),decode_seconds=sum(c['decode_seconds'] for c in cc),
            actual_forwards=sum(c['actual_forwards'] for c in cc),actual_vision=sum(c['actual_vision'] for c in cc),
            diagnostic_forwards=sum(c['diagnostic_forwards'] for c in cc),peak_GiB=max(c['peak_GiB'] for c in cc),
            temporary_dense_peak_bytes=max(c['temporary_dense_bytes'] for c in cc),
            times={name:sum(c['times'][name] for c in cc) for name in cc[0]['times']})
    result['actual_prepare_seconds']=time.perf_counter()-prepare_start
    result['cost_scope']='model/decode/binding processing estimate plus separately recorded extraction/reader attempt wall times and validation/proof I/O'
    result['pipeline_cost_sources']=[str(path.relative_to(ROOT)) for path in sorted(root.glob('pipeline_attempt_*.json'))]
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));print('PREPARE_PASS',flush=True)


def evaluate(root,decoded,name):
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(root/name/'predictions.jsonl'),
        '--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(root/name/'metrics.json')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(root/name),'--datasets',*DATASETS,
        '--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length','--min-windows','2',
        '--bma-grid','6','--arm','m2','--out-root',str(decoded),'--tag',name],cwd=ROOT,check=True)


def metrics(path):return {row['dataset']:row for row in json.loads(path.read_text())['per_dataset']}


def report(decoded,out):
    mm={name:metrics(decoded/name/'metrics.json') for name in ('base','optimized')}
    current=metrics(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json')
    result=dict(scope='development-selected; unchanged r6; mechanism unproven',metric_sources={name:str((decoded/name/'metrics.json').relative_to(ROOT)) for name in mm},datasets={},mechanism_supported=False)
    for dataset in DATASETS:
        assert all(mm['base'][dataset][key]==current[dataset][key] for key in METRICS)
        assert mm['base'][dataset]['n_videos_defined']==mm['optimized'][dataset]['n_videos_defined']==(84 if dataset=='HateMM' else 99)
        result['datasets'][dataset]=dict(final={name:{key:mm[name][dataset][key] for key in METRICS} for name in mm},
            n_eligible=mm['base'][dataset]['n_videos_defined'],delta={key:mm['optimized'][dataset][key]-current[dataset][key] for key in METRICS})
    common=[key for key in METRICS if all(result['datasets'][dataset]['delta'][key]>=.01 for dataset in DATASETS)]
    losses=all(result['datasets'][dataset]['delta'][key]>=(-.01 if key==METRICS[-1] else -.005) for dataset in DATASETS for key in METRICS)
    result['gates']=dict(common_gain_metrics=common,no_losses_beyond_noise=losses,performance_pass=bool(common and losses),
        any_qualifying_gain=any(result['datasets'][dataset]['delta'][key]>=.01 for dataset in DATASETS for key in METRICS))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('ANALYSIS_DONE',flush=True)


def main():
    stage_start=time.perf_counter()
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','evaluate','report'),required=True)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--name',choices=('base','optimized'));args=ap.parse_args()
    stem='r1_full_'+('smoke' if args.smoke else 'main');root=ROOT/'runs/20261006_m1_rekv'/stem
    out=root.parent/(stem+'_analysis');out.mkdir(parents=True,exist_ok=True);decoded=root.parent/(stem+'_decoded')
    completed=False
    try:
        if args.stage=='prepare':prepare(root,out,args.smoke)
        elif args.stage=='evaluate':assert not args.smoke and args.name;evaluate(root,decoded,args.name)
        else:assert not args.smoke;report(decoded,out)
        completed=True
    finally:
        attempt=1
        label=args.stage+('_'+args.name if args.name else '')
        while (out/f'{label}_attempt_{attempt:04d}_cost.json').exists():attempt+=1
        (out/f'{label}_attempt_{attempt:04d}_cost.json').write_text(json.dumps(dict(stage=args.stage,name=args.name,
            completed=completed,elapsed_seconds=time.perf_counter()-stage_start,scope='actual current stage wall time including strict source/native/proof replay and output I/O'),indent=2)+'\n')


if __name__=='__main__':main()
