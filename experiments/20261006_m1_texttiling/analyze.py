"""No-GT source/native replay, then canonical evaluator and unchanged fixedr6."""
import argparse
import json
import subprocess
import sys
import numpy as np
import torch
from inputs import ROOT,SPEC,DATASETS,selected_rows,source,validate_source
from validate import validate_bundle
from src.mllm_renderer import cpu_position_renderer
from src.video_inputs import load_asr,frame_paths
from src.mllm_judge import VIDEO_QUESTION
from src.stance_cache import positions
METRICS=('frame_ROC_AUC','frame_PR_AUC','within_video_macro_ROC_AUC')


def read(path):
    result={}
    for line in path.open():
        row=json.loads(line);key=row['dataset'],row['video_id'];assert key not in result;result[key]=row
    return result


def prepare(run,out,smoke):
    from transformers import AutoProcessor
    start=__import__('time').perf_counter();whisper=AutoProcessor.from_pretrained(SPEC['asr_model'],local_files_only=True)
    cfg=json.loads((run/'config.json').read_text());assert cfg['spec']==SPEC and cfg['smoke']==smoke and cfg['GT_read'] is False
    rows=selected_rows(smoke);expected={(r['dataset'],r['video_id']):r for r in rows}
    readings={name:read(run/name/'predictions.jsonl') for name in ('base','optimized')}
    assert all(r.keys()==expected.keys() for r in readings.values())
    old=read(ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl');j=cpu_position_renderer();asr={ds:load_asr(ds) for ds in DATASETS};bundles={}
    for key,row in expected.items():
        segments=asr[key[0]].get(key[1],[]);m,folder=source(row);validate_source(whisper,row,m,folder)
        b=json.loads((run/'records'/key[0]/(key[1]+'.json')).read_text());validate_bundle(j,row,segments,m,b,smoke)
        assert all(b[name]==readings[name][key] for name in readings)
        base=b['base'];assert base['extra']['z_video']==old[key]['extra']['z_video'] and base['extra']['windows']==old[key]['extra']['windows'] and np.array_equal(base['score_curve'],old[key]['score_curve'])
        msgs,files=j.prefix_messages(frame_paths(*key,SPEC['native_frames']),segments);text,encoded=j.encode_prefix(msgs,files);ctx=b['native_ctx']
        qids,qtext=j.branch_ids(msgs,VIDEO_QUESTION,head_text=text);aids,atext=j.answer_ids(msgs,VIDEO_QUESTION,ctx['stance'])
        assert ctx['msgs']==msgs and ctx['head']==text+qtext+atext and ctx['stance']==('Yes' if ctx['global_margin']>0 else 'No')
        assert ctx['history']==[dict(role='user',content=[dict(type='text',text=VIDEO_QUESTION)]),j.turn('assistant',ctx['stance'])]
        ids=torch.cat([encoded['input_ids'],torch.tensor([qids+aids])],1);p,delta=positions(j,ids,encoded['image_grid_thw'])
        assert base['extra']['stance_cache_tokens']==ids.shape[1] and base['extra']['stance_cache_logical_start']==int(p.max())+1 and b['native_rope']==delta.tolist()
        bundles[key]=b
    result=dict(coverage=len(rows),GT_read=False,native_allraw_exact=True,global_visual_stance_exact=True,actual_audio_source_time_pixel_replay=True,
        scope='development-selected; mechanism unproven',mechanism_supported=False,datasets={})
    for ds in DATASETS:
        bb=[b for key,b in bundles.items() if key[0]==ds];compiled=sum(p['reason']=='compiled' for b in bb for p in b['packets'])
        changed=sum(t['native_speech']!=t['new_speech'] for b in bb for t in b['traces']);assert compiled>0 and changed>0,'actual speech compiler execution guard failed'
        result['datasets'][ds]=dict(compiled=compiled,changed_speech_windows=changed,
            parser_forwards=sum(b['checks']['parser_forwards'] for b in bb),
            source_counts={key:sum(b['checks']['source_counts'][key] for b in bb) for key in bb[0]['checks']['source_counts']},
            times={key:sum(b['checks']['times'][key] for b in bb) for key in bb[0]['checks']['times']},
            source_seconds=sum(b['checks']['source_seconds'] for b in bb),
            processing_seconds={name:sum(b[name]['extra']['standalone_seconds'] for b in bb) for name in readings})
    result['prepare_wall_seconds']=__import__('time').perf_counter()-start
    (out/('plumbing_summary.json' if smoke else 'alignment.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('PREPARE_PASS',flush=True)


def evaluate(run,decoded,name):
    subprocess.run([sys.executable,'-m','src.eval.evaluate_four_datasets','--predictions',str(run/name/'predictions.jsonl'),'--gt-dir','data/gt_4fps','--datasets',*DATASETS,'--out',str(run/name/'metrics.json')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'experiments/20260926_twolevel/twolevel_r2.py','--run',str(run/name),'--datasets',*DATASETS,'--noleak','--transform','nscore','--key','calib','--duration','bma','--bma-prior','length','--min-windows','2','--bma-grid','6','--arm','m2','--out-root',str(decoded),'--tag',name],cwd=ROOT,check=True)


def metric_rows(path):return {r['dataset']:r for r in json.loads(path.read_text())['per_dataset']}


def report(decoded,out):
    values={name:metric_rows(decoded/name/'metrics.json') for name in ('base','optimized')};reference=metric_rows(ROOT/'runs/20260926_twolevel/r6_bma/metrics.json')
    result=dict(scope='development-selected; fixedr6; mechanism unproven',mechanism_supported=False,
        metric_sources={name:str((decoded/name/'metrics.json').relative_to(ROOT)) for name in values},datasets={})
    for ds in DATASETS:
        assert all(values['base'][ds][m]==reference[ds][m] for m in METRICS)
        assert values['base'][ds]['n_videos_defined']==values['optimized'][ds]['n_videos_defined']==(84 if ds=='HateMM' else 99)
        result['datasets'][ds]=dict(final={name:{m:values[name][ds][m] for m in METRICS} for name in values},n_eligible=values['base'][ds]['n_videos_defined'],
            delta={m:values['optimized'][ds][m]-reference[ds][m] for m in METRICS})
    common=[m for m in METRICS if all(result['datasets'][ds]['delta'][m]>=.01 for ds in DATASETS)]
    noise=all(result['datasets'][ds]['delta'][m]>=(-.01 if m==METRICS[-1] else -.005) for ds in DATASETS for m in METRICS)
    result['gates']=dict(common_gain_metrics=common,no_losses_beyond_noise=noise,performance_pass=bool(common and noise),
        any_qualifying_gain=any(result['datasets'][ds]['delta'][m]>=.01 for ds in DATASETS for m in METRICS))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));print('ANALYSIS_DONE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=('prepare','evaluate','report'));ap.add_argument('--smoke',action='store_true');ap.add_argument('--name',choices=('base','optimized'));args=ap.parse_args()
    stem='r1_full_smoke' if args.smoke else 'r1_full_main';run=ROOT/'runs/20261006_m1_texttiling'/stem;out=run.parent/(stem+'_analysis');out.mkdir(parents=True,exist_ok=True);decoded=run.parent/(stem+'_decoded')
    if args.stage=='prepare':prepare(run,out,args.smoke)
    elif args.stage=='evaluate':assert not args.smoke and args.name;evaluate(run,decoded,args.name)
    else:assert not args.smoke;report(decoded,out)


if __name__=='__main__':main()
