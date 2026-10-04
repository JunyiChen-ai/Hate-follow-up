#!/usr/bin/env python3
"""Full-corpus original-audio character alignment; no labels or evaluation."""
import argparse
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import numpy as np
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.video_inputs import load_manifest,load_asr,fixed_windows
from alignment import ALIGN_MODEL,ALGORITHM,CACHE_VERSION,WhisperAligner,words_from_segments

DATASETS=('HateMM','HateClipSeg')
CACHE=ROOT/'data/acoustic_path_support'


def resolve_video(row):
    given=Path(row['video_path'])
    candidates=[given]+[p for folder in ('video','videos')
        for p in sorted((Path.home()/'data'/row['dataset']/folder).glob(row['video_id']+'.*'))]
    video=next((p for p in candidates if p.is_file()),None)
    if video is None:raise FileNotFoundError((row['dataset'],row['video_id']))
    return video


def selected_rows(smoke):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:
        rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[
            r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def validate_cache(r,row,segments):
    assert r['algorithm']==ALGORITHM and r['model']==ALIGN_MODEL
    assert r['cache_version']==CACHE_VERSION
    assert r['dataset']==row['dataset'] and r['video_id']==row['video_id']
    assert r['duration']==float(row['duration']) and r['words']==words_from_segments(segments)
    assert r['segments']==[list(s) for s in segments]
    assert r.get('manifest_video_path',r['input_video'])==row['video_path']
    assert Path(r['input_video']).stem==row['video_id']
    wins=fixed_windows(float(row['duration']),8)
    assert r['windows']==[list(w) for w in wins] and r['GT_read'] is False
    for name in ('soft','viterbi','proportional'):
        p=np.asarray(r[name],dtype=float)
        if len(r['words']):
            assert p.shape==(len(r['words']),len(wins)) and np.isfinite(p).all()
            assert (p>=0).all() and np.allclose(p.sum(1),1)
        else: assert r[name]==[]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261004_m1_acoustic'/('r1_extract_'+('smoke' if a.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    config=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=ALIGN_MODEL,algorithm=ALGORITHM,cache_version=CACHE_VERSION,
        torch=torch.__version__,transformers=transformers.__version__,GT_read=False,smoke=a.smoke,
        code='experiments/20261004_m1_acoustic/{alignment,extract}.py + src/video_inputs.py; sources2026-10-04')
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n');torch.manual_seed(0)
    rows=selected_rows(a.smoke);asr={ds:load_asr(ds) for ds in DATASETS};aligner=WhisperAligner();started=time.time()
    CACHE.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(rows):
        ds,vid=row['dataset'],row['video_id'];segments=asr[ds].get(vid,[])
        dest=CACHE/ds/(vid+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            r=json.loads(dest.read_text());validate_cache(r,row,segments);logging.info('%d/%d reuse %s/%s',i+1,len(rows),ds,vid);continue
        video=resolve_video(row)
        begin=time.perf_counter()
        if words_from_segments(segments):
            pcm=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vn','-ac','1','-ar','16000','-f','f32le','-'],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
            audio=np.frombuffer(pcm,dtype='<f4').copy();assert len(audio)>0 and np.isfinite(audio).all()
        else:audio=np.empty(0,dtype=np.float32)
        decode_seconds=time.perf_counter()-begin
        torch.cuda.synchronize();t0=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        r=aligner.align(audio,segments,fixed_windows(float(row['duration']),8))
        torch.cuda.synchronize();align_seconds=time.perf_counter()-t0
        r.update(dataset=ds,video_id=vid,duration=float(row['duration']),segments=segments,
            windows=fixed_windows(float(row['duration']),8),algorithm=ALGORITHM,model=ALIGN_MODEL,cache_version=CACHE_VERSION,GT_read=False,
            input_video=str(video),manifest_video_path=row['video_path'],input_asr=f'data/asr_whisper_large_v3/{ds}/timestamped_chunks.jsonl',
            host=socket.gethostname(),date=config['date'],code=config['code'],audio_seconds=len(audio)/16000,
            decode_seconds=decode_seconds,align_seconds=align_seconds,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
        validate_cache(json.loads(json.dumps(r)),row,segments)
        temporary=dest.with_suffix('.partial');temporary.write_text(json.dumps(r)+'\n');temporary.replace(dest)
        logging.info('%d/%d %s/%s words=%d blocks=%d %.2fs',i+1,len(rows),ds,vid,len(r['words']),r['encoder_calls'],align_seconds)
    selected=[json.loads((CACHE/r['dataset']/(r['video_id']+'.json')).read_text()) for r in rows]
    summary=dict(coverage=len(rows),GT_read=False,model=ALIGN_MODEL,host=socket.gethostname(),datasets={})
    for ds in DATASETS:
        records=[r for r in selected if r['dataset']==ds]
        summary['datasets'][ds]={name:sum(r[name] for r in records) for name in (
            'decode_seconds','align_seconds','encoder_calls','decoder_calls','character_tokens','audio_seconds')}
        summary['datasets'][ds]['peak_GiB']=max(r['peak_GiB'] for r in records)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    provenance='''# Acoustic path support provenance

Generated by experiments/20261004_m1_acoustic/{alignment,extract}.py (sources2026-10-04).
Frozen model openai/whisper-large-v3; no ground truth. Original source ASR:
data/asr_whisper_large_v3/{dataset}/timestamped_chunks.jsonl, filled shared loader.
Original media path, generation host/date, complete config and actual costs are in each JSON.
Command: sbatch experiments/20261004_m1_acoustic/launch/lab2.sbatch {smoke|main} soft
Native test manifest supplies only identity/duration/path; no labels in alignment.
Algorithm and source differences: experiments/20261004_m1_acoustic/README.md.
Cached words/support validated by actual parsing, exact lexical/time identity, array shapes/normalization.
'''
    provenance+='\nGenerating hosts in returned cache: '+', '.join(sorted({r['host'] for r in selected}))+'; date '+config['date']+'.\n'
    (CACHE/'PROVENANCE.md').write_text(provenance)
    logging.info('EXTRACTION_DONE coverage=%d elapsed=%.2fs',len(rows),time.time()-started)


if __name__=='__main__': main()
