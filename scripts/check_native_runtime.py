#!/usr/bin/env python3
"""Five-video actual GPU reproduction of frozen native readings, without GT."""
import json
import argparse
import logging
import socket
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mllm_judge import Judge,MODEL,yesno_question
from src.stance_cache import build,margin
from src.video_inputs import load_manifest,load_asr,frame_paths,fixed_windows,window_text


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='runs/_setup_local_hatevlm/native_smoke');args=ap.parse_args();out=ROOT/args.out;assert out.resolve().is_relative_to(ROOT);out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler(sys.stdout)])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    import transformers
    config=dict(host=socket.gethostname(),model=MODEL,torch=torch.__version__,transformers=transformers.__version__,GT_read=False,
        code='scripts/check_native_runtime.py + unchanged src/{mllm_judge,stance_cache,video_inputs}.py; sources2026-10-05',command='python -u '+' '.join(sys.argv))
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    allrows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',('HateMM','HateClipSeg'))
    rows=[r for ds in ('HateMM','HateClipSeg') for r in [x for x in allrows if x['dataset']==ds][:2]]+[
        r for r in allrows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    old={}
    for line in (ROOT/'runs/20260926_glr/base_gridA/predictions.jsonl').read_text().splitlines():
        r=json.loads(line);old[r['dataset'],r['video_id']]=r
    asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};torch.manual_seed(0);j=Judge(MODEL);results=[]
    for row in rows:
        ds,video=row['dataset'],row['video_id'];segments=asr[ds].get(video,[])
        frames=frame_paths(ds,video,20);cache,ctx=build(j,frames,segments);reference=old[ds,video]
        assert ctx['global_margin']==reference['extra']['z_video'] and ctx['stance']==reference['extra']['stance']
        windows=fixed_windows(float(row['duration']),8);values=[]
        for i,(a,b) in enumerate(windows):
            body=window_text(segments,a,b)
            w=dict(i=i,start=a,end=b,z_visual=margin(j,cache,ctx,yesno_question(i,len(windows),a,b,body,'visual')))
            if body.strip():w['z_speech']=margin(j,cache,ctx,yesno_question(i,len(windows),a,b,body,'speech'))
            w['z']=max(w['z_visual'],w.get('z_speech',float('-inf')));values.append(w)
        assert values==reference['extra']['windows'];del cache
        result=dict(dataset=ds,video_id=video,overview_images=len(frames),windows=len(windows),native_allraw_exact=True)
        results.append(result);logging.info('native exact %s/%s windows=%d',ds,video,len(windows))
    (out/'summary.json').write_text(json.dumps(dict(**config,videos=results,coverage=5,native_allraw_exact=True),indent=2)+'\n')
    logging.info('NATIVE_RUNTIME_DONE')


if __name__=='__main__':main()
