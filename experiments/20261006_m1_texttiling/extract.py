"""Actual PTS audio, same Whisper greedy words/DTW, lexical scope sources."""
import argparse
import json
import logging
import math
import os
import socket
import time
import numpy as np
import torch
from inputs import ROOT,SPEC,CACHE,DATASETS,selected_rows,validate_source
from alignment import Recognizer
from partition import partition,scope
from src.audio_inputs import resolve_video,decode_audio,crop_audio
from src.video_inputs import fixed_windows


def clock():torch.cuda.synchronize();return time.perf_counter()


def main():
    start_pipeline=time.perf_counter();ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    out=ROOT/'runs/20261006_m1_texttiling'/('r1_extract_smoke' if args.smoke else 'r1_extract_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers,av
    config=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),spec=SPEC,model=SPEC['asr_model'],GT_read=False,
        smoke=args.smoke,torch=torch.__version__,transformers=transformers.__version__,av=av.__version__,
        code='experiments/20261006_m1_texttiling/{alignment,extract,partition}.py;src/audio_inputs.py;2026-10-06')
    path=out/'config.json'
    if path.exists():assert json.loads(path.read_text())==config,'changed source configuration; refuse overwrite'
    else:path.write_text(json.dumps(config,indent=2)+'\n')
    audit=dict(host=socket.gethostname(),completed=False,videos=[],model_load_seconds=None)
    attempt=1
    while (out/f'pipeline_attempt_{attempt:04d}.json').exists():attempt+=1
    try:
        torch.manual_seed(0);start=time.perf_counter();model=Recognizer();audit['model_load_seconds']=time.perf_counter()-start
        for number,row in enumerate(selected_rows(args.smoke),1):
            ds,vid=row['dataset'],row['video_id'];folder=CACHE/ds/vid;folder.mkdir(parents=True,exist_ok=True)
            path=folder/'metadata.json';entry=dict(dataset=ds,video_id=vid,reused=path.exists());video_start=time.perf_counter()
            if path.exists():record=json.loads(path.read_text());validate_source(model.processor,row,record,folder)
            else:
                first=dict(model.counts);start=clock();samples,observed,timeline=decode_audio(resolve_video(row),float(row['duration']));decode_seconds=clock()-start
                start=clock();np.save(folder/'audio.npy',samples,allow_pickle=False);np.save(folder/'observed.npy',observed,allow_pickle=False);proof_seconds=clock()-start
                language_id=language=None;language_seconds=0.;blocks=[];words=[]
                (folder/'alignment').mkdir(exist_ok=True)
                for i,(a,b) in enumerate(fixed_windows(float(row['duration']),SPEC['source_audio_seconds'])):
                    audio,crop=crop_audio(samples,observed,a,b);block=dict(i=i,nominal=[a,b],crop=crop,generation=None,seconds=0.)
                    if len(audio):
                        if language_id is None:
                            start=clock();language_id,language=model.language(audio);language_seconds=clock()-start
                        start=clock();generation,matrix=model.block(audio,language_id,language,*crop['actual_interval'],i)
                        block['generation']=generation;block['seconds']=clock()-start
                        start=clock()
                        if generation['content_tokens']:np.save(folder/'alignment'/f'block_{i:06d}.npy',matrix,allow_pickle=False)
                        proof_seconds+=clock()-start;words.extend(dict(word) for word in generation['words'])
                    blocks.append(block)
                for i,word in enumerate(words):word['id']=i
                p=partition(words)
                prefix=[] if language_id is None else [model.model.generation_config.decoder_start_token_id,language_id,
                    model.model.generation_config.task_to_id['transcribe'],model.model.generation_config.no_timestamps_token_id]
                record=dict(version=SPEC['version'],spec=SPEC,host=socket.gethostname(),GT_read=False,dataset=ds,video_id=vid,
                    duration=float(row['duration']),manifest_video_path=row['video_path'],input_video=str(resolve_video(row)),timeline=timeline,
                    prefix_tokens=prefix,language_id=language_id,language=language,blocks=blocks,words=words,partition=p,
                    windows=[scope(words,p,a,b) for a,b in fixed_windows(float(row['duration']),8)],
                    counts={key:model.counts[key]-first[key] for key in first},decode_seconds=decode_seconds,
                    language_seconds=language_seconds,proof_write_seconds=proof_seconds,
                    source_seconds=decode_seconds+language_seconds+sum(x['seconds'] for x in blocks)+proof_seconds)
                validate_start=time.perf_counter();validate_source(model.processor,row,record,folder,decode=False);entry['validation_seconds']=time.perf_counter()-validate_start
                partial=path.with_suffix('.partial');partial.write_text(json.dumps(record,ensure_ascii=False)+'\n');partial.replace(path)
            entry['elapsed_seconds']=time.perf_counter()-video_start;audit['videos'].append(entry)
            logging.info('%d/%d %s/%s words=%d segments=%d %.2fs',number,len(selected_rows(args.smoke)),ds,vid,len(record['words']),len(record['partition']['segments']),record['source_seconds'])
        audit['completed']=True;logging.info('DONE source coverage=%d',len(audit['videos']))
    finally:
        audit['elapsed_seconds']=time.perf_counter()-start_pipeline;(out/f'pipeline_attempt_{attempt:04d}.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
