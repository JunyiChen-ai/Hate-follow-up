"""Paired complete reader measurement, same actual video source host, no labels."""
import argparse
import json
import logging
import os
import socket
import time
import torch
from inputs import ROOT,SPEC,DATASETS,selected_rows,source,validate_source
from reader import read_video,IMPLEMENTATION,REVISION
from validate import validate_bundle
from src.video_inputs import load_asr
from src.mllm_judge import Judge,MODEL


def main():
    start_pipeline=time.perf_counter();ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    out=ROOT/'runs/20261006_m1_texttiling'/(f'r{REVISION}_full_'+('smoke' if args.smoke else 'main'));out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers,av
    config=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,spec=SPEC,implementation=IMPLEMENTATION,
        GT_read=False,smoke=args.smoke,torch=torch.__version__,transformers=transformers.__version__,av=av.__version__,
        code='experiments/20261006_m1_texttiling/{reader,measure,compiler,validate}.py;src/native_input_binding.py;2026-10-06')
    if REVISION==2:config['reader_revision']=2
    path=out/'config.json'
    if path.exists():assert json.loads(path.read_text())==config,'changed reader configuration; refuse overwrite'
    else:path.write_text(json.dumps(config,indent=2)+'\n')
    audit=dict(host=socket.gethostname(),completed=False,model_load_seconds=None,videos=[]);hooks=[]
    attempt=1
    while (out/f'pipeline_attempt_{attempt:04d}.json').exists():attempt+=1
    try:
        from transformers import AutoProcessor
        whisper=AutoProcessor.from_pretrained(SPEC['asr_model'],local_files_only=True)
        torch.manual_seed(0);start=time.perf_counter();j=Judge(MODEL);audit['model_load_seconds']=time.perf_counter()-start
        j.forward_calls=j.vision_calls=0
        hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),
            j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
        asr={ds:load_asr(ds) for ds in DATASETS};results={name:[] for name in ('base','optimized')};rows=selected_rows(args.smoke)
        for number,row in enumerate(rows,1):
            video_start=time.perf_counter();entry=dict(dataset=row['dataset'],video_id=row['video_id']);segments=asr[row['dataset']].get(row['video_id'],[])
            meta,folder=source(row);assert meta['host']==socket.gethostname(),'source host differs; no pilot cache splice'
            start=time.perf_counter();validate_source(whisper,row,meta,folder);entry['source_validation_seconds']=time.perf_counter()-start
            path=out/'records'/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
            entry['reused']=path.exists()
            if path.exists():bundle=json.loads(path.read_text())
            else:
                start=time.perf_counter();bundle=read_video(j,row,segments,meta,out,args.smoke);entry['reader_seconds']=time.perf_counter()-start
                validate_bundle(j,row,segments,meta,bundle,args.smoke)
                start=time.perf_counter();partial=path.with_suffix('.partial');partial.write_text(json.dumps(bundle,ensure_ascii=False)+'\n');partial.replace(path);entry['record_write_seconds']=time.perf_counter()-start
            start=time.perf_counter();validate_bundle(j,row,segments,meta,bundle,args.smoke);entry['bundle_validation_seconds']=time.perf_counter()-start
            for name in results:results[name].append(bundle[name])
            entry['elapsed_seconds']=time.perf_counter()-video_start;audit['videos'].append(entry)
            logging.info('%d/%d %s/%s compiled=%d %.2fs',number,len(rows),row['dataset'],row['video_id'],sum(p['reason']=='compiled' for p in bundle['packets']),bundle['optimized']['extra']['standalone_seconds'])
        for name,records in results.items():
            folder=out/name;folder.mkdir(exist_ok=True);(folder/'config.json').write_text(json.dumps(dict(config,measurement=name),indent=2)+'\n')
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        audit['completed']=True;logging.info('DONE coverage=%d',len(rows))
    finally:
        for hook in hooks:hook.remove()
        audit['elapsed_seconds']=time.perf_counter()-start_pipeline;(out/f'pipeline_attempt_{attempt:04d}.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
