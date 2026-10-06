"""Complete native/source/new-V paired measurement; annotations never loaded."""
import argparse
import copy
import json
import logging
import math
import os
import socket
import sys
import time
import numpy as np
import torch
from inputs import ROOT,SPEC,CACHE,DATASETS,selected_rows,source_frames,windows_for,validate_input
from source_frames import owned_source_ids
from memory import VideoMemory
from reader import collect_source,visual_margin,clock
from src.mllm_judge import Judge,MODEL,yesno_question
from src.stance_cache import build,margin
from src.video_inputs import frame_paths,load_asr
from binding import save_native_binding,IMPLEMENTATION


def prediction(row,ctx,visual,speech,seconds,method,source_calls=0):
    windows=[]
    for window,v,s in zip(windows_for(row,[]),visual,speech):
        record=dict(i=window['i'],start=window['start'],end=window['end'],z_visual=v,z=max(v,s) if s is not None else v)
        if s is not None:record['z_speech']=s
        windows.append(record)
    idx=np.clip(((np.arange(math.ceil(float(row['duration'])*4))+.5)/4//SPEC['window_seconds']).astype(int),0,len(windows)-1)
    return dict(schema_version=1,method=method,dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),native_rate=4,
        score_curve=np.asarray([w['z'] for w in windows])[idx].tolist(),intervals=[],error=None,seed=0,
        calls=3+len(windows)+sum(s is not None for s in speech)+source_calls,code_path='experiments/20261006_m1_rekv/measure.py',
        extra=dict(z_video=ctx['global_margin'],stance=ctx['stance'],prefix_tokens=ctx['prefix_tokens'],
            stance_cache_tokens=ctx['stance_cache_tokens'],stance_cache_logical_start=ctx['stance_cache_logical_start'],windows=windows,standalone_seconds=seconds))


@torch.no_grad()
def read_video(j,row,segments,meta,out,smoke,source_transform=None,previous_frames=None,selector_factory=None,after_read=None):
    start=clock(j);first=j.forward_calls;first_vision=j.vision_calls
    if j.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    cache,ctx=build(j,frame_paths(row['dataset'],row['video_id'],SPEC['native_requested_frames']),segments)
    times=dict(prefix=clock(j)-start,native_visual=0.,native_speech=0.,source=0.,new_visual=0.,diagnostic=0.)
    windows=windows_for(row,segments);visual=[];speech=[]
    for w in windows:
        start=clock(j);visual.append(margin(j,cache,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')));times['native_visual']+=clock(j)-start
        if w['body'].strip():
            start=clock(j);s=margin(j,cache,ctx,yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'speech'));times['native_speech']+=clock(j)-start
        else:s=None
        speech.append(s)
    proof=out/'proof'/row['dataset']/row['video_id'];proof.mkdir(parents=True,exist_ok=True)
    start=clock(j);binding=save_native_binding(j,row,segments,ctx,proof/'native_pixels.npy');binding_seconds=clock(j)-start
    temporary_folder=out/'temporary'/row['dataset']/row['video_id']
    if temporary_folder.exists() and any(temporary_folder.iterdir()):
        # Preserve incomplete evidence, then rebuild this same video from input.
        failed=out/'failed_temporary'/row['dataset'];failed.mkdir(parents=True,exist_ok=True)
        number=1
        while (failed/(row['video_id']+f'_{number:04d}')).exists():number+=1
        destination=failed/(row['video_id']+f'_{number:04d}')
        temporary_folder.rename(destination)
        logging.info('Preserved interrupted denseKV at %s; rebuilding same video',destination)
    memory=VideoMemory(temporary_folder)
    new_visual=[];traces=[];clones=0;source=source_frames(row,meta['source'])
    if source_transform is not None:source=source_transform(source)
    try:
        acquisition=collect_source(j,cache,ctx,memory,source,previous_frames=previous_frames);times['source']=acquisition['seconds']
        representatives=memory.representative_layers('cpu').numpy() if source else np.zeros((len(cache.layers),0,cache.layers[0].keys.shape[1]*cache.layers[0].keys.shape[-1]),np.float32)
        proof_start=clock(j);representative_path=proof/'source_keys.npy';np.save(representative_path,representatives,allow_pickle=False)
        proof_write_seconds=clock(j)-proof_start
        queries=np.zeros((len(windows),representatives.shape[0],representatives.shape[2]),np.float32)
        for w,v,s in zip(windows,visual,speech):
            local=owned_source_ids(meta['source'],(w['start'],w['end']))
            question=yesno_question(w['i'],len(windows),w['start'],w['end'],w['body'],'visual')
            start=clock(j)
            if local:
                selector=selector_factory(w,memory,local) if selector_factory is not None else None
                z,details=visual_margin(j,cache,ctx,memory,local,question,selection=selector)
                layer_records=[]
                for layer in range(len(cache.layers)):
                    record=details['layers'][layer]
                    queries[w['i'],layer]=record['query_vector'].numpy()
                    layer_records.append({key:value for key,value in record.items() if key not in ('query_vector','similarities')})
                branch=dict(input=details['input'],layers=layer_records)
            else:z=margin(j,cache,ctx,question);branch=None;assert z==v
            times['new_visual']+=clock(j)-start;new_visual.append(z)
            trace=dict(i=w['i'],bounds=[w['start'],w['end']],body=w['body'],question=question,local_ids=local,
                native_visual=v,native_speech=s,new_visual=z,branch=branch)
            if smoke and clones==0:
                start=clock(j);clone=copy.deepcopy(cache)
                if local:
                    replay_selector=selector_factory(w,memory,local) if selector_factory is not None else None
                    replayed,other=visual_margin(j,clone,ctx,memory,local,question,selection=replay_selector);assert other['input']==details['input']
                else:replayed=margin(j,clone,ctx,question)
                assert replayed==z and clone.get_seq_length()==ctx['stance_cache_tokens']
                assert all(torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values) for a,b in zip(cache.layers,clone.layers))
                trace['clone_exact']=True;clones+=1;times['diagnostic']+=clock(j)-start;del clone
            traces.append(trace)
        proof_start=clock(j);query_path=proof/'question_vectors.npy';np.save(query_path,queries,allow_pickle=False)
        proof_write_seconds+=clock(j)-proof_start
        source_metadata=memory.metadata()
        source_seconds=meta['source']['decode_seconds']+times['source']+binding_seconds
        base=prediction(row,ctx,visual,speech,sum(times[k] for k in ('prefix','native_visual','native_speech')),'m1_native')
        optimized=prediction(row,ctx,new_visual,speech,source_seconds+sum(times[k] for k in ('prefix','new_visual','native_speech')),'m1_rekv',len(source))
        checks=dict(GT_read=False,host=socket.gethostname(),times=times,source_seconds=source_seconds,decode_seconds=meta['source']['decode_seconds'],
            source_forwards=len(source),source_vision=len(source),actual_forwards=j.forward_calls-first,actual_vision=j.vision_calls-first_vision,
            diagnostic_forwards=clones,peak_GiB=torch.cuda.max_memory_allocated()/2**30 if j.device.type=='cuda' else 0.,
            native_binding_seconds=binding_seconds,
            proof_write_seconds=proof_write_seconds,
            source_frames=len(source),missing_local_windows=sum(not t['local_ids'] for t in traces),
            remote_layer_reads=sum(bool(layer['remote_ids']) for t in traces if t['branch'] for layer in t['branch']['layers']),
            temporary_dense_bytes=sum(block['storage_bytes'] for block in source_metadata))
        assert checks['actual_forwards']==base['calls']+len(source)+len(windows)+clones
        assert checks['actual_vision']==1+len(source)
        native={key:value for key,value in ctx.items() if key not in ('positions','rope','files')}
        result=dict(version=SPEC['version'],spec=SPEC,base=base,optimized=optimized,checks=checks,traces=traces,
            binding=binding,
            native_ctx=native,native_rope=ctx['rope'].tolist(),source_blocks=source_metadata,source_acquisition=acquisition,
            representative_path=str(representative_path.relative_to(ROOT)),query_path=str(query_path.relative_to(ROOT)),segments=[list(s) for s in segments])
        if after_read is not None:result['controls']=after_read(j,cache,ctx,memory,result)
        return result,memory
    except Exception:
        memory.close(release=False)
        raise


def main():
    pipeline_start=time.perf_counter()
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    out=ROOT/'runs/20261006_m1_rekv'/('r1_full_smoke' if args.smoke else 'r1_full_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers,av
    config=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,spec=SPEC,implementation=IMPLEMENTATION,GT_read=False,smoke=args.smoke,
        torch=torch.__version__,transformers=transformers.__version__,av=av.__version__,
        source='experiments/20261006_m1_rekv/{measure,reader,attention,memory}.py;2026-10-06',command='python -u '+' '.join(sys.argv))
    config_path=out/'config.json'
    audit=dict(host=socket.gethostname(),date=config['date'],model_load_seconds=None,completed=False,
        scope='actual current-attempt pipeline wall time incl setup/validation/proof and records; pairednative/diagnostics retained',videos=[])
    attempt=1
    while (out/f'pipeline_attempt_{attempt:04d}.json').exists():attempt+=1
    audit_path=out/f'pipeline_attempt_{attempt:04d}.json'
    hooks=[]
    try:
        if config_path.exists():
            previous=json.loads(config_path.read_text())
            for key in ('host','model','spec','implementation','GT_read','smoke','torch','transformers','av'):
                assert previous[key]==config[key],'run configuration changed; refuse overwrite/resume'
        else:config_path.write_text(json.dumps(config,indent=2)+'\n')
        torch.manual_seed(0);model_start=time.perf_counter();j=Judge(MODEL)
        audit['model_load_seconds']=time.perf_counter()-model_start;j.forward_calls=j.vision_calls=0
        hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),
            j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
        asr={dataset:load_asr(dataset) for dataset in DATASETS};results={name:[] for name in ('base','optimized')}
        from validate import validate_bundle
        rows=selected_rows(args.smoke)
        for number,row in enumerate(rows,1):
            video_start=time.perf_counter();entry=dict(dataset=row['dataset'],video_id=row['video_id'])
            segments=asr[row['dataset']].get(row['video_id'],[])
            validation_start=time.perf_counter();meta=json.loads((CACHE/row['dataset']/row['video_id']/'metadata.json').read_text());validate_input(meta,row)
            entry['input_validation_seconds']=time.perf_counter()-validation_start
            path=out/'records'/row['dataset']/(row['video_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
            memory=None
            try:
                if path.exists():bundle=json.loads(path.read_text());entry['reused_completed_record']=True
                else:
                    entry['reused_completed_record']=False;reader_start=time.perf_counter()
                    bundle,memory=read_video(j,row,segments,meta,out,args.smoke)
                    entry['reader_pipeline_seconds']=time.perf_counter()-reader_start
                    validation_start=time.perf_counter()
                    validate_bundle(j,row,segments,meta,bundle,args.smoke)
                    entry['first_bundle_validation_seconds']=time.perf_counter()-validation_start
                    write_start=time.perf_counter();partial=path.with_suffix('.partial');partial.write_text(json.dumps(bundle)+'\n');partial.replace(path)
                    entry['record_write_seconds']=time.perf_counter()-write_start
                validation_start=time.perf_counter()
                validate_bundle(j,row,segments,meta,bundle,args.smoke)
                entry['current_bundle_validation_seconds']=time.perf_counter()-validation_start
            finally:
                if memory is not None:memory.close(release=path.exists())
            for name in results:results[name].append(bundle[name])
            entry['video_pipeline_seconds']=time.perf_counter()-video_start;audit['videos'].append(entry)
            audit['elapsed_seconds']=time.perf_counter()-pipeline_start;audit_path.write_text(json.dumps(audit,indent=2)+'\n')
            logging.info('%d/%d %s/%s %.2fs',number,len(rows),row['dataset'],row['video_id'],bundle['optimized']['extra']['standalone_seconds'])
        for name,records in results.items():
            folder=out/name;folder.mkdir(exist_ok=True);(folder/'config.json').write_text(json.dumps({**config,'measurement':name},indent=2)+'\n')
            (folder/'predictions.jsonl').write_text(''.join(json.dumps(record)+'\n' for record in records))
        logging.info('DONE coverage=%d',len(rows))
        audit['completed']=True
    finally:
        for hook in hooks:hook.remove()
        audit['elapsed_seconds']=time.perf_counter()-pipeline_start
        audit_path.write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':main()
