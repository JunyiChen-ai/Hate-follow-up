"""Actual media, same-model literal observations, pixel occurrences; no GT."""
import argparse
import json
import logging
from pathlib import Path
import socket
import time
import numpy as np
from PIL import Image
import torch
from interface import SPEC,ROOT,content,writer,compile_records
from inputs import DATASETS,CACHE,selected_rows,windows_for,samples,save_rgb
from controller import Controller,interval_lookup
from tracking import crop
from src.actual_video_frames import acquire_window_frames,validate_frames
from src.structured_source_generation import generate,validate_generation
from src.mllm_judge import Judge,MODEL
from src.mllm_renderer import cpu_renderer


def observation(j,frames,kind,window):
    items,paths=content(frames,kind,window)
    output_frames=frames if kind=='initial' else frames[:1]
    system=SPEC['initial_system' if kind=='initial' else 'repair_system']
    cap=SPEC['initial_max_tokens' if kind=='initial' else 'repair_max_tokens']
    g=generate(j,ROOT,system,items,paths,cap,writer(output_frames),
        field_tokens=SPEC['text_tokens'],description_words=SPEC['text_words'])
    return dict(generation=g,compiled=compile_records(g,output_frames))


def anchor_map(windows,initial):
    result={}
    for w,r in zip(windows,initial):
        assert len(w['frames'])==len(r['compiled'])
        for f,o in zip(w['frames'],r['compiled']):
            result.setdefault(f['index'],[]).append((o,f'initial:{w["i"]}:{f["id"]}'))
    return result


def repair_frames(folder,entry,rgb,failure,number):
    h,w=rgb.shape[:2];box=failure['evidence'].get('candidate_box',failure['old_box'])
    x0=max(0,min(int(box[0]),w-1));y0=max(0,min(int(box[1]),h-1))
    x1=max(x0+1,min(int(box[2]),w));y1=max(y0+1,min(int(box[3]),h))
    hint_box=[x0,y0,x1,y1]
    full=folder/'repairs'/f'r{number:06d}_frame_{entry["index"]:08d}.png'
    hint=folder/'repairs'/f'r{number:06d}_candidate.png'
    frames=[dict(id='p0',index=entry['index'],time=entry['time'],shape=[w,h],path=str(full.relative_to(ROOT))),
        dict(id='candidate',index=entry['index'],time=entry['time'],shape=[x1-x0,y1-y0],path=str(hint.relative_to(ROOT)))]
    return frames,hint_box


def acquire(j,row,folder):
    if j.device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();source=acquire_window_frames(row,folder/'frames',SPEC['window_seconds'])
    windows=windows_for(row,source,folder);initial=[]
    for w in windows:
        initial.append(observation(j,w['frames'],'initial',(w['start'],w['end'])) if w['frames'] else dict(generation=None,compiled=[]))
    repairs=[]
    def observe(w,entry,rgb,failure,number):
        frames,box=repair_frames(folder,entry,rgb,failure,number)
        save_rgb(rgb,ROOT/frames[0]['path']);save_rgb(crop(rgb,box),ROOT/frames[1]['path'])
        result=observation(j,frames,'repair',(windows[w]['start'],windows[w]['end']))
        identity=len(repairs);repairs.append(dict(window=w,entry=entry,failed_id=failure['id'],number=number,
            frames=frames,hint_box=box,**result))
        return result['compiled'][0],f'repair:{identity}'
    controller=Controller(folder,windows,observe);anchors=anchor_map(windows,initial)
    for entry,rgb in samples(row,source):controller.step(entry,rgb,anchors.get(entry['index'],[]))
    occurrences=controller.finish();gens=[r['generation'] for r in initial+repairs if r['generation'] is not None]
    return dict(version=SPEC['version'],spec=SPEC,model=MODEL,host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),
        dataset=row['dataset'],video_id=row['video_id'],GT_read=False,source=source,windows=windows,
        peak_GiB=torch.cuda.max_memory_allocated()/2**30 if j.device.type=='cuda' else 0.,
        initial=initial,repair_observations=repairs,occurrences=occurrences,
        lookups=[interval_lookup(occurrences['events'],w) for w in windows],
        cost=dict(standalone_seconds=time.perf_counter()-start,decode_seconds=source['decode_seconds'],
            generation_seconds=sum(g['seconds'] for g in gens),controller_seconds=occurrences['processing_seconds'],
            generation_calls=len(gens),initial_calls=sum(r['generation'] is not None for r in initial),repair_calls=len(repairs),
            actual_forwards=sum(g['actual_forwards'] for g in gens),actual_vision_forwards=sum(g['actual_vision_forwards'] for g in gens)))


def validate_observation(j,r,frames,kind,window):
    if not frames:assert r==dict(generation=None,compiled=[]);return
    items,paths=content(frames,kind,window);output_frames=frames if kind=='initial' else frames[:1]
    system=SPEC['initial_system' if kind=='initial' else 'repair_system']
    cap=SPEC['initial_max_tokens' if kind=='initial' else 'repair_max_tokens']
    validate_generation(j,ROOT,r['generation'],system,items,paths,cap,writer(output_frames),
        field_tokens=SPEC['text_tokens'],description_words=SPEC['text_words'])
    assert r['compiled']==compile_records(r['generation'],output_frames)


def same_pixels(rgb,path):
    with Image.open(path) as image:assert image.format=='PNG' and np.array_equal(np.asarray(image.convert('RGB')),rgb)


def validate(j,m,row,folder):
    assert m['version']==SPEC['version'] and m['spec']==SPEC and m['model']==MODEL and m['GT_read'] is False
    assert (m['dataset'],m['video_id'])==(row['dataset'],row['video_id'])
    validate_frames(m['source'],row,folder/'frames');windows=windows_for(row,m['source'],folder);assert windows==m['windows']
    for w,r in zip(windows,m['initial']):validate_observation(j,r,w['frames'],'initial',(w['start'],w['end']))
    pending=iter(m['repair_observations']);seen=[]
    def observe(w,entry,rgb,failure,number):
        r=next(pending);frames,box=repair_frames(folder,entry,rgb,failure,number)
        assert r['window']==w and r['entry']==entry and r['number']==number and r['failed_id']==failure['id']
        assert r['frames']==frames and r['hint_box']==box
        same_pixels(rgb,ROOT/frames[0]['path']);same_pixels(crop(rgb,box),ROOT/frames[1]['path'])
        validate_observation(j,r,frames,'repair',(windows[w]['start'],windows[w]['end']))
        identity=len(seen);seen.append(r);return r['compiled'][0],f'repair:{identity}'
    controller=Controller(folder,windows,observe,save=same_pixels);anchors=anchor_map(windows,m['initial'])
    for entry,rgb in samples(row,m['source']):controller.step(entry,rgb,anchors.get(entry['index'],[]))
    replay=controller.finish();assert next(pending,None) is None
    assert {k:v for k,v in replay.items() if k!='processing_seconds'}=={k:v for k,v in m['occurrences'].items() if k!='processing_seconds'}
    assert m['lookups']==[interval_lookup(replay['events'],w) for w in windows]
    gens=[r['generation'] for r in m['initial']+seen if r['generation'] is not None];cost=m['cost']
    assert cost['generation_calls']==len(gens) and cost['repair_calls']==len(seen)
    assert cost['actual_forwards']==sum(g['actual_forwards'] for g in gens)
    assert cost['actual_vision_forwards']==sum(g['actual_vision_forwards'] for g in gens)
    assert len(seen)<=2*len(windows) and all(np.isfinite(v) and v>=0 for v in cost.values())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261005_m1_text_tracking'/('source_smoke' if a.smoke else 'source_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    import transformers
    config=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),model=MODEL,spec=SPEC,GT_read=False,smoke=a.smoke,
        torch=torch.__version__,transformers=transformers.__version__,code='experiments/20261005_m1_text_tracking/{extract,interface,inputs,tracking,controller}.py; sources2026-10-05',command='python -u '+' '.join(__import__('sys').argv))
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    torch.manual_seed(0);torch.set_num_threads(4);j=Judge(MODEL);j.forward_calls=j.vision_calls=0
    hooks=[j.model.model.language_model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),
        j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    rows=selected_rows(a.smoke)
    for number,row in enumerate(rows,1):
        folder=CACHE/row['dataset']/row['video_id'];folder.mkdir(parents=True,exist_ok=True);path=folder/'metadata.json'
        if path.exists():m=json.loads(path.read_text())
        else:
            m=acquire(j,row,folder);validate(j,m,row,folder);partial=path.with_suffix('.partial');partial.write_text(json.dumps(m,allow_nan=False)+'\n');partial.replace(path)
        validate(j,m,row,folder)
        logging.info('%d/%d %s/%s %.2fs occurrences=%d repairs=%d',number,len(rows),row['dataset'],row['video_id'],m['cost']['standalone_seconds'],len(m['occurrences']['events']),m['cost']['repair_calls'])
    for hook in hooks:hook.remove()
    (CACHE/'PROVENANCE.md').write_text('# Pixel-supported text occurrences\n\nGenerated by experiments/20261005_m1_text_tracking/extract.py and shared actual_video_frames/structured_source_generation; R1 sources2026-10-05, Qwen/Qwen3-VL-8B-Instruct, opencv-python-headless4.13.0.92. Each metadata records generating host/date, all original readable paths, real PTS/shape, immutable common spec, current image tokens, generation calls/cost, geometry and supported endpoint crop paths. No GT used, no content checksums. Four-fps pixel tracking is source acquisition; evaluation remains canonical4fps. Source decoding/OCR/reobservations/tracking/PNG endpoint preparation included in standalone seconds; audit replay/serialization separately visible in complete Slurm wall.\n')
    logging.info('EXTRACTION_DONE coverage=%d',len(rows))


if __name__=='__main__':main()
