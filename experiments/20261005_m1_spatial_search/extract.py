"""Target discovery, actual-region search feedback and original-pixel crop memory."""
import argparse
import heapq
import json
import logging
import os
from pathlib import Path
import socket
import time
import numpy as np
from PIL import Image
import torch
from inputs import ROOT,SPEC,CACHE,DATASETS,selected_rows,windows_for
from interface import write_detector,write_search,detector_record,search_record,detector_content,search_content
from geometry import map_box,priority_children
from src.actual_video_frames import acquire_window_frames,validate_frames
from src.structured_source_generation import generate,validate_generation
from src.source_generation import clock
from src.mllm_judge import Judge,MODEL
from src.video_inputs import load_asr


def source_node_path(folder,window,step,frame,node):
    if node['box']==[0,0,*frame['shape']]:return ROOT/frame['path']
    return folder/'nodes'/f'w{window["i"]:06d}_step{step:02d}.png'


def save_pixels(original,box,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    crop=original.crop(tuple(box));crop.save(path);crop.close()


def check_pixels(original,box,path):
    with Image.open(path) as actual:
        expected=original.crop(tuple(box))
        try:assert np.array_equal(np.asarray(actual.convert('RGB')),np.asarray(expected))
        finally:expected.close()


@torch.no_grad()
def acquire(j,row,segments,folder):
    start=clock(j);before=j.forward_calls;before_vision=j.vision_calls;torch.cuda.reset_peak_memory_stats()
    source=acquire_window_frames(row,folder/'frames');windows=windows_for(row,segments,source,folder);results=[]
    for w in windows:
        frame_ids=[f['id'] for f in w['frames']]
        if not frame_ids:
            results.append(dict(window=w,detector=None,decision=dict(kind='UNKNOWN',reason='no_actual_LOCAL_frame'),steps=[],found=None));continue
        content,paths=detector_content(w,segments)
        g=generate(j,ROOT,SPEC['detector_system'],content,paths,SPEC['detector_tokens'],lambda s:write_detector(s,frame_ids),
            field_tokens=SPEC['description_tokens'],description_words=SPEC['description_words'])
        decision=detector_record(g,frame_ids);steps=[];found=None
        if decision['kind']=='SEARCH':
            frame=next(f for f in w['frames'] if f['id']==decision['frame']);serial=1
            queue=[(-1,0,dict(box=[0,0,*frame['shape']],priority=-1,serial=0,parent=None))]
            with Image.open(ROOT/frame['path']) as raw:
                original=raw.convert('RGB')
                try:
                    while queue and len(steps)<SPEC['max_search_nodes']:
                        _,_,node=heapq.heappop(queue);number=len(steps)
                        path=source_node_path(folder,w,number,frame,node)
                        if path!=ROOT/frame['path']:save_pixels(original,node['box'],path)
                        content,paths=search_content(w,frame,decision['target'],node,str(path.relative_to(ROOT)))
                        sg=generate(j,ROOT,SPEC['search_system'],content,paths,SPEC['search_tokens'],lambda s:write_search(s,node['box']),
                            field_tokens=SPEC['description_tokens'],description_words=SPEC['description_words'])
                        result=search_record(sg,node['box']);steps.append(dict(node=node,path=str(path.relative_to(ROOT)),generation=sg,record=result))
                        if result['kind']=='FOUND':
                            box=map_box(node['box'],result['box']);crop=folder/'crops'/f'w{w["i"]:06d}.png';save_pixels(original,box,crop)
                            found=dict(frame=frame['id'],time=frame['time'],original_shape=frame['shape'],original_pixel_box=box,
                                path=str(crop.relative_to(ROOT)),parent_node=number);break
                        if result['kind']=='UNKNOWN':break
                        for child in priority_children(node['box'],result['order'],serial):
                            child['parent']=number;heapq.heappush(queue,(child['priority'],child['serial'],child))
                        serial+=4
                finally:original.close()
        results.append(dict(window=w,detector=g,decision=decision,steps=steps,found=found))
    gens=[g for r in results for g in ([r['detector']] if r['detector'] else [])+[s['generation'] for s in r['steps']]]
    return dict(version=SPEC['version'],spec=SPEC,model=MODEL,host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),GT_read=False,
        dataset=row['dataset'],video_id=row['video_id'],duration=float(row['duration']),segments=[list(s) for s in segments],
        source=source,windows=results,actual_forwards=j.forward_calls-before,actual_vision_forwards=j.vision_calls-before_vision,
        generation_seconds=sum(g['seconds'] for g in gens),standalone_seconds=clock(j)-start,peak_GiB=torch.cuda.max_memory_allocated()/2**30)


def validate(m,row,segments,j):
    assert (m['version'],m['spec'],m['model'])==(SPEC['version'],SPEC,MODEL)
    assert (m['dataset'],m['video_id'],m['duration'])==(row['dataset'],row['video_id'],float(row['duration']))
    assert m['GT_read'] is False and m['segments']==[list(s) for s in segments]
    folder=CACHE/row['dataset']/row['video_id'];validate_frames(m['source'],row,folder/'frames')
    windows=windows_for(row,segments,m['source'],folder);assert len(m['windows'])==len(windows);gens=[]
    for w,r in zip(windows,m['windows']):
        assert r['window']==w;ids=[f['id'] for f in w['frames']]
        if not ids:
            assert r==dict(window=w,detector=None,decision=dict(kind='UNKNOWN',reason='no_actual_LOCAL_frame'),steps=[],found=None);continue
        content,paths=detector_content(w,segments)
        validate_generation(j,ROOT,r['detector'],SPEC['detector_system'],content,paths,SPEC['detector_tokens'],lambda s:write_detector(s,ids),
            field_tokens=SPEC['description_tokens'],description_words=SPEC['description_words'])
        decision=detector_record(r['detector'],ids);assert decision==r['decision'];gens.append(r['detector'])
        if decision['kind']!='SEARCH':assert not r['steps'] and r['found'] is None;continue
        frame=next(f for f in w['frames'] if f['id']==decision['frame']);serial=1
        queue=[(-1,0,dict(box=[0,0,*frame['shape']],priority=-1,serial=0,parent=None))];found=None;stopped=False
        with Image.open(ROOT/frame['path']) as raw:
            original=raw.convert('RGB')
            try:
                for number,step in enumerate(r['steps']):
                    assert not stopped and queue and number<SPEC['max_search_nodes']
                    _,_,node=heapq.heappop(queue);assert node==step['node']
                    path=source_node_path(folder,w,number,frame,node);assert str(path.relative_to(ROOT))==step['path'];check_pixels(original,node['box'],path)
                    content,paths=search_content(w,frame,decision['target'],node,step['path'])
                    validate_generation(j,ROOT,step['generation'],SPEC['search_system'],content,paths,SPEC['search_tokens'],lambda s:write_search(s,node['box']),
                        field_tokens=SPEC['description_tokens'],description_words=SPEC['description_words'])
                    result=search_record(step['generation'],node['box']);assert result==step['record'];gens.append(step['generation'])
                    if result['kind']=='FOUND':
                        box=map_box(node['box'],result['box']);crop=folder/'crops'/f'w{w["i"]:06d}.png';check_pixels(original,box,crop)
                        found=dict(frame=frame['id'],time=frame['time'],original_shape=frame['shape'],original_pixel_box=box,path=str(crop.relative_to(ROOT)),parent_node=number);stopped=True
                    elif result['kind']=='UNKNOWN':stopped=True
                    else:
                        for child in priority_children(node['box'],result['order'],serial):
                            child['parent']=number;heapq.heappush(queue,(child['priority'],child['serial'],child))
                        serial+=4
                assert r['steps'] and (stopped or not queue or len(r['steps'])==SPEC['max_search_nodes'])
                assert found==r['found']
            finally:original.close()
    assert m['actual_forwards']==sum(g['actual_forwards'] for g in gens) and m['actual_vision_forwards']==sum(g['actual_vision_forwards'] for g in gens)
    assert abs(m['generation_seconds']-sum(g['seconds'] for g in gens))<1e-6
    assert m['standalone_seconds']>=m['generation_seconds']+m['source']['decode_seconds']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    out=ROOT/'runs/20261005_m1_spatial_search'/('r1_extract_smoke' if a.smoke else 'r1_extract_main');out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(os.getpid()))
    import transformers
    cfg=dict(host=socket.gethostname(),date=time.strftime('%Y-%m-%d'),spec=SPEC,model=MODEL,GT_read=False,smoke=a.smoke,
        torch=torch.__version__,transformers=transformers.__version__,code='experiments/20261005_m1_spatial_search/{geometry,inputs,interface,extract}.py + stable sources;2026-10-05',command='python -u '+' '.join(__import__('sys').argv))
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');torch.manual_seed(0);j=Judge(MODEL);j.forward_calls=j.vision_calls=0
    hooks=[j.model.model.register_forward_pre_hook(lambda *_:setattr(j,'forward_calls',j.forward_calls+1)),j.model.model.visual.register_forward_pre_hook(lambda *_:setattr(j,'vision_calls',j.vision_calls+1))]
    asr={ds:load_asr(ds) for ds in DATASETS};rows=selected_rows(a.smoke)
    for number,row in enumerate(rows,1):
        folder=CACHE/row['dataset']/row['video_id'];folder.mkdir(parents=True,exist_ok=True);path=folder/'metadata.json';segments=asr[row['dataset']].get(row['video_id'],[])
        if path.exists():m=json.loads(path.read_text())
        else:
            m=acquire(j,row,segments,folder);validate(m,row,segments,j);temporary=folder/'metadata.partial';temporary.write_text(json.dumps(m)+'\n');temporary.replace(path)
        validate(m,row,segments,j)
        logging.info('%d/%d %s/%s %.2fs search=%d found=%d',number,len(rows),row['dataset'],row['video_id'],m['standalone_seconds'],sum(r['decision']['kind']=='SEARCH' for r in m['windows']),sum(r['found'] is not None for r in m['windows']))
    for hook in hooks:hook.remove()
    (CACHE/'PROVENANCE.md').write_text('# Actual spatial-search sources\n\n'+cfg['code']+', '+cfg['date']+'; '+cfg['command']+' on '+cfg['host']+'. Model '+MODEL+'. Actualdecoded PTS/originalpaths/shapes/pixels, originalASR, prompts/tokens/grammar, allnodes/coordinatechains/ROI/costs/host per metadata. NoGT/contentchecksums. Newvideo source costs charged even on reuse.\n')
    logging.info('EXTRACTION_DONE coverage=%d',len(rows))


if __name__=='__main__':main()
