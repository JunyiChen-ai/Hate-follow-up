"""Actual one-fps source observations with original PTS and readonly pixels."""
import math
import time
from pathlib import Path
import numpy as np
from PIL import Image
from timeline import ROOT,SPEC,uniform,draw
from src.actual_video_frames import resolve_video
from src.video_inputs import load_manifest,fixed_windows,window_text
DATASETS=('HateMM','HateClipSeg');CACHE=ROOT/'data/temporal_time_tools'


def selected_rows(smoke=False):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def chosen(entries,duration):
    pointer=0;result=[]
    for k in range(math.ceil(duration*SPEC['source_fps'])):
        target=k/SPEC['source_fps']
        while pointer<len(entries) and entries[pointer]['time']<target:pointer+=1
        if pointer<len(entries) and entries[pointer]['index'] not in result:result.append(entries[pointer]['index'])
    return result


def decoded(row):
    import av
    with av.open(str(resolve_video(row))) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO';origin=float(container.start_time/av.time_base) if container.start_time is not None else None;last=-math.inf
        for index,frame in enumerate(container.decode(stream)):
            if frame.pts is None:raise ValueError('Actual video frame has no PTS')
            absolute=float(frame.pts*frame.time_base)
            if origin is None:origin=absolute
            t=absolute-origin;assert t>last;last=t
            yield dict(index=index,pts=int(frame.pts),time=t,absolute=absolute,time_base=[frame.time_base.numerator,frame.time_base.denominator],width=frame.width,height=frame.height),frame,origin


def acquire_sources(row,folder):
    start=time.perf_counter();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    entries=[];selected=[];pointer=0;decoded_count=0;origin=None;duration=float(row['duration'])
    for e,frame,origin in decoded(row):
        decoded_count+=1
        if not 0<=e['time']<duration:continue
        entries.append(e)
        if pointer<math.ceil(duration*SPEC['source_fps']) and e['time']>=pointer/SPEC['source_fps']:
            selected.append(e['index'])
            with frame.to_image().convert('RGB') as image:image.save(folder/f'frame_{e["index"]:08d}.png')
            while pointer<math.ceil(duration*SPEC['source_fps']) and e['time']>=pointer/SPEC['source_fps']:pointer+=1
    assert selected==chosen(entries,duration)
    result=dict(input_video=str(resolve_video(row)),manifest_video_path=row['video_path'],duration=duration,origin=origin,entries=entries,selected_indices=selected,decoded_frames=decoded_count,decode_seconds=time.perf_counter()-start)
    validate_sources(result,row,folder);result['decode_seconds']=time.perf_counter()-start;return result


def validate_sources(source,row,folder):
    assert Path(source['input_video']).stem==row['video_id'] and source['manifest_video_path']==row['video_path'] and source['duration']==float(row['duration'])
    assert source['selected_indices']==chosen(source['entries'],source['duration']);wanted=set(source['selected_indices']);entries=[];seen=set();count=0;origin=None
    for e,frame,origin in decoded(row):
        count+=1
        if 0<=e['time']<source['duration']:entries.append(e)
        if e['index'] in wanted:
            with frame.to_image().convert('RGB') as actual,Image.open(Path(folder)/f'frame_{e["index"]:08d}.png') as saved:
                assert saved.format=='PNG' and np.array_equal(np.asarray(actual),np.asarray(saved.convert('RGB'))),'Actual source pixels changed'
            seen.add(e['index'])
    assert entries==source['entries'] and seen==wanted and count==source['decoded_frames'] and origin==source['origin']


def windows_for(row,segments,source):
    entries={e['index']:e for e in source['entries']};result=[]
    for i,(a,b) in enumerate(fixed_windows(float(row['duration']),SPEC['window_seconds'])):
        ids=uniform([k for k in source['selected_indices'] if a<=entries[k]['time']<b],SPEC['current_frames_cap'])
        result.append(dict(i=i,start=a,end=b,body=window_text(segments,a,b),sample_ids=ids))
    return result


def media_content(source,ids,folder,state=None,current=None,save=True,stem='memory'):
    entries={e['index']:e for e in source['entries']};items=[];paths=[]
    for index in ids:
        e=entries[index];path=Path(folder)/'frames'/f'frame_{index:08d}.png'
        if state is not None and state['progress']:
            name=Path(folder)/'rendered'/stem/f'frame_{index:08d}.png'
            with Image.open(path) as image:
                annotated=draw(image.convert('RGB'),e['time'],source['duration'],state['highlights'],current)
            try:
                if save:name.parent.mkdir(parents=True,exist_ok=True);annotated.save(name)
                else:
                    with Image.open(name) as saved:assert np.array_equal(np.asarray(annotated),np.asarray(saved.convert('RGB')))
            finally:annotated.close()
            path=name
        relative=str(path.relative_to(ROOT));items.extend([dict(type='text',text=f"Actual original frame {index} at PTS {e['time']:.9f}s in original video time; any highlight is a retrieval condition, not a certified event."),dict(type='image',image=relative)]);paths.append(relative)
    return items,paths
