"""Actual in-window fractional-point media acquisition and exact RGB replay."""
import math
import time
from pathlib import Path
import numpy as np
from PIL import Image
from src.actual_video_frames import resolve_video
from src.video_inputs import fixed_windows


def selected(entries,windows,fractions):
    assert fractions and all(0<x<1 for x in fractions) and fractions==sorted(set(fractions))
    result=[]
    for a,b in windows:
        own=[e for e in entries if a<=e['time']<b]
        result.append([next((e['index'] for e in own if e['time']>=a+(b-a)*f),None) for f in fractions])
    return result


def decoded(row):
    import av
    duration=float(row['duration']);origin=None;previous=-math.inf
    with av.open(str(resolve_video(row))) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        origin=float(container.start_time/av.time_base) if container.start_time is not None else None
        for index,frame in enumerate(container.decode(stream)):
            assert frame.pts is not None;absolute=float(frame.pts*frame.time_base)
            if origin is None:origin=absolute
            t=absolute-origin;assert t>previous;previous=t
            e=dict(index=index,pts=int(frame.pts),time=t,absolute=absolute,time_base=[frame.time_base.numerator,frame.time_base.denominator],width=frame.width,height=frame.height)
            yield frame,e,origin,0<=t<duration


def acquire(row,folder,fractions,window_seconds):
    start=time.perf_counter();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    duration=float(row['duration']);windows=fixed_windows(duration,window_seconds)
    targets=[[a+(b-a)*f for f in fractions] for a,b in windows];picks=[[None]*len(fractions) for _ in windows]
    entries=[];saved=set();count=0;origin=None
    for frame,e,origin,available in decoded(row):
        count+=1
        if not available:continue
        entries.append(e);w=min(int(e['time']//window_seconds),len(windows)-1)
        for k,target in enumerate(targets[w]):
            if picks[w][k] is not None or e['time']<target:continue
            picks[w][k]=e['index']
            if e['index'] not in saved:
                image=frame.to_image().convert('RGB')
                try:image.save(folder/f'frame_{e["index"]:08d}.png')
                finally:image.close()
                saved.add(e['index'])
    assert picks==selected(entries,windows,fractions)
    meta=dict(input_video=str(resolve_video(row)),manifest_video_path=row['video_path'],duration=duration,origin=origin,decoded_frames=count,entries=entries,
        fractions=fractions,window_seconds=window_seconds,targets=targets,target_indices=picks,selected_indices=[sorted(set(i for i in pp if i is not None)) for pp in picks],
        uncovered_windows=[i for i,pp in enumerate(picks) if all(x is None for x in pp)],decode_seconds=time.perf_counter()-start)
    validate(meta,row,folder,fractions,window_seconds);meta['decode_seconds']=time.perf_counter()-start
    return meta


def validate(meta,row,folder,fractions,window_seconds):
    windows=fixed_windows(float(row['duration']),window_seconds)
    assert meta['manifest_video_path']==row['video_path'] and meta['duration']==float(row['duration']) and Path(meta['input_video']).stem==row['video_id']
    assert meta['fractions']==fractions and meta['window_seconds']==window_seconds
    assert meta['targets']==[[a+(b-a)*f for f in fractions] for a,b in windows]
    assert meta['target_indices']==selected(meta['entries'],windows,fractions)
    picks=[sorted(set(i for i in pp if i is not None)) for pp in meta['target_indices']]
    assert meta['selected_indices']==picks and meta['uncovered_windows']==[i for i,pp in enumerate(picks) if not pp]
    wanted={i for pp in picks for i in pp};checked=set();entries=[];count=0;origin=None
    for frame,e,origin,available in decoded(row):
        count+=1
        if available:entries.append(e)
        if e['index'] not in wanted:continue
        image=frame.to_image().convert('RGB')
        try:
            with Image.open(Path(folder)/f'frame_{e["index"]:08d}.png') as saved:
                assert saved.format=='PNG' and saved.size==(e['width'],e['height'])
                assert np.array_equal(np.asarray(image),np.asarray(saved.convert('RGB')))
        finally:image.close()
        checked.add(e['index'])
    assert checked==wanted and entries==meta['entries'] and count==meta['decoded_frames'] and origin==meta['origin'] and meta['decode_seconds']>=0
