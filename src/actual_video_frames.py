"""Decode actual PTS witnesses on a fixed window grid; no scores or annotations."""
from pathlib import Path
import math
import time
from PIL import Image
from src.video_inputs import fixed_windows


def resolve_video(row):
    choices=[Path(row['video_path'])]+[p for sub in ('video','videos')
        for p in sorted((Path.home()/'data'/row['dataset']/sub).glob(row['video_id']+'.*'))]
    path=next((p for p in choices if p.is_file()),None)
    if path is None:raise FileNotFoundError((row['dataset'],row['video_id']))
    return path


def select_indices(entries,windows):
    """First at/after each third point, otherwise last INSIDE this window."""
    selected=[]
    for a,b in windows:
        local=[e for e in entries if a<=e['time']<b]
        picks=[]
        for target in (a+(b-a)/3,a+2*(b-a)/3):
            pick=next((e for e in local if e['time']>=target),local[-1] if local else None)
            if pick is not None and pick['index'] not in picks:picks.append(pick['index'])
        selected.append(sorted(picks))
    return selected


def acquire_window_frames(row,folder,window_seconds=8):
    """One full decoded pass; retain only two target frames and tail per window."""
    import av
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=resolve_video(row);duration=float(row['duration']);windows=fixed_windows(duration,window_seconds)
    entries=[];picks=[[] for _ in windows];pending={};last={};origin=None;previous=-math.inf
    start=time.perf_counter();decoded=0
    def save(frame,entry):
        destination=folder/f'frame_{entry["index"]:08d}.png'
        image=frame.to_image().convert('RGB');image.save(destination);image.close()
    def close_window(w):
        if w not in last:return
        frame,entry=last.pop(w)
        a,b=windows[w]
        for target in (a+(b-a)/3,a+2*(b-a)/3):
            if target not in pending.get(w,set()):
                if entry['index'] not in picks[w]:save(frame,entry);picks[w].append(entry['index'])
        pending.pop(w,None)
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        origin=float(container.start_time/av.time_base) if container.start_time is not None else None
        current=None
        for index,frame in enumerate(container.decode(stream)):
            decoded+=1
            if frame.pts is None:raise ValueError('Actual decoded frame has no PTS')
            absolute=float(frame.pts*frame.time_base)
            if origin is None:origin=absolute
            t=absolute-origin
            assert t>previous,('Nonmonotone video PTS',str(path),index);previous=t
            if not 0<=t<duration:continue
            entry=dict(index=index,pts=int(frame.pts),time=t,absolute=absolute,
                time_base=[frame.time_base.numerator,frame.time_base.denominator],
                width=frame.width,height=frame.height)
            entries.append(entry);w=min(int(t//window_seconds),len(windows)-1)
            if current is not None and current!=w:close_window(current)
            current=w;last[w]=(frame,entry);pending.setdefault(w,set())
            a,b=windows[w]
            for target in (a+(b-a)/3,a+2*(b-a)/3):
                if target not in pending[w] and t>=target:
                    if index not in picks[w]:save(frame,entry);picks[w].append(index)
                    pending[w].add(target)
        if current is not None:close_window(current)
    picks=[sorted(v) for v in picks]
    assert picks==select_indices(entries,windows)
    meta=dict(input_video=str(path),manifest_video_path=row['video_path'],duration=duration,
        origin=origin,decoded_frames=decoded,entries=entries,selected_indices=picks,
        uncovered_windows=[i for i,p in enumerate(picks) if not p],
        window_seconds=window_seconds,decode_seconds=time.perf_counter()-start)
    validate_frames(meta,row,folder)
    meta['decode_seconds']=time.perf_counter()-start
    return meta


def validate_frames(meta,row,folder):
    # Original generating-host path remains provenance. Current-host source is
    # resolved independently, then ACTUALLY decoded/compared below after return.
    source=resolve_video(row)
    assert Path(meta['input_video']).stem==row['video_id'] and source.stem==row['video_id']
    assert meta['manifest_video_path']==row['video_path']
    assert meta['duration']==float(row['duration'])
    entries=meta['entries'];ids={e['index']:e for e in entries};assert len(ids)==len(entries)
    assert all(0<=e['time']<meta['duration'] and e['width']>0 and e['height']>0 for e in entries)
    assert all((a['time'],a['index'])<(b['time'],b['index']) for a,b in zip(entries,entries[1:]))
    for e in entries:
        assert abs(e['pts']*e['time_base'][0]/e['time_base'][1]-e['absolute'])<1e-8
        assert abs(e['absolute']-meta['origin']-e['time'])<1e-8
    windows=fixed_windows(meta['duration'],meta['window_seconds'])
    assert meta['selected_indices']==select_indices(entries,windows)
    assert meta['uncovered_windows']==[i for i,p in enumerate(meta['selected_indices']) if not p]
    for picks in meta['selected_indices']:
        for index in picks:
            e=ids[index]
            with Image.open(Path(folder)/f'frame_{index:08d}.png') as image:
                assert image.format=='PNG' and image.size==(e['width'],e['height'])
                image.load()
    import av
    import numpy as np
    wanted={i for picks in meta['selected_indices'] for i in picks};verified=set();current_origin=None;current_entries=[];decoded=0
    with av.open(str(source)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        current_origin=float(container.start_time/av.time_base) if container.start_time is not None else None
        for index,frame in enumerate(container.decode(stream)):
            decoded+=1
            if frame.pts is None:raise ValueError('Actual decoded verification frame lacks PTS')
            absolute=float(frame.pts*frame.time_base)
            if current_origin is None:current_origin=absolute
            t=absolute-current_origin
            if 0<=t<meta['duration']:
                current_entries.append(dict(index=index,pts=int(frame.pts),time=t,absolute=absolute,
                    time_base=[frame.time_base.numerator,frame.time_base.denominator],width=frame.width,height=frame.height))
            if index not in wanted:continue
            entry=ids[index]
            assert (int(frame.pts),frame.time_base.numerator,frame.time_base.denominator)==(entry['pts'],*entry['time_base'])
            assert abs(absolute-current_origin-entry['time'])<1e-8 and current_origin==meta['origin']
            actual=frame.to_image().convert('RGB')
            try:
                with Image.open(Path(folder)/f'frame_{index:08d}.png') as image:
                    assert np.array_equal(np.asarray(actual),np.asarray(image.convert('RGB'))),'Source witness pixels mismatch'
            finally:actual.close()
            verified.add(index)
    assert verified==wanted and current_entries==entries and decoded==meta['decoded_frames'] and current_origin==meta['origin']
    assert meta['decoded_frames']>=len(entries) and meta['decode_seconds']>=0
