"""Read-only exact-PTS/RGB validation of uniform supplementary media caches."""
import math
from pathlib import Path
import numpy as np
from PIL import Image
from src.actual_video_frames import resolve_video


def target_indices(entries,duration,fps):
    assert duration>0 and fps>0
    targets=[i/fps for i in range(math.ceil(duration*fps))];cursor=0;chosen=[]
    for target in targets:
        while cursor<len(entries) and entries[cursor]['time']<target:cursor+=1
        chosen.append(entries[cursor]['index'] if cursor<len(entries) else None)
    return targets,chosen


def validate(meta,row,folder,fps):
    import av
    duration=float(row['duration']);folder=Path(folder)
    assert meta['duration']==duration and meta['manifest_video_path']==row['video_path']
    assert Path(meta['input_video']).stem==row['video_id'] and meta['source_fps']==fps
    targets,chosen=target_indices(meta['entries'],duration,fps)
    assert meta['targets']==targets and meta['target_indices']==chosen
    selected=sorted(set(i for i in chosen if i is not None))
    assert meta['selected_indices']==selected and meta['uncovered_targets']==[i for i,x in enumerate(chosen) if x is None]
    wanted=set(selected);checked=set();entries=[];decoded=0;previous=-math.inf;origin=None
    with av.open(str(resolve_video(row))) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        origin=float(container.start_time/av.time_base) if container.start_time is not None else None
        for index,frame in enumerate(container.decode(stream)):
            decoded+=1;assert frame.pts is not None
            absolute=float(frame.pts*frame.time_base)
            if origin is None:origin=absolute
            t=absolute-origin;assert t>previous;previous=t
            entry=dict(index=index,pts=int(frame.pts),time=t,absolute=absolute,time_base=[frame.time_base.numerator,frame.time_base.denominator],width=frame.width,height=frame.height)
            if 0<=t<duration:entries.append(entry)
            if index not in wanted:continue
            actual=frame.to_image().convert('RGB')
            try:
                with Image.open(folder/f'frame_{index:08d}.png') as image:
                    assert image.format=='PNG' and image.size==(frame.width,frame.height)
                    assert np.array_equal(np.asarray(actual),np.asarray(image.convert('RGB')))
            finally:actual.close()
            checked.add(index)
    assert checked==wanted and entries==meta['entries'] and origin==meta['origin'] and decoded==meta['decoded_frames']
    assert meta['decode_seconds']>=0
