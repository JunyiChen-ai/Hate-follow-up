"""Current raw media/PTS and shared native inputs, independent of annotations."""
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from interface import SPEC
from src.actual_video_frames import resolve_video,select_indices
from src.video_inputs import load_manifest,fixed_windows
DATASETS=('HateMM','HateClipSeg')
CACHE=ROOT/'data/temporal_text_occurrences'


def selected_rows(smoke=False):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def windows_for(row,source,folder):
    entries={e['index']:e for e in source['entries']};windows=[]
    for i,((a,b),indices) in enumerate(zip(fixed_windows(float(row['duration']),SPEC['window_seconds']),source['selected_indices'])):
        frames=[]
        for k,index in enumerate(indices):
            e=entries[index];assert a<=e['time']<b
            frames.append(dict(id='p'+str(k),time=e['time'],shape=[e['width'],e['height']],index=index,
                path=str((folder/'frames'/f'frame_{index:08d}.png').relative_to(ROOT))))
        windows.append(dict(i=i,start=a,end=b,frames=frames))
    return windows


def sample_indices(source):
    entries=source['entries'];pointer=0;grid=[]
    for k in range(math.ceil(source['duration']*SPEC['tracking_fps'])):
        target=k/SPEC['tracking_fps']
        while pointer<len(entries) and entries[pointer]['time']<target:pointer+=1
        if pointer<len(entries):grid.append(entries[pointer]['index'])
    anchors=[i for local in source['selected_indices'] for i in local]
    return sorted(set(grid+anchors))


def samples(row,source):
    """Actual selected RGB frames; source metadata is checked against every PTS."""
    import av
    entries={e['index']:e for e in source['entries']};wanted=set(sample_indices(source))
    verified=set();origin=None;last=-math.inf
    with av.open(str(resolve_video(row))) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        origin=float(container.start_time/av.time_base) if container.start_time is not None else None
        for index,frame in enumerate(container.decode(stream)):
            if frame.pts is None:raise ValueError('Actual frame lacks PTS')
            absolute=float(frame.pts*frame.time_base)
            if origin is None:origin=absolute
            t=absolute-origin;assert t>last;last=t
            if index not in wanted:continue
            e=entries[index]
            assert e['pts']==int(frame.pts) and e['time_base']==[frame.time_base.numerator,frame.time_base.denominator]
            assert origin==source['origin'] and abs(t-e['time'])<1e-8
            assert [frame.width,frame.height]==[e['width'],e['height']]
            with frame.to_image().convert('RGB') as image:rgb=np.asarray(image).copy()
            verified.add(index);yield e,rgb
    assert verified==wanted


def pixel_box(box,shape):
    width,height=shape;x0,y0,x1,y1=box;scale=SPEC['coordinate_scale']
    assert all(type(v) is int for v in box) and 0<=x0<x1<=scale and 0<=y0<y1<=scale
    return [math.floor(x0*width/scale),math.floor(y0*height/scale),math.ceil(x1*width/scale),math.ceil(y1*height/scale)]


def save_rgb(rgb,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(rgb).save(path)
    with Image.open(path) as image:assert image.format=='PNG' and np.array_equal(np.asarray(image.convert('RGB')),rgb)
