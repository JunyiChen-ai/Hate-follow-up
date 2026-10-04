"""Actual audio sample timeline, relative to the video's presentation origin."""
import math
from pathlib import Path
import numpy as np
RATE=16000
OVERLAP_POLICY='first observed resampled sample in decode order; no shift or averaging'


def resolve_video(row):
    given=Path(row['video_path'])
    options=[given]+[p for directory in ('video','videos') for p in sorted(
        (Path.home()/'data'/row['dataset']/directory).glob(row['video_id']+'.*'))]
    found=next((p for p in options if p.is_file()),None)
    if found is None:raise FileNotFoundError((row['dataset'],row['video_id']))
    return found


def place_block(samples,observed,start,values):
    """Keep presentation coordinates even when malformed media overlaps blocks."""
    lo=max(start,0);hi=min(start+len(values),len(samples))
    if hi<=lo:return dict(placed_intervals=[],overlap_discarded_intervals=[])
    prior=observed[lo:hi].copy();keep=~prior
    samples[lo:hi][keep]=values[lo-start:hi-start][keep]
    observed[lo:hi][keep]=True
    return dict(placed_intervals=intervals(keep,lo),overlap_discarded_intervals=intervals(prior,lo))


def decode_audio(path,duration):
    import av
    total=math.ceil(duration*RATE);samples=np.zeros(total,dtype=np.float32)
    observed=np.zeros(total,dtype=bool);blocks=[]
    with av.open(str(path)) as container:
        origin=container.start_time/av.time_base if container.start_time is not None else None
        if origin is None:
            assert container.streams.video
            first=next(container.decode(container.streams.video[0]))
            assert first.pts is not None
            origin=float(first.pts*first.time_base)
        streams=list(container.streams.audio)
    if not streams:return samples,observed,dict(video_origin=origin,blocks=[],audio_stream=None,missing_audio=True,overlap_policy=OVERLAP_POLICY)
    with av.open(str(path)) as container:
        stream=container.streams.audio[0];resampler=av.AudioResampler(format='flt',layout='mono',rate=RATE)
        def consume(frame):
            assert frame.pts is not None and frame.sample_rate==RATE
            start=int(round((float(frame.pts*frame.time_base)-origin)*RATE))
            a=np.asarray(frame.to_ndarray(),dtype=np.float32).reshape(-1)
            assert len(a)==frame.samples and np.isfinite(a).all()
            placement=place_block(samples,observed,start,a)
            blocks.append(dict(pts=int(frame.pts),time_base=[frame.time_base.numerator,frame.time_base.denominator],
                start_sample=start,count=len(a),**placement))
        for source in container.decode(stream):
            assert source.pts is not None
            for frame in resampler.resample(source):consume(frame)
        for frame in resampler.resample(None):consume(frame)
        stream_info=dict(index=stream.index,codec=stream.codec_context.name,
            source_rate=stream.codec_context.sample_rate,source_layout=stream.codec_context.layout.name)
    return samples,observed,dict(video_origin=origin,blocks=blocks,audio_stream=stream_info,missing_audio=False,overlap_policy=OVERLAP_POLICY)


def intervals(mask,offset=0):
    padded=np.r_[False,np.asarray(mask,dtype=bool),False]
    starts=np.flatnonzero(~padded[:-1]&padded[1:])+offset
    ends=np.flatnonzero(padded[:-1]&~padded[1:])+offset
    return [[int(a),int(b)] for a,b in zip(starts,ends)]


def crop_audio(samples,observed,a,b):
    lo=max(0,int(math.floor(a*RATE)));hi=min(len(samples),int(math.ceil(b*RATE)))
    good=np.flatnonzero(observed[lo:hi])
    if not len(good):return np.empty(0,dtype=np.float32),dict(request=[a,b],samples=[],observed_intervals=[],gaps=[])
    start=lo+int(good[0]);end=lo+int(good[-1])+1
    return samples[start:end].copy(),dict(request=[a,b],samples=[start,end],
        actual_interval=[start/RATE,end/RATE],observed_intervals=intervals(observed[start:end],start),
        gaps=intervals(~observed[start:end],start),gap_fill='zero at unobserved positions; timeline never compressed')
