"""Bounded acquisition primitives. No annotations, fitting or content hashes."""
import math
from pathlib import Path
import numpy as np
import torch


def binary_entropy(z):
    x=abs(float(z)); assert math.isfinite(x)
    e=math.exp(-x)
    return math.log1p(e)+x*e/(1+e)


def choose_frames(candidates, observed_times, prior, count=2, attention=True):
    """Return candidate row indices; all input times are seconds on one timeline."""
    times=np.asarray([c['time'] for c in candidates],dtype=float)
    observed=np.asarray(observed_times,dtype=float); p=np.asarray(prior,dtype=float)
    assert len(observed)==len(p)>0 and np.isfinite(p).all() and (p>=0).all()
    if not len(times):return []
    # Coincident observations carry their mean prior; equal distances choose earlier.
    unique=np.unique(observed)
    means=np.asarray([p[observed==t].mean() for t in unique])
    nearest=np.abs(times[:,None]-unique[None,:]).argmin(1)
    a=np.sqrt(means[nearest]) if attention else np.ones(len(times))
    distance=np.abs(times[:,None]-observed[None,:]).min(1)
    selected=[]
    for _ in range(min(count,len(times))):
        score=a*distance; score[selected]=-np.inf
        # Input candidates are sorted by actual time and source index.
        pick=int(np.argmax(score));selected.append(pick)
        distance=np.minimum(distance,np.abs(times-times[pick]))
    return selected


class FrameAttention:
    """Read-only pre-RoPE normalized Q/K capture, preserving native forwards."""
    def __init__(self,judge):
        self.j=judge; self.layers=judge.model.model.language_model.layers
        assert len(self.layers)==36
        self.mode=None;self.mask=None;self.prefix_keys={};self.query_keys={};self.queries={}
        self.handles=[]
        for i,layer in enumerate(self.layers):
            self.handles.append(layer.self_attn.k_norm.register_forward_hook(self._key(i)))
            self.handles.append(layer.self_attn.q_norm.register_forward_hook(self._query(i)))

    def _key(self,i):
        def hook(module,args,output):
            if self.mode is None:return
            assert output.ndim==4 and output.shape[:2]==(1,len(self.mask))
            keys=output[0,self.mask].detach().clone().transpose(0,1)
            if self.mode=='prefix':self.prefix_keys[i]=keys
            else:self.query_keys[i]=keys
        return hook

    def _query(self,i):
        def hook(module,args,output):
            if self.mode=='query':self.queries[i]=output[0,-1].detach().clone()
        return hook

    def start(self,mode,mask):
        assert self.mode is None and mode in ('prefix','query')
        self.mode=mode;self.mask=mask.to(self.j.device,dtype=torch.bool)
        if mode=='prefix':self.prefix_keys={}
        else:self.query_keys={};self.queries={}

    def stop(self):
        self.mode=None;self.mask=None

    @torch.no_grad()
    def prior(self,counts):
        assert self.mode is None and len(self.queries)==len(self.query_keys)==len(self.prefix_keys)==36
        total=None
        for i in range(36):
            q=self.queries[i].float()
            k=torch.cat((self.prefix_keys[i],self.query_keys[i]),1).float()
            assert q.shape[0]==32 and k.shape[0]==8 and q.shape[-1]==k.shape[-1]==128
            k=k.repeat_interleave(q.shape[0]//k.shape[0],dim=0)
            w=torch.softmax(torch.einsum('hd,hnd->hn',q,k)/math.sqrt(q.shape[-1]),-1).mean(0)
            total=w if total is None else total+w
        total=total/36
        assert sum(counts)==len(total) and all(n>0 for n in counts)
        values=torch.stack([v.mean() for v in total.split(list(map(int,counts)))])
        values=values/values.sum()
        assert torch.isfinite(values).all()
        self.query_keys={};self.queries={}
        return values.cpu().numpy()

    def close(self):
        assert self.mode is None
        for handle in self.handles:handle.remove()
        self.handles=[];self.prefix_keys={};self.query_keys={};self.queries={}


class FrameSource:
    """Index actual PTS once, decode only requested witness images on demand."""
    def __init__(self,row,original_frames,out):
        import av
        self.av=av;self.row=row;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        given=Path(row['video_path'])
        possible=[given]+[p for sub in ('video','videos')
            for p in sorted((Path.home()/'data'/row['dataset']/sub).glob(row['video_id']+'.*'))]
        self.path=next((p for p in possible if p.is_file()),None)
        if self.path is None:raise FileNotFoundError((row['dataset'],row['video_id']))
        with av.open(str(self.path)) as container:
            stream=container.streams.video[0];stream.thread_type='AUTO'
            self.origin=float(container.start_time/av.time_base) if container.start_time is not None else None
            entries=[]
            for index,frame in enumerate(container.decode(stream)):
                if frame.pts is None:raise ValueError('Decoded frame lacks PTS')
                absolute=float(frame.pts*frame.time_base)
                if self.origin is None:self.origin=absolute
                entries.append({'index':index,'pts':int(frame.pts),'time':absolute-self.origin,
                    'time_base':[frame.time_base.numerator,frame.time_base.denominator],
                    'width':frame.width,'height':frame.height})
        assert entries and all(a['time']<b['time'] for a,b in zip(entries,entries[1:])), 'nonmonotone decoded source'
        self.entries=entries;self.times=np.array([r['time'] for r in entries])
        self.legacy_excluded=set()
        for _,path in original_frames:
            index=int(Path(path).stem.split('_')[0][1:])
            target=float(f'{(index+.5)*float(row["duration"])/20:.3f}')
            for t in (target,float(f'{max(0.,target-.5):.3f}')):
                pos=int(np.searchsorted(self.times,t))
                if pos<len(entries):self.legacy_excluded.add(pos)
        self.container=av.open(str(self.path));self.stream=self.container.streams.video[0]
        self.stream.thread_type='AUTO'

    def candidates(self,start,end,used=()):
        indices=set();excluded=self.legacy_excluded|set(used)
        lo=max(0,math.ceil(4*start-.5));hi=math.ceil(4*end-.5)
        for k in range(lo,hi):
            t=(k+.5)/4
            if not start<=t<end:continue
            index=int(np.searchsorted(self.times,t))
            if index<len(self.entries) and start<=self.times[index]<end and index not in excluded:
                indices.add(index)
        return [self.entries[i].copy() for i in sorted(indices)]

    def image(self,entry):
        """PNG witness in run output, never in the read-only input/cache directory."""
        from PIL import Image
        dst=self.out/f'frame_{entry["index"]:08d}.png'
        if dst.exists():
            with Image.open(dst) as im:
                assert im.size==(entry['width'],entry['height'])
                return im.convert('RGB'),dst
        seek=int((entry['time']+self.origin)/float(self.stream.time_base))
        self.container.seek(seek,stream=self.stream,backward=True,any_frame=False)
        for frame in self.container.decode(self.stream):
            absolute=float(frame.pts*frame.time_base)
            expected=entry['time']+self.origin
            if abs(absolute-expected)<1e-8:
                assert frame.width==entry['width'] and frame.height==entry['height']
                im=frame.to_image().convert('RGB');im.save(dst)
                return im,dst
            if absolute>expected+1e-8:raise ValueError('Seek skipped requested indexed frame')
        raise ValueError('Source ended before requested indexed frame')

    def close(self):self.container.close()
