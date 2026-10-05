"""Synthetic actual video, source grammar/token/pixel/event/background replay."""
import copy
import json
import os
import socket
import av
import numpy as np
import torch
from PIL import Image
from events import ROOT,SPEC,select,candidates
import extract as e
from src.mllm_renderer import cpu_renderer
from src.structured_source_generation import Stream,image_rope_delta


def main():
    out=ROOT/'runs/20261006_m1_videoevent/collector_cpu_checks';out.mkdir(parents=True,exist_ok=True);print('host',socket.gethostname(),flush=True);(out/'run.pid').write_text(str(os.getpid()))
    j=cpu_renderer();torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats=lambda:None;torch.cuda.max_memory_allocated=lambda:0
    class Scripted(Stream):
        def __init__(self,texts,choices,limit,**kw):super().__init__(j,limit,tokens=[],**kw);self.texts=iter(texts);self.choices=iter(choices)
        def force(self,text):self.saved.extend(j.tok.encode(text,add_special_tokens=False));return super().force(text)
        def choose(self,options):
            text=next(self.choices);assert text in options;self.saved.extend(j.tok.encode(text,add_special_tokens=False));return super().choose(options)
        def description(self):self.saved.extend(j.tok.encode(next(self.texts),add_special_tokens=False)+j.tok.encode('"',add_special_tokens=False));return super().description()
    def gen(j,root,system,content,paths,limit,writer,field_tokens=64,description_words=24):
        choices=[]
        if system==SPEC['caption_system']:texts=['visible moving cup']
        elif system==SPEC['background_system']:texts=['a cup beside table']
        else:
            texts=[];context=json.loads(content[0]['text']);choices=['1 ' if c['id'] in (1,4) else '10 ' for c in context['candidates']]
        stream=Scripted(texts,choices,limit,field_tokens=field_tokens,description_words=description_words);selection=writer(stream);prompt=j.render([j.turn('system',system),dict(role='user',content=content)],True)
        images=[Image.open(root/p).convert('RGB') for p in paths]
        try:enc=j.encode(prompt,images)
        finally:
            for image in images:image.close()
        n=enc['input_ids'].shape[1];delta=image_rope_delta(j,enc)
        return dict(selection=selection,truncated=False,text=j.tok.decode(stream.tokens,skip_special_tokens=True).strip(),tokens=stream.tokens,events=stream.events,seconds=.001,actual_forwards=1+len(stream.tokens),actual_vision_forwards=int(bool(paths)),prompt=prompt,system=system,max_tokens=limit,image_paths=paths,input_tokens=enc['input_ids'][0].tolist(),image_grid=enc['image_grid_thw'].tolist() if paths else [],prefill_cache_tokens=n,rope_delta=delta,positions=list(range(n+delta,n+delta+len(stream.tokens))),field_tokens=field_tokens,description_words=description_words)
    e.generate=gen;video=out/'fixture.mkv'
    with av.open(str(video),'w') as c:
        stream=c.add_stream('ffv1',rate=4);stream.width=96;stream.height=48;stream.pix_fmt='bgr0'
        for i in range(320):
            rgb=np.zeros((48,96,3),np.uint8);rgb[:,:,0]=(i*2)%256;rgb[:,:,1]=np.arange(96);rgb[:,:,2]=np.arange(48)[:,None]
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):c.mux(packet)
        for packet in stream.encode():c.mux(packet)
    row=dict(dataset='synthetic',video_id='fixture',video_path=str(video),duration=80.);segments=[(i*8.,i*8+7.,'literal source '+str(i)) for i in range(10)];folder=out/'source'
    m=e.acquire(j,row,segments,folder);before=copy.deepcopy(m);files={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in folder.rglob('*.png')};e.validate(j,m,row,segments,folder)
    assert m==before and all(p.read_bytes()==b and p.stat().st_mtime_ns==t for p,(b,t) in files.items())
    assert len(m['windows'])==m['cost']['caption_calls']==m['cost']['relevance_calls']==10 and 1<=m['cost']['background_calls']<=10
    for i,p in enumerate(m['packets']):
        ids=candidates(i,10);values=[1 if k in (1,4) else 10 for k in ids];assert p==select(ids,values,i) and len(p['selected_windows'])<=4
        assert m['backgrounds'][str(p['representative'])]['frame']==m['windows'][p['representative']]['frames'][0]
    for kind in ('ASR','event','background','token'):
        bad=copy.deepcopy(m)
        if kind=='ASR':bad['windows'][0]['body']='wrong literal ASR'
        elif kind=='event':bad['packets'][0]['representative']=9
        elif kind=='background':next(iter(bad['backgrounds'].values()))['frame']['time']+=1.
        else:bad['relevance'][0]['input_tokens'][0]+=1
        try:e.validate(j,bad,row,segments,folder)
        except AssertionError:pass
        else:raise AssertionError('corruption accepted '+kind)
    (out/'synthetic_metadata.json').write_text(json.dumps(m)+'\n');(out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,scope='real synthetic video/scripted provider; no pretrained semantics',actual_PTS=True,readonly_pixels_tokens=True,source_corruption_rejected=True,background_reuse_owned=True,cost=m['cost']),indent=2)+'\n');print('COLLECTOR_CPU_PASS',flush=True)


if __name__=='__main__':main()
