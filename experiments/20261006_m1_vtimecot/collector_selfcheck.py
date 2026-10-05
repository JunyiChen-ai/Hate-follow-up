"""Synthetic actual video + scripted provider, exercising full collector replay."""
import copy
import json
import os
import socket
from types import SimpleNamespace
import av
import numpy as np
import torch
from PIL import Image
from timeline import ROOT,SPEC,retrieval,draw
import extract as e
import relevance as r
from src.mllm_renderer import cpu_renderer
from src.structured_source_generation import Stream,image_rope_delta


def main():
    out=ROOT/'runs/20261006_m1_vtimecot/collector_cpu_checks';out.mkdir(parents=True,exist_ok=True);print('host',socket.gethostname(),flush=True);(out/'run.pid').write_text(str(os.getpid()))
    j=cpu_renderer();torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats=lambda:None;torch.cuda.max_memory_allocated=lambda:0
    class Scripted(Stream):
        def __init__(self,texts,choices,limit,**kw):super().__init__(j,limit,tokens=[],**kw);self.texts=iter(texts);self.choices=iter(choices)
        def force(self,text):self.saved.extend(j.tok.encode(text,add_special_tokens=False));return super().force(text)
        def choose(self,options):
            text=next(self.choices);assert text in options;self.saved.extend(j.tok.encode(text,add_special_tokens=False));return super().choose(options)
        def description(self):self.saved.extend(j.tok.encode(next(self.texts),add_special_tokens=False)+j.tok.encode('"',add_special_tokens=False));return super().description()
    def gen(j,root,system,content,paths,limit,writer,field_tokens=64,description_words=24):
        choices=[]
        if system==SPEC['query_system']:texts=['moving cup','visible sign']
        elif system==SPEC['feedback_system']:texts=['actual selected frames']
        else:
            state=json.loads(content[0]['text'])['current_state'];texts=['inspect actual sources']
            choices=['PROGRESS_BAR"'] if not state['progress'] else ['HIGHLIGHT"','0 '] if state['query_id'] is None else ['CUT"','0 ']
        stream=Scripted(texts,choices,limit,field_tokens=field_tokens,description_words=description_words);selection=writer(stream);prompt=j.render([j.turn('system',system),dict(role='user',content=content)],True)
        images=[Image.open(root/p).convert('RGB') for p in paths]
        try:enc=j.encode(prompt,images)
        finally:
            for image in images:image.close()
        n=enc['input_ids'].shape[1];delta=image_rope_delta(j,enc)
        return dict(selection=selection,truncated=False,text=j.tok.decode(stream.tokens,skip_special_tokens=True).strip(),tokens=stream.tokens,events=stream.events,seconds=.001,actual_forwards=1+len(stream.tokens),actual_vision_forwards=int(bool(paths)),prompt=prompt,system=system,max_tokens=limit,image_paths=paths,input_tokens=enc['input_ids'][0].tolist(),image_grid=enc['image_grid_thw'].tolist() if paths else [],prefill_cache_tokens=n,rope_delta=delta,positions=list(range(n+delta,n+delta+len(stream.tokens))),field_tokens=field_tokens,description_words=description_words)
    def clip(j,window,source,folder,queries):
        msgs,paths,head,enc=r.encode_prefix(j,window,source,folder);n=enc['input_ids'].shape[1];delta=image_rope_delta(j,enc);generations=[]
        for q,query in enumerate(queries):
            text,ids=r.query_suffix(j,msgs,head,query);stream=Scripted([],['10}' if q==0 and window['i'] not in (1,4) else '0}'],16);selection=r.relevance_writer(stream)
            generations.append(dict(query=query,prompt=text,input_tokens=enc['input_ids'][0].tolist()+ids,tokens=stream.tokens,events=stream.events,selection=selection,truncated=False,text=j.tok.decode(stream.tokens,skip_special_tokens=True).strip(),max_tokens=16,prefill_cache_tokens=n+len(ids),rope_delta=delta,positions=list(range(n+len(ids)+delta,n+len(ids)+delta+len(stream.tokens))),actual_forwards=1+len(stream.tokens),actual_vision_forwards=0,seconds=.001))
        return dict(head=head,paths=paths,input_tokens=enc['input_ids'][0].tolist(),image_grid=enc['image_grid_thw'].tolist(),prefix_tokens=n,rope_delta=delta,prefix_seconds=.001,prefix_forwards=1,prefix_vision=1,generations=generations)
    e.generate=gen;e.read_clip=clip
    video=out/'fixture.mkv'
    with av.open(str(video),'w') as c:
        stream=c.add_stream('ffv1',rate=4);stream.width=96;stream.height=48;stream.pix_fmt='bgr0'
        for i in range(320):
            rgb=np.zeros((48,96,3),np.uint8);rgb[:,:,0]=(i*2)%256;rgb[:,:,1]=np.arange(96);rgb[:,:,2]=np.arange(48)[:,None]
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb,format='rgb24')):c.mux(packet)
        for packet in stream.encode():c.mux(packet)
    row=dict(dataset='synthetic',video_id='fixture',video_path=str(video),duration=80.);segments=[(i*8.,i*8+7.,'literal source '+str(i)) for i in range(10)];folder=out/'source'
    original=out/'native.png';Image.fromarray(np.full((48,96,3),125,np.uint8)).save(original);e.frame_paths=lambda *a:[(float(i),original) for i in range(20)]
    m=e.acquire(j,row,segments,folder);before=copy.deepcopy(m);files={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in folder.rglob('*.png')};e.validate(j,m,row,segments,folder)
    assert m==before and all(p.read_bytes()==b and p.stat().st_mtime_ns==t for p,(b,t) in files.items())
    assert m['source']['selected_indices']==list(range(0,320,4)) and m['relevance_values']==[[10,0,10,10,0,10,10,10,10,10],[0]*10]
    assert [s['plan']['action'] for s in m['steps']]==['PROGRESS_BAR','HIGHLIGHT','CUT'] and all(s['result']['executed'] for s in m['steps'])
    assert m['final_state']['memory_ids']==list(range(0,32,4)) and m['cost']['clip_prefix_calls']==10 and m['cost']['relevance_calls']==20
    for window,final in zip(m['windows'],m['final_inputs']):
        assert len(final['paths'])<=24 and any('previous_step' in c.get('text','') for c in final['content'])
        assert not any('inspect actual sources' in c.get('text','') or 'actual selected frames' in c.get('text','') for c in final['content'])
    for kind in ('source','tool','pixel'):
        bad=copy.deepcopy(m)
        if kind=='source':bad['windows'][0]['body']='wrong literal ASR'
        elif kind=='tool':bad['steps'][2]['plan']['interval_id']=1
        else:bad['clips'][0]['input_tokens'][0]+=1
        try:e.validate(j,bad,row,segments,folder)
        except AssertionError:pass
        else:raise AssertionError('corruption accepted '+kind)
    a=np.zeros((48,96,3),np.uint8);image=Image.fromarray(a);rendered=draw(image,3.,24.,[(0.,8.)],[8.,16.]);assert np.array_equal(np.asarray(rendered)[:48],a) and rendered.size==(96,112);image.close();rendered.close()
    (out/'synthetic_metadata.json').write_text(json.dumps(m)+'\n');(out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,scope='real synthetic video/scripted provider; not pretrained performance',actual_PTS=True,one_fps=True,tools_actual=True,changed_memory=True,readonly_pixels_tokens=True,corruption_rejected=True,clip_prefix_calls=10,relevance_calls=20,source_forwards=m['cost']['actual_forwards']),indent=2)+'\n');print('COLLECTOR_CPU_PASS',flush=True)


if __name__=='__main__':main()
