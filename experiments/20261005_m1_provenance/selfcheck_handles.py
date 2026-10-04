#!/usr/bin/env python3
"""Actual source inputs and legal grammar replay, no generated factual claims or GT."""
import json
import socket
import sys
import re
from pathlib import Path
from inputs_handles import ROOT,selected_rows,native_frames,ledger_content
from handle_interface import catalog,write_ledger,write_links,compiled_ledger,compiled_links
from graph import compile_graph,retrieve
from src.mllm_renderer import cpu_renderer
from src.structured_source_generation import Stream,Capped,image_rope_delta
from src.actual_video_frames import validate_frames
from src.video_inputs import load_asr,fixed_windows,window_text
from PIL import Image


class FixtureStream(Stream):
    """Select fixture routes explicitly; no model or alternative scores."""
    def __init__(self,j,limit,empty=False):
        super().__init__(j,limit);self.empty=empty;self.links=0

    def append(self,token):
        if len(self.tokens)>=self.limit:raise Capped()
        self.tokens.append(token)

    def choose(self,options):
        if self.empty:value=options[0]
        elif options[0]==']':value=options[-1]
        elif '"STOP"' in options:
            value=options[0] if self.links>=12 or len(options)==1 else options[-1];self.links+=1
        else:value=options[-1]
        ids=self.j.tok.encode(value,add_special_tokens=False);start=len(self.tokens)
        for token in ids:self.append(token)
        self.events.append(dict(kind='choice',options=options,selected=value,start=start,end=len(self.tokens)))
        return value

    def description(self):
        start=len(self.tokens)
        for token in self.j.tok.encode('visible person"',add_special_tokens=False):self.append(token)
        self.events.append(dict(kind='description',text='visible person',reason='model_quote',start=start,end=len(self.tokens)))


def generation(j,writer,limit=2048,empty=False):
    stream=FixtureStream(j,limit,empty);value=None;complete=True
    try:value=writer(stream)
    except Capped:complete=False
    replay=Stream(j,limit,tokens=stream.tokens);other=None;replay_complete=True
    try:other=writer(replay)
    except Capped:replay_complete=False
    assert stream.tokens==replay.tokens and stream.events==replay.events
    assert complete==replay_complete and other==value
    return dict(tokens=stream.tokens,text=j.tok.decode(stream.tokens),truncated=not complete,selection=value)


def main():
    out=ROOT/'runs/20261005_m1_provenance/handle_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    print(socket.gethostname(),flush=True);(out/'run.pid').write_text(str(__import__('os').getpid()))
    j=cpu_renderer();fixture=[];windows=[]
    for i in range(6):
        w=dict(i=i,start=i*8,end=(i+1)*8,body=f'Alice{i} said hello to Bob{i}',frames=[dict(id='p0')]);windows.append(w)
        g=generation(j,lambda s:write_ledger(s,w),512)
        parsed=compiled_ledger(g,w);assert parsed['error'] is None;fixture.append(parsed)
        empty=generation(j,lambda s:write_ledger(s,w),512,True)
        assert compiled_ledger(empty,w)['nodes']==[n for n in parsed['nodes'] if n['kind']=='utterance']
        cap=generation(j,lambda s:write_ledger(s,w),6)
        rejected=compiled_ledger(cap,w);assert cap['truncated'] and rejected['error'] and len(rejected['nodes'])==1
    local=compile_graph(fixture,[]);g=generation(j,lambda s:write_links(s,local,[0,1]),2048)
    linked=compiled_links(g,local['nodes'],[0,1]);assert linked['error'] is None and linked['edges']
    graph=compile_graph(fixture,[linked]);assert any(retrieve(graph,i,windows)['selected_windows'] for i in range(6))
    stop=generation(j,lambda s:write_links(s,local,[0,1]),2048,True)
    assert not compiled_links(stop,local['nodes'],[0,1])['edges'] and 'STOP' in stop['text']
    repeated=dict(body='hello hello 你 好 你 好');assert all(repeated['body'].count(repeated['body'][p['start']:p['end']])==1 for p in catalog(repeated).values())
    result=dict(host=socket.gethostname(),GT_read=False,model_scores_computed=False,
        grammar_fixture_graph_edges=len(graph['edges']),fixture_remote_exercised=True,actual_inputs=[],full_source_span_preflight=[])
    # Original A actual decoded sources are validated, then B prompt constructed
    # from those same real witnesses. This is not a generated B observation.
    for row in selected_rows(True):
        path=ROOT/'data/temporal_entity_discourse_graph'/row['dataset']/row['video_id']/'metadata.json'
        metadata=json.loads(path.read_text());validate_frames(metadata['source'],row,path.parent/'frames')
        window=metadata['windows'][0];content,paths=ledger_content(native_frames(row),window)
        from graph import LEDGER_SYSTEM
        text=j.render([j.turn('system',LEDGER_SYSTEM),dict(role='user',content=content)],True)
        images=[Image.open(ROOT/p).convert('RGB') for p in paths]
        try:encoded=j.encode(text,images)
        finally:
            for image in images:image.close()
        delta=image_rope_delta(j,encoded)
        result['actual_inputs'].append(dict(dataset=row['dataset'],video_id=row['video_id'],images=len(paths),
            expanded_tokens=len(encoded['input_ids'][0]),rope_delta=delta,span_handles=len(catalog(window)),actual_pixels_validated=True))
    asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')}
    for row in selected_rows(False):
        for i,(a,b) in enumerate(fixed_windows(float(row['duration']),8)):
            body=window_text(asr[row['dataset']].get(row['video_id'],[]),a,b);spans=catalog(dict(body=body))
            result['full_source_span_preflight'].append(dict(dataset=row['dataset'],video_id=row['video_id'],window=i,
                body_characters=len(body),span_handles=len(spans),
                compact_boundary_tokens=len(j.tok.encode(json.dumps([[m.start(),m.end()] for m in re.finditer(r'\S+',body)],separators=(',',':')),add_special_tokens=False))))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='full_source_span_preflight'},indent=2),flush=True)
    print('CPU_CHECKS_DONE',flush=True)


if __name__=='__main__':main()
