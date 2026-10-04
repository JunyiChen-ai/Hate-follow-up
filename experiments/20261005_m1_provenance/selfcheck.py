#!/usr/bin/env python3
"""Meaningful source/temporal/typed-graph fixtures, no model judgments or GT."""
import json
import logging
import socket
from pathlib import Path
import numpy as np
from inputs import ROOT,selected_rows,windows_for,native_frames,ledger_content
from graph import canonical,parse_ledger,parse_links,compile_graph,retrieve,LEDGER_SYSTEM
from src.actual_video_frames import select_indices,acquire_window_frames
from src.mllm_renderer import cpu_renderer
from src.video_inputs import load_asr


def generation(value):return dict(text=canonical(value),truncated=False)


def graph_checks():
    ww=[dict(i=i,start=8.*i,end=8.*(i+1),body=f'person{i} said hello',frames=[dict(id='p0')]) for i in range(6)]
    ledgers=[]
    for w in ww:
        value=dict(entities=[dict(id='e0',description='visible person',frame='p0',quote='')],
            actions=[dict(id='a0',description='speaking',actor='e0',target='UNKNOWN',frame='p0')],
            utterance=dict(speaker='e0'),quotations=[])
        parsed=parse_ledger(generation(value),w);assert parsed['error'] is None;ledgers.append(parsed)
        bad=json.loads(canonical(value));bad['entities'][0]['frame']='overview0'
        failed=parse_ledger(generation(bad),w);assert failed['error'] is not None and len(failed['nodes'])==1 and not failed['edges']
        bad=json.loads(canonical(value));bad['entities'][0].update(frame=None,quote='hello')
        assert parse_ledger(generation(bad),w)['error'] is None
        repeated={**w,'body':'hello hello'};assert parse_ledger(generation(bad),repeated)['error'] is not None
    local=compile_graph(ledgers,[])
    edges=[{'from':f'w{i:04d}/e0','to':f'w{i+1:04d}/e0','type':'same_entity'} for i in range(5)]
    links=parse_links(generation(dict(edges=edges)),local['nodes'],list(range(6)));assert links['error'] is None
    graph=compile_graph(ledgers,[links]);assert len(set(graph['entity_components'].values()))==1
    assert retrieve(graph,2,ww)['selected_windows']==[0,1,3,4]
    wrong={'from':'w0000/u','to':'w0005/u','type':'retracts'}
    assert parse_links(generation(dict(edges=[wrong])),local['nodes'],[0])['error'] is not None
    bad={'from':'w0000/e0','to':'w0005/u','type':'same_entity'}
    assert parse_links(generation(dict(edges=[edges[0],bad])),local['nodes'],[0])['edges']==[]
    # Independent all-pairs path oracle on random typed source graphs. No scores,
    # labels or method fitting. Half the links are zero-cost identity relations.
    rng=np.random.default_rng(0)
    for _ in range(50):
        picks=[]
        for a in range(6):
            for b in range(a+1,6):
                if rng.random()<.3:picks.append({'from':f'w{a:04d}/e0','to':f'w{b:04d}/e0','type':'same_entity'})
                if rng.random()<.2:picks.append({'from':f'w{a:04d}/u','to':f'w{b:04d}/u','type':'quotes'})
        linked=parse_links(generation(dict(edges=picks)),local['nodes'],list(range(6)));assert linked['error'] is None
        gg=compile_graph(ledgers,[linked]);names=[n['id'] for n in gg['nodes']];indices={n:i for i,n in enumerate(names)}
        d=np.full((len(names),len(names)),100.);np.fill_diagonal(d,0.)
        for edge in gg['edges']:
            a,b=indices[edge['from']],indices[edge['to']];cost=0. if edge['type']=='same_entity' else 1.
            d[a,b]=d[b,a]=min(d[a,b],cost)
        for k in range(len(names)):d=np.minimum(d,d[:,k,None]+d[None,k,:])
        for w in range(6):
            seeds=[indices[n['id']] for n in gg['nodes'] if n['window']==w]
            reachable={n['window'] for n in gg['nodes'] if n['window']!=w and min(d[s,indices[n['id']]] for s in seeds)<=2}
            before=sorted([i for i in reachable if i<w],key=lambda i:(abs(i-w),i))[:2]
            after=sorted([i for i in reachable if i>w],key=lambda i:(abs(i-w),i))[:2]
            assert retrieve(gg,w,ww)['selected_windows']==sorted(before+after)
    print('typed ledger/whole rejection/identity components/300 path oracles PASS',flush=True)


def main():
    out=ROOT/'runs/20261005_m1_provenance/cpu_checks';out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(out/'run.log'),logging.StreamHandler()])
    logging.info('host %s',socket.gethostname());(out/'run.pid').write_text(str(__import__('os').getpid()))
    entries=[dict(index=i,time=t) for i,t in enumerate((0.,.25,7.5,8.,10.,15.99,17.4))]
    assert select_indices(entries,[(0.,8.),(8.,16.),(16.,24.),(24.,25.)])==[[2],[5],[6],[]]
    graph_checks();asr={ds:load_asr(ds) for ds in ('HateMM','HateClipSeg')};j=cpu_renderer();records=[]
    for row in selected_rows(True):
        folder=out/'sources'/row['dataset']/row['video_id'];folder.mkdir(parents=True,exist_ok=True)
        source=acquire_window_frames(row,folder/'frames');segments=asr[row['dataset']].get(row['video_id'],[])
        windows=windows_for(row,segments,source,folder);sizes=[]
        # All current window bodies/coordinates are rendered. Expanded image
        # token audit covers the longest plain window, no GT/prediction.
        overview=native_frames(row);rank=[]
        for w in windows:
            content,paths=ledger_content(overview,w);rendered=j.render([j.turn('system',LEDGER_SYSTEM),dict(role='user',content=content)],True)
            rank.append((len(j.tok.encode(rendered,add_special_tokens=False)),w['i']))
        _,largest=max(rank);content,paths=ledger_content(overview,windows[largest])
        from PIL import Image
        images=[Image.open(ROOT/p).convert('RGB') for p in paths]
        try:enc=j.encode(j.render([j.turn('system',LEDGER_SYSTEM),dict(role='user',content=content)],True),images)
        finally:
            for image in images:image.close()
        record=dict(dataset=row['dataset'],video_id=row['video_id'],windows=len(windows),
            selected_frames=sum(len(w['frames']) for w in windows),uncovered=source['uncovered_windows'],
            actual_source_decode_seconds=source['decode_seconds'],largest_plain_window=largest,
            largest_plain_tokens=max(rank)[0],actual_expanded_tokens=len(enc['input_ids'][0]))
        (folder/'source.json').write_text(json.dumps(source)+'\n');records.append(record);logging.info('%s',record)
    (out/'summary.json').write_text(json.dumps(dict(host=socket.gethostname(),GT_read=False,
        scope='CPU fixtures and actual fixed5 source/PTS/token input audit, not actual Qwen perceptions or performance',records=records),indent=2)+'\n')
    logging.info('CPU_CHECKS_DONE')


if __name__=='__main__':main()
