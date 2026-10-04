"""Current native inputs and actual two-frame window witnesses; no annotations."""
from pathlib import Path
import sys
import math
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.video_inputs import load_manifest,frame_paths,fixed_windows,window_text
from src.actual_video_frames import validate_frames
from src.mllm_judge import MODEL
from graph import (VERSION,CONSTANTS,LEDGER_SYSTEM,LEDGER_SCHEMA,LEDGER_INSTRUCTION,
    LINK_SYSTEM,LINK_INSTRUCTION,canonical,parse_ledger,parse_links,compile_graph,retrieve)

DATASETS=('HateMM','HateClipSeg')
CACHE=ROOT/'data/temporal_entity_discourse_graph_handles'


def selected_rows(smoke=False):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:
        rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[
            r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def windows_for(row,segments,source,folder):
    entries={e['index']:e for e in source['entries']};windows=[]
    for i,((a,b),indices) in enumerate(zip(fixed_windows(float(row['duration']),8),source['selected_indices'])):
        frames=[]
        for k,index in enumerate(indices):
            entry=entries[index]
            frames.append(dict(id=f'p{k}',path=str((folder/'frames'/f'frame_{index:08d}.png').relative_to(ROOT)),**entry))
        windows.append(dict(i=i,start=a,end=b,body=window_text(segments,a,b),frames=frames))
    return windows


def native_frames(row):
    return [dict(id=f'overview{k}',time=t,path=str(p.relative_to(ROOT)),time_kind='nominal overview only')
        for k,(t,p) in enumerate(frame_paths(row['dataset'],row['video_id'],20))]


from handle_interface import (VERSION,CONSTANTS,ledger_content,link_content,write_ledger,write_links,compiled_ledger,compiled_links)

def validate(m,row,segments,renderer=None):
    folder=CACHE/row['dataset']/row['video_id']
    assert (m['version'],m['constants'],m['model'])==(VERSION,CONSTANTS,MODEL)
    assert (m['dataset'],m['video_id'],m['duration'])==(row['dataset'],row['video_id'],float(row['duration']))
    assert m['GT_read'] is False and m['segments']==[list(s) for s in segments]
    validate_frames(m['source'],row,folder/'frames');windows=windows_for(row,segments,m['source'],folder)
    assert m['windows']==windows and m['overview']==native_frames(row) and len(m['ledgers'])==len(windows)
    ledgers=[];generations=[]
    from src.structured_source_generation import validate_generation
    for w,l in zip(windows,m['ledgers']):
        parsed=compiled_ledger(l['generation'],w);assert l['parsed']==parsed;ledgers.append(parsed);generations.append(l['generation'])
        if renderer:
            content,paths=ledger_content(m['overview'],w)
            validate_generation(renderer,ROOT,l['generation'],LEDGER_SYSTEM,content,paths,512,lambda stream:write_ledger(stream,w))
    local=compile_graph(ledgers,[]);links=[]
    expected=[list(range(i,min(i+8,len(windows)))) for i in range(0,len(windows),8)]
    assert len(m['links'])==len(expected)
    for anchors,link in zip(expected,m['links']):
        assert link['anchors']==anchors
        assert link['input_size_before_prefill']==len(link['generation']['input_tokens'])
        parsed=compiled_links(link['generation'],local['nodes'],anchors);assert link['parsed']==parsed;links.append(parsed);generations.append(link['generation'])
        if renderer:validate_generation(renderer,ROOT,link['generation'],LINK_SYSTEM,link_content(local,anchors),[],2048,lambda stream:write_links(stream,local,anchors))
    graph=compile_graph(ledgers,links);assert graph==m['graph']
    assert m['packets']==[retrieve(graph,i,windows) for i in range(len(windows))]
    assert m['actual_forwards']==sum(g['actual_forwards'] for g in generations)
    assert m['actual_vision_forwards']==sum(g['actual_vision_forwards'] for g in generations)==len(windows)
    assert abs(m['generation_seconds']-sum(g['seconds'] for g in generations))<1e-6
    assert m['standalone_seconds']>=m['generation_seconds']+m['source']['decode_seconds'] and m['peak_GiB']>=0
    assert all(math.isfinite(m[k]) for k in ('standalone_seconds','generation_seconds','peak_GiB'))
    return m
