"""Source-handle interface B source inventory and independent execution replay."""
import json
from inputs import ROOT,DATASETS,selected_rows,sources
from src.mllm_judge import MODEL
from program import execute
from handle_interface import CACHE_VERSION,CONSTANTS,compile_selection
CACHE=ROOT/'data/temporal_evidence_program_handles'


def plans_for(source,requested,g):
    if g['truncated']:
        return {w['id']:dict(status='UNKNOWN',reason='truncated_output',ops=[]) for w in requested}
    selected=json.loads(g['text']);assert selected==g['selection'] and len(selected)==len(requested)
    return {w['id']:compile_selection(source,w,s) for w,s in zip(requested,selected)}


def validate(metadata,row,segments):
    source,windows,paths=sources(row,segments)
    assert metadata['cache_version']==CACHE_VERSION and metadata['constants']==CONSTANTS and metadata['model']==MODEL
    assert metadata['dataset']==row['dataset'] and metadata['video_id']==row['video_id']
    assert metadata['duration']==float(row['duration']) and metadata['manifest_video_path']==row['video_path']
    assert metadata['GT_read'] is False and metadata['source']==source and metadata['frame_paths']==paths
    assert len(metadata['windows'])==len(windows)
    plans={}
    for chunk,offset in zip(metadata['chunks'],range(0,len(windows),CONSTANTS['planner_windows'])):
        requested=windows[offset:offset+CONSTANTS['planner_windows']]
        assert chunk['requested_windows']==requested
        rebuilt=plans_for(source,requested,chunk['generation'])
        assert chunk['plans']=={str(k):v for k,v in rebuilt.items()};plans.update(rebuilt)
    assert len(metadata['chunks'])==(len(windows)+CONSTANTS['planner_windows']-1)//CONSTANTS['planner_windows']
    for w,saved in zip(windows,metadata['windows']):
        calls=iter(saved['execution']['calls'])
        def replay(kind,packet):
            c=next(calls);assert c['kind']==kind and c['packet']==packet;return c['raw']
        rebuilt=execute(source,w,plans[w['id']],replay)
        assert rebuilt==saved['execution'] and saved['window']==w and next(calls,None) is None
        assert len(saved['module_generations'])==len(rebuilt['calls'])
        for call,g in zip(rebuilt['calls'],saved['module_generations']):
            assert g['text']==call['raw'] and g['kind']==call['kind'] and g['packet']==call['packet']
    return metadata
