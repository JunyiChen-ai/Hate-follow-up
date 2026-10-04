"""Native source inventory and replay of the saved executed acquisition."""
import json
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.video_inputs import load_manifest,frame_paths,fixed_windows
from src.mllm_judge import MODEL
from program import CACHE_VERSION,CONSTANTS,parse_chunk,execute
DATASETS=('HateMM','HateClipSeg')
CACHE=ROOT/'data/temporal_evidence_program'


def selected_rows(smoke):
    rows=load_manifest(ROOT/'data/omsl_v6_inputs/manifests/all_test.jsonl',DATASETS)
    if smoke:
        rows=[r for ds in DATASETS for r in [x for x in rows if x['dataset']==ds][:2]]+[
            r for r in rows if r['dataset']=='HateMM' and r['video_id']=='hate_video_114']
    assert len(rows)==(5 if smoke else 333)
    return rows


def sources(row,segments):
    frames=frame_paths(row['dataset'],row['video_id'],20)
    assert frames and all(p.is_file() for t,p in frames)
    source=dict(frames=[dict(id=i,time=t,time_source='native_filename_nominal') for i,(t,p) in enumerate(frames)],
        segments=[dict(id=i,start=s,end=e,text=text,time_source='native_asr_segment_proportional_chars') for i,(s,e,text) in enumerate(segments)])
    windows=[dict(id=i,start=a,end=b) for i,(a,b) in enumerate(fixed_windows(float(row['duration']),8))]
    return source,windows,[str(p.relative_to(ROOT)) for t,p in frames]


def validate(metadata,row,segments):
    source,windows,paths=sources(row,segments)
    assert metadata['cache_version']==CACHE_VERSION and metadata['constants']==CONSTANTS and metadata['model']==MODEL
    assert metadata['dataset']==row['dataset'] and metadata['video_id']==row['video_id']
    assert metadata['duration']==float(row['duration']) and metadata['manifest_video_path']==row['video_path']
    assert metadata['GT_read'] is False and metadata['source']==source and metadata['frame_paths']==paths
    assert len(metadata['windows'])==len(windows)
    plans={};rejects=[]
    for chunk,offset in zip(metadata['chunks'],range(0,len(windows),CONSTANTS['planner_windows'])):
        requested=windows[offset:offset+CONSTANTS['planner_windows']]
        assert chunk['requested_windows']==requested
        parsed,rejected=parse_chunk(chunk['generation']['text'],requested,chunk['generation']['truncated'])
        assert chunk['plans']=={str(k):v for k,v in parsed.items()} and chunk['rejected']==rejected
        plans.update(parsed);rejects.extend(rejected)
    assert len(metadata['chunks'])==(len(windows)+CONSTANTS['planner_windows']-1)//CONSTANTS['planner_windows']
    for w,saved in zip(windows,metadata['windows']):
        calls=iter(saved['execution']['calls'])
        def replay(kind,packet):
            c=next(calls)
            assert c['kind']==kind and c['packet']==packet
            return c['raw']
        rebuilt=execute(source,w,plans[w['id']],replay)
        assert rebuilt==saved['execution'] and saved['window']==w
        assert next(calls,None) is None
        assert len(saved['module_generations'])==len(rebuilt['calls'])
        for call,g in zip(rebuilt['calls'],saved['module_generations']):
            assert g['text']==call['raw'] and g['kind']==call['kind'] and g['packet']==call['packet']
    return metadata
