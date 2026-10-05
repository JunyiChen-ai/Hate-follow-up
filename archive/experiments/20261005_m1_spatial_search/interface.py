"""Bounded source conditions and actual-region search responses, never scores."""
import json
import re
from geometry import ROOT,SPEC,children,validate_box


def canonical(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))


def write_detector(stream,frame_ids):
    assert frame_ids
    stream.force('{"kind":"');kind=stream.choose(['SEARCH','NONE','UNKNOWN']);stream.force('"')
    result=dict(kind=kind)
    if kind=='SEARCH':
        stream.force(',"frame":"');frame=stream.choose(frame_ids);stream.force('","target":"')
        stream.description();target=stream.events[-1]['text'];result.update(frame=frame,target=target)
    stream.force('}');return result


def write_search(stream,node_box):
    allowed=['FOUND','UNKNOWN']+(['TARGET_CUE','CONTEXT_CUE'] if children(node_box) else [])
    stream.force('{"kind":"');kind=stream.choose(allowed);stream.force('"');result=dict(kind=kind)
    if kind=='FOUND':
        stream.force(',"box":[');scale=SPEC['coordinate_scale'];coords=[]
        for index in range(4):
            low=0 if index<2 else coords[index-2]+1;high=scale-1 if index<2 else scale
            terminal=',' if index<3 else ']'
            chosen=stream.choose([str(n)+terminal for n in range(low,high+1)])
            coords.append(int(chosen[:-1]))
        result['box']=coords
    elif kind in ('TARGET_CUE','CONTEXT_CUE'):
        stream.force(',"cue":"');stream.description();result['cue']=stream.events[-1]['text']
        stream.force(',"order":[');order=[]
        for index in range(4):
            terminal=',' if index<3 else ']';remain=[i for i in range(4) if i not in order]
            chosen=stream.choose([str(i)+terminal for i in remain]);order.append(int(chosen[:-1]))
        result['order']=order
    stream.force('}');return result


def detector_record(g,frame_ids):
    if g['truncated']:return dict(kind='UNKNOWN',reason='whole_generation_cap')
    result=g['selection'];assert isinstance(result,dict)
    if result['kind']=='SEARCH':
        assert set(result)=={'kind','frame','target'} and result['frame'] in frame_ids
        if not result['target'].strip() or re.match(r'^UNKNOWN\b',result['target'].lstrip(),flags=re.IGNORECASE):return dict(kind='UNKNOWN',reason='unavailable_target')
        assert len(result['target'].split())<=SPEC['description_words']
    else:assert set(result)=={'kind'} and result['kind'] in ('NONE','UNKNOWN')
    return result


def search_record(g,node_box):
    if g['truncated']:return dict(kind='UNKNOWN',reason='whole_generation_cap')
    result=g['selection'];assert isinstance(result,dict)
    if result['kind']=='FOUND':
        assert set(result)=={'kind','box'};validate_box(result['box'],(SPEC['coordinate_scale'],)*2)
    elif result['kind'] in ('TARGET_CUE','CONTEXT_CUE'):
        assert set(result)=={'kind','cue','order'} and children(node_box)
        assert sorted(result['order'])==list(range(4)) and len(result['cue'].split())<=SPEC['description_words']
        if not result['cue'].strip() or re.match(r'^UNKNOWN\b',result['cue'].lstrip(),flags=re.IGNORECASE):return dict(kind='UNKNOWN',reason='unavailable_cue')
    else:assert result==dict(kind='UNKNOWN')
    return result


def frame_visible(frame):return {k:frame[k] for k in ('id','time','shape')}


def detector_content(window,segments):
    content=[dict(type='text',text='Current window and original transcript:\n'+canonical(dict(i=window['i'],start=window['start'],end=window['end'],body=window['body'],segments=[list(s) for s in segments]))+'\nActual LOCAL frames follow.\n')]
    paths=[]
    for frame in window['frames']:
        content.extend([dict(type='text',text=canonical(frame_visible(frame))+'\n'),dict(type='image')]);paths.append(frame['path'])
    content.append(dict(type='text',text='Return one required-but-unclear observable target, or NONE/UNKNOWN. A target is a search condition, not a witnessed fact.'))
    return content,paths


def search_content(window,frame,target,node,path):
    pieces=children(node['box'])
    text=canonical(dict(window=dict(i=window['i'],start=window['start'],end=window['end'],body=window['body']),
        source=frame_visible(frame),request=target,original_pixel_region=node['box'],
        children=[dict(id=i,original_pixel_box=b) for i,b in enumerate(pieces)]))
    return [dict(type='text',text=text+'\nThe following image is this exact region. FOUND box coordinates refer to the displayed region, normalized0–1000.\n'),dict(type='image')],[path]
