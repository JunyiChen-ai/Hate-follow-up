"""Real-source temporal retrieval and deterministic visual tools, no labels."""
import json
from pathlib import Path
import sys
import os
import math
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
INTERFACE=os.environ.get('SOURCE_INTERFACE','A');assert INTERFACE in ('A','B','C')
OUTPUT_SUFFIX='' if INTERFACE=='A' else '_'+INTERFACE
SPEC=json.loads((Path(__file__).parent/('spec.json' if INTERFACE=='A' else f'spec_{INTERFACE}.json')).read_text())
GLYPHS={
 '0':['111','101','101','101','111'],'1':['010','110','010','010','111'],
 '2':['111','001','111','100','111'],'3':['111','001','111','001','111'],
 '4':['101','101','111','001','001'],'5':['111','100','111','001','111'],
 '6':['111','100','111','101','111'],'7':['111','001','010','010','010'],
 '8':['111','101','111','101','111'],'9':['111','101','111','001','111'],
 '.':['000','000','000','000','010'],'/':['001','001','010','100','100'],
 's':['000','111','100','011','111']}


def uniform(indices,cap):
    return list(indices) if len(indices)<=cap else [indices[int(k)] for k in np.linspace(0,len(indices)-1,cap)]


def retrieval(values,windows):
    assert len(values)==len(windows)
    valid=[i for i,v in enumerate(values) if type(v) is int and 0<=v<=10]
    chosen=sorted(sorted(valid,key=lambda i:(-values[i],i))[:SPEC['retrieval_top_windows']]);groups=[]
    for i in chosen:
        if groups and i==groups[-1][-1]+1:groups[-1].append(i)
        else:groups.append([i])
    return dict(selected_windows=chosen,intervals=[dict(id=k,start=windows[g[0]]['start'],end=windows[g[-1]]['end'],windows=g) for k,g in enumerate(groups)])


def draw(image,frame_time,duration,highlights=(),current=None):
    assert duration>0 and 0<=frame_time<duration
    width,height=image.size;canvas=Image.new('RGB',(width,height+SPEC['bar_height']),tuple(SPEC['bar_background']));canvas.paste(image,(0,0));paint=ImageDraw.Draw(canvas)
    margin=min(SPEC['bar_margin'],width//8);left=margin;right=max(left,width-margin-1)
    coordinate=lambda t:left+math.floor(np.clip(t/duration,0,1)*(right-left))
    paint.rectangle((left,height+14,right,height+24),fill=tuple(SPEC['bar_remaining']))
    cursor=coordinate(frame_time);paint.rectangle((left,height+14,cursor,height+24),fill=tuple(SPEC['bar_progress']))
    for a,b in highlights:
        assert 0<=a<b<=duration
        paint.rectangle((coordinate(a),height+30,coordinate(b),height+36),fill=tuple(SPEC['bar_highlight']))
    if current is not None:
        a,b=current;assert 0<=a<b<=duration
        paint.rectangle((coordinate(a),height+14,coordinate(b),height+24),outline=tuple(SPEC['bar_current']),width=2)
    paint.ellipse((cursor-3,height+11,cursor+3,height+27),fill=tuple(SPEC['bar_cursor']))
    x=margin
    for ch in f'{frame_time:.1f}s/{duration:.1f}s':
        for y,row in enumerate(GLYPHS[ch]):
            for col,value in enumerate(row):
                if value=='1':paint.rectangle((x+col*2,height+44+y*2,x+col*2+1,height+44+y*2+1),fill=(0,0,0))
        x+=8
    assert np.array_equal(np.asarray(canvas)[:height],np.asarray(image.convert('RGB')))
    return canvas


def execute(state,plan,windows,tables,sample_ids):
    state=json.loads(json.dumps(state));action=plan['action'];result=dict(action=action,executed=False)
    if action=='PROGRESS_BAR':
        assert not state['progress'];state['progress']=True;result['executed']=True
    elif action=='HIGHLIGHT':
        assert state['progress'] and state['query_id'] is None
        q=plan['query_id'];assert 0<=q<len(tables);table=tables[q]
        if table['intervals']:
            state['query_id']=q;state['highlights']=[[r['start'],r['end']] for r in table['intervals']];result['executed']=True
    elif action=='CUT':
        assert state['progress'] and state['query_id'] is not None and not state['cut']
        q=state['query_id'];r=tables[q]['intervals'][plan['interval_id']]
        ids=[k for i in r['windows'] for k in windows[i]['sample_ids']];ids=uniform(sorted(set(ids)),SPEC['cut_memory_frames'])
        if ids:
            state['memory_ids']=ids;state['cut']=dict(query_id=q,interval_id=r['id'],bounds=[r['start'],r['end']]);result['executed']=True
    else:assert action in ('TERMINATE','UNKNOWN')
    assert set(state['memory_ids'])<=set(sample_ids)
    return state,result
