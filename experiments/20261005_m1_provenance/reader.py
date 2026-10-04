"""Literal retrieved sources enter fresh independent multimodal local readings."""
import copy
import math
from PIL import Image
import torch
from graph import PACKET_HEADER,canonical
from inputs import ROOT
from src.mllm_judge import yesno_question
from src.source_generation import clock


def field_subset(graph,w,kind):
    nodes=[n for n in graph['nodes'] if n['window']==w]
    if kind=='visual':nodes=[n for n in nodes if n['kind'] in ('entity','action') and n.get('frame') is not None]
    else:nodes=[n for n in nodes if n['kind']!='action']
    available={n['id'] for n in nodes}
    edges=[e for e in graph['edges'] if e['from'] in available and e['to'] in available]
    if kind=='visual':edges=[e for e in edges if e['type'] in ('actor','target','same_entity')]
    return dict(nodes=nodes,local_relations=edges)


def compile_content(metadata,i,kind):
    windows=metadata['windows'];local=windows[i];packet=metadata['packets'][i]
    content=[dict(type='text',text=PACKET_HEADER)];paths=[];records=[]
    for w in packet['selected_windows']+[i]:
        source=windows[w];role='local' if w==i else 'context';fields=field_subset(metadata['graph'],w,kind)
        record=dict(window=w,role=role,start=source['start'],end=source['end'],fields=fields,
            frames=source['frames'],body=source['body'] if kind=='speech' else None)
        records.append(record)
        content.append(dict(type='text',text=f'[source_window={w}; role={role}; t={source["start"]:.6f}-{source["end"]:.6f}s]\n'))
        for frame in source['frames']:
            content.extend([dict(type='text',text=f'[frame={frame["id"]}; actual_t={frame["time"]:.6f}s; index={frame["index"]}]\n'),dict(type='image')]);paths.append(frame['path'])
        content.append(dict(type='text',text=canonical(fields)+'\n'))
        if kind=='speech':content.append(dict(type='text',text='Literal source speech:\n'+source['body']+'\n'))
    selected={i,*packet['selected_windows']};allowed=[]
    for witness in packet['witnesses']:
        for edge in witness['path']:
            if kind=='visual' and edge['type'] not in ('actor','target','same_entity'):continue
            if edge not in allowed:allowed.append(edge)
    content.append(dict(type='text',text='Source paths:\n'+canonical(allowed)+'\n'+
        yesno_question(i,len(windows),local['start'],local['end'],local['body'],kind)))
    return content,paths,dict(packet=packet,records=records,relations=allowed)


def encoding(j,ctx,metadata,i,kind):
    content,paths,records=compile_content(metadata,i,kind)
    messages=copy.deepcopy(ctx['msgs'])+copy.deepcopy(ctx['history'])+[dict(role='user',content=content)]
    text=j.render(messages,True);allpaths=[str(p.relative_to(ROOT)) for p in ctx['files']]+paths
    images=[Image.open(ROOT/p).convert('RGB') for p in allpaths]
    try:enc=j.encode(text,images)
    finally:
        for image in images:image.close()
    trace=dict(messages=messages,prompt=text,image_paths=allpaths,records=records,
        input_tokens=enc['input_ids'][0].tolist(),image_grid=enc['image_grid_thw'].tolist())
    assert trace['input_tokens'][:ctx['stance_cache_tokens']]==ctx['native_stance_ids']
    return enc,trace


@torch.no_grad()
def read(j,ctx,metadata,i,kind):
    before=j.forward_calls;start=clock(j);enc,trace=encoding(j,ctx,metadata,i,kind)
    j.model.model.rope_deltas=None
    output=j.model.model(**j.model_inputs(enc),use_cache=False)
    z=j.margins_fp32(output.last_hidden_state[0,-1:])[0];del output
    trace.update(margin=z,seconds=clock(j)-start,actual_forwards=j.forward_calls-before)
    assert trace['actual_forwards']==1
    return z,trace


def validate_trace(j,ctx,metadata,i,kind,trace):
    _,expected=encoding(j,ctx,metadata,i,kind)
    assert all(trace[k]==v for k,v in expected.items())
    assert trace['actual_forwards']==1 and math.isfinite(trace['seconds']) and trace['seconds']>=0
