"""Bounded literal text and frame-owned boxes; no task labels or judgments."""
import json
import re
from pathlib import Path
import sys
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.structured_source_generation import Capped
SPEC=json.loads((Path(__file__).parent/'spec.json').read_text())


def json_string_prefix(raw):
    """Accept complete or unfinished escaped JSON string contents."""
    i=0
    while i<len(raw):
        c=raw[i]
        if ord(c)<32 or c=='"' or c=='\ufffd':return False,False,None
        if c=='\\':
            i+=1
            if i==len(raw):return True,False,None
            c=raw[i]
            if c=='u':
                remaining=raw[i+1:i+5]
                if any(x not in '0123456789abcdefABCDEF' for x in remaining):return False,False,None
                if len(remaining)<4:return True,False,None
                i+=4
            elif c not in '"\\/bfnrt':return False,False,None
        i+=1
    try:text=json.loads('"'+raw+'"')
    except json.JSONDecodeError:return False,False,None
    return True,True,text


def literal(stream):
    """Escaped punctuation can be generated; source text is never interpolated."""
    start=len(stream.tokens);content=[];raw='';reason='model_quote'
    quotes=stream.j.tok.encode('"',add_special_tokens=False);assert len(quotes)==1
    quote=quotes[0]
    if not hasattr(stream.j,'literal_json_tokens'):
        contexts=('', '\\', '\\u', '\\u0', '\\u00', '\\u000')
        stream.j.literal_json_tokens=[i for i in range(len(stream.j.tok))
            if i not in stream.j.tok.all_special_ids and stream.j.tok.decode([i])
            and any(json_string_prefix(prefix+stream.j.tok.decode([i]))[0] for prefix in contexts)]
    while True:
        valid,complete,text=json_string_prefix(raw);assert valid
        if len(content)>=SPEC['text_tokens'] or (complete and len(text.split())>=SPEC['text_words']):
            if not complete:
                # Close syntax only; the entire capped field is UNKNOWN below.
                # This does not convert a partial escape into accepted evidence.
                index=0;completion=None
                while index<len(raw):
                    if raw[index]!='\\':index+=1;continue
                    if index+1==len(raw):completion='n';break
                    if raw[index+1]=='u':
                        available=len(raw[index+2:index+6])
                        if available<4:completion='0'*(4-available);break
                        index+=6
                    else:index+=2
                assert completion is not None
                stream.force(completion);raw+=completion
            stream.append(quote);reason='field_cap';break
        allowed=stream.j.literal_json_tokens+([quote] if quote not in stream.j.literal_json_tokens else [])
        if stream.replay:
            if len(stream.tokens)>=len(stream.saved):raise Capped()
            token=stream.saved[len(stream.tokens)];assert token in allowed
        else:
            ids=torch.tensor(allowed,device=stream.j.device,dtype=torch.long)
            ranked=stream.logits()[ids].clone()
            while True:
                best=int(ranked.argmax());token=allowed[best]
                if token==quote and complete:break
                candidate=stream.j.tok.decode(content+[token]);okay,closed,decoded=json_string_prefix(candidate)
                if okay and (not closed or len(decoded.split())<=SPEC['text_words']):break
                ranked[best]=-torch.inf
        stream.append(token)
        if token==quote and complete:break
        content.append(token);raw=stream.j.tok.decode(content)
        okay,closed,decoded=json_string_prefix(raw);assert okay
        if closed:assert len(decoded.split())<=SPEC['text_words']
    okay,complete,text=json_string_prefix(raw);assert okay and complete
    stream.events.append(dict(kind='literal_json',text=text,encoded_content=raw,reason=reason,start=start,end=len(stream.tokens)))
    return text,reason=='field_cap'


def writer(frames):
    def emit(stream):
        result=[];stream.force('{"frames":[')
        for i,frame in enumerate(frames):
            if i:stream.force(',')
            stream.force('{"id":'+json.dumps(frame['id'])+',"status":"')
            status=stream.choose(['TEXT",','NONE",','UNKNOWN",'])[:-2]
            record=dict(id=frame['id'],status=status)
            if status=='TEXT':
                stream.force('"box":[');box=[]
                for axis in range(4):
                    low=0 if axis<2 else box[axis-2]+1
                    high=999 if axis<2 else 1000
                    end=',' if axis<3 else ']'
                    value=stream.choose([str(v)+end for v in range(low,high+1)])
                    box.append(int(value[:-1]))
                stream.force(',"text":"');text,capped=literal(stream)
                record.update(box=box,text=text,field_capped=capped)
            else:stream.force('"text":"UNKNOWN"')
            stream.force('}');result.append(record)
        stream.force(']}');return result
    return emit


def compile_records(generation,frames):
    if generation['truncated'] or len(generation['tokens'])>=generation['max_tokens']:
        return [dict(id=f['id'],status='UNKNOWN') for f in frames]
    records=generation['selection'];assert len(records)==len(frames)
    result=[]
    for r,f in zip(records,frames):
        assert r['id']==f['id'];record=dict(r)
        if r['status']=='TEXT':
            text=r['text'];invalid=r['field_capped'] or not text.strip() or re.match(r'^UNKNOWN\b',text.lstrip(),re.I)
            invalid=invalid or any(0xD800<=ord(c)<=0xDFFF or c=='\ufffd' for c in text)
            if invalid:record=dict(id=f['id'],status='UNKNOWN')
            else:
                assert len(text.split())<=SPEC['text_words']
                x0,y0,x1,y1=r['box'];assert 0<=x0<x1<=1000 and 0<=y0<y1<=1000
        result.append(record)
    return result


def content(frames,kind,window=None):
    description='Read at most one literal visible screen-text region in each numbered frame. Return through the supplied bounded frame/status/box/text interface. Boxes are normalized integers0..1000 in the FULL original frame. NONE means no visible text; UNKNOWN means it cannot be read. Text must be literal, including punctuation; do not describe the scene or guess omitted words.'
    if kind=='repair':description+=' The second image is only a candidate region; choose the current box relative to the FIRST full original frame. No earlier words or identity are supplied.'
    if window is not None:description+=f' Window [{window[0]},{window[1]}) seconds.'
    items=[dict(type='text',text=description)];paths=[]
    for frame in frames:
        items.append(dict(type='text',text=f'Frame {frame["id"]}; actual PTS time {frame["time"]:.9f}s; original width,height {frame["shape"]}.'))
        items.append(dict(type='image',image=frame['path']));paths.append(frame['path'])
    return items,paths
