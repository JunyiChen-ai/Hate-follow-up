"""Restricted temporal/source interpreter; no predictions, labels or model loading."""
import copy
import json
import math

CACHE_VERSION='R1 executable temporal evidence program; sources2026-10-05'
CONSTANTS=dict(window_seconds=8,planner_windows=8,planner_tokens=2048,module_tokens=96,
    max_ops=12,max_perception=2,max_context=2,context_chars=96,description_words=12,
    char_boundary_tolerance=1e-9,seed=0)
PLANNER_SYSTEM='You write source-grounded evidence programs, not moderation decisions. Use only the supplied indexed video sources and the API below. Select observable sources, then compose temporal and discourse evidence. Do not classify hate, supply confidence scores, invent sources, or output a rationale. Return exactly one JSON array and no other text. Include each requested window exactly once. Use an empty ops list if no program is supported.'
API='''All IDs and offsets are zero-based integers. Source times are nominal cache coordinates; transcript character times are proportional estimates, not measured word times. Character offsets count Unicode characters in the supplied original segment text. Windows and character spans are half-open. Use at most 12 operations and 2 perception calls per window. Variable names must be unique; arguments may reference only earlier variables in that window. Context interprets a local utterance and does not establish a local occurrence. Output schema: [{"window":integer,"ops":[operation,...]},...].
API:
["span",name,segment,start_char,end_char] -> exact original transcript substring and its proportional time interval.
["frame",name,frame_index] -> the original cached frame and nominal time.
["local",name,input_name] -> fully contained characters of a span in the requested window, or an in-window frame; otherwise UNKNOWN.
["context",name,span_name] -> interpretation-only transcript context, at most 2 spans, first 96 characters each.
["scope",name,local_span_name,[context_names]] -> fresh factual reading of the local utterance with those contexts: speaker role, mode, exact target span reference, and support.
["action",name,local_frame_name] -> fresh factual reading of the selected local frame: visible actor, action, target, and support.
["join",name,span_name,frame_name] -> paired local witnesses only if span.start <= frame.time < span.end and both are in the requested window; otherwise UNKNOWN.
["emit",[names]] -> existing values and their dependencies; use at most once, as the last operation.
Do not use other operations or add decision fields.'''
PLANNER_END='Return programs for requested_windows using only the supplied sources and API.'
MODULE_SYSTEM='You describe only the supplied source content. Do not classify hate or infer unsupported facts. Return only the requested JSON; use UNKNOWN when uncertain.'
SCOPE_QUESTION='Identify the speaker role (speaker/quoted/reported/UNKNOWN), mode (direct/quoted/rejected/reported/UNKNOWN), the exact target span reference if present, and supporting source references for this utterance. Interpretation context is not an additional local utterance. Return keys speaker,mode,target,support.'
ACTION_QUESTION='Describe the visible actor, action and target in this frame. Return keys actor,action,target,support. Keep each description at most twelve words and support only by the supplied frame identifier; use UNKNOWN where not visible.'
EVIDENCE_HEADER='Executed local evidence (uncertain measurements, not a moderation decision):\n'


def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def integer(x):return type(x) is int
def unknown(reason):return dict(kind='UNKNOWN',reason=reason)


def parse_chunk(raw,requested,truncated=False):
    """No code fences, repair, permissive trailing JSON, or duplicate-window choice."""
    ids=[w['id'] for w in requested];plans={};rejected=[]
    if truncated:return {i:dict(status='UNKNOWN',reason='truncated_output',ops=[]) for i in ids},[dict(reason='truncated_output',raw=raw)]
    try:
        arr=json.loads(raw)
        if not isinstance(arr,list):raise ValueError('expected_array')
    except (ValueError,TypeError):
        return {i:dict(status='UNKNOWN',reason='invalid_array',ops=[]) for i in ids},[dict(reason='invalid_array',raw=raw)]
    seen={i:[] for i in ids}
    for item in arr:
        if not isinstance(item,dict) or set(item)!=set(('window','ops')) or not integer(item.get('window')) or item['window'] not in seen:
            rejected.append(dict(reason='invalid_or_out_of_chunk_window',entry=item));continue
        seen[item['window']].append(item)
    for i,items in seen.items():
        if len(items)!=1:plans[i]=dict(status='UNKNOWN',reason='duplicate_window' if items else 'missing_window',ops=[])
        elif not isinstance(items[0]['ops'],list):plans[i]=dict(status='UNKNOWN',reason='invalid_ops',ops=[])
        else:plans[i]=dict(status='valid',ops=items[0]['ops'])
        if len(items)>1:rejected.append(dict(reason='duplicate_window',window=i,entries=items))
    return plans,rejected


def span(source,segment,a,b,kind='span'):
    segs=source['segments']
    if not all(integer(v) for v in (segment,a,b)) or not 0<=segment<len(segs):return unknown('invalid_span')
    s=segs[segment];length=len(s['text'])
    if not 0<=a<b<=length or s['end']<=s['start']:return unknown('invalid_span')
    d=s['end']-s['start']
    return dict(kind=kind,ref=dict(segment=segment,start_char=a,end_char=b),text=s['text'][a:b],
        start=s['start']+a*d/length,end=s['start']+b*d/length,time_source=s['time_source'])


def local(source,obj,window):
    if obj['kind']=='frame':
        return {**obj,'local':True,'local_window':window['id']} if window['start']<=obj['time']<window['end'] else unknown('frame_outside_window')
    if obj['kind']!='span':return unknown('local_requires_span_or_frame')
    r=obj['ref'];s=source['segments'][r['segment']];length=len(s['text']);d=s['end']-s['start'];tol=CONSTANTS['char_boundary_tolerance']
    a=max(r['start_char'],math.ceil((window['start']-s['start'])*length/d-tol),0)
    b=min(r['end_char'],math.floor((window['end']-s['start'])*length/d+tol),length)
    value=span(source,r['segment'],a,b)
    return {**value,'local':True,'local_window':window['id']} if value['kind']!='UNKNOWN' else unknown('no_contained_characters')


def in_window(obj,window):
    if not obj.get('local') or obj.get('local_window')!=window['id']:return False
    if obj['kind']=='frame':return window['start']<=obj['time']<window['end']
    if obj['kind']=='span':return True  # only the declared character clip sets local_window
    return False


def normalize_module(kind,packet,raw):
    """Only declared fields and support IDs from this actual call survive."""
    try:answer=json.loads(raw)
    except (ValueError,TypeError):return unknown('invalid_module_json')
    keys={'speaker','mode','target','support'} if kind=='scope' else {'actor','action','target','support'}
    if not isinstance(answer,dict) or set(answer)!=keys:return unknown('invalid_module_schema')
    allowed={x['id']:x for x in packet['sources']}
    support=answer['support']
    if not isinstance(support,list) or not support or any(not isinstance(v,str) or v not in allowed for v in support):return unknown('unsupported_source')
    out=dict(kind=kind,support=sorted(set(support)))
    if kind=='scope':
        out['speaker']=answer['speaker'] if answer['speaker'] in ('speaker','quoted','reported','UNKNOWN') else 'UNKNOWN'
        out['mode']=answer['mode'] if answer['mode'] in ('direct','quoted','rejected','reported','UNKNOWN') else 'UNKNOWN'
        target=answer['target'];out['target']='UNKNOWN'
        if isinstance(target,dict) and set(target)=={'segment','start_char','end_char'} and all(integer(target[k]) for k in target):
            for x in allowed.values():
                r=x['value'].get('ref')
                if r and target['segment']==r.get('segment') and r['start_char']<=target['start_char']<target['end_char']<=r['end_char']:
                    out['target']=target;break
    else:
        for key in ('actor','action','target'):
            value=answer[key];out[key]=value if isinstance(value,str) and 0<len(value.split())<=CONSTANTS['description_words'] else 'UNKNOWN'
    out['sources']=copy.deepcopy(packet['sources'])
    return out


def execute(source,window,plan,perceive):
    """perceive(kind, actual packet) returns the raw JSON string; no score input."""
    if plan['status']!='valid':return dict(window=window['id'],status='UNKNOWN',reason=plan['reason'],emitted=[],trace=[],calls=[])
    env={};trace=[];calls=[];rejected=[];emitted=[];count=perception=context=0;has_emit=False
    for index,op in enumerate(plan['ops']):
        reason=None
        if not isinstance(op,list) or not op or not isinstance(op[0],str):reason='invalid_operation'
        elif has_emit:reason='operation_after_emit'
        elif count>=CONSTANTS['max_ops']:reason='operation_cap'
        else:
            kind=op[0];lens={'span':5,'frame':3,'local':3,'context':3,'scope':4,'action':3,'join':4,'emit':2}
            if kind not in lens or len(op)!=lens[kind]:reason='invalid_operation_schema'
            elif kind=='emit':
                if index!=len(plan['ops'])-1 or not isinstance(op[1],list) or any(not isinstance(v,str) or v not in env for v in op[1]):reason='invalid_emit'
            elif not isinstance(op[1],str) or not op[1] or op[1] in env:reason='invalid_variable'
            elif kind in ('local','context','action','join','scope'):
                refs=op[2:4] if kind=='join' else [op[2]]
                if kind=='scope':
                    if not isinstance(op[3],list) or len(op[3])>CONSTANTS['max_context']:reason='invalid_scope_context'
                    else:refs+=op[3]
                if any(not isinstance(v,str) or v not in env for v in refs):reason='forward_or_missing_reference'
            if reason is None and kind in ('scope','action') and perception>=CONSTANTS['max_perception']:reason='perception_cap'
            if reason is None and kind=='context' and context>=CONSTANTS['max_context']:reason='context_cap'
        if reason:
            rejected.append(dict(index=index,op=op,reason=reason));continue
        count+=1
        if kind=='emit':
            emitted=[dict(id=v,value=copy.deepcopy(env[v])) for v in op[1]];has_emit=True
            trace.append(dict(index=index,op=op,emitted=emitted));continue
        name=op[1]
        if kind=='span':value=span(source,*op[2:])
        elif kind=='frame':
            fid=op[2]
            value=dict(kind='frame',ref=dict(frame=fid),time=source['frames'][fid]['time'],time_source=source['frames'][fid]['time_source']) if integer(fid) and 0<=fid<len(source['frames']) else unknown('invalid_frame')
        elif kind=='local':value=local(source,env[op[2]],window)
        elif kind=='context':
            context+=1;obj=env[op[2]]
            if obj['kind']=='span':
                r=obj['ref'];value=span(source,r['segment'],r['start_char'],min(r['end_char'],r['start_char']+CONSTANTS['context_chars']),'context')
                value['interpretation_only']=True
            else:value=unknown('context_requires_span')
        elif kind=='join':
            s,f=env[op[2]],env[op[3]]
            if s['kind']=='span' and f['kind']=='frame' and in_window(s,window) and in_window(f,window) and s['start']<=f['time']<s['end']:
                value=dict(kind='join',span=copy.deepcopy(s),frame=copy.deepcopy(f),window=window['id'])
            else:value=unknown('no_local_temporal_join')
        elif kind in ('scope','action'):
            perception+=1;obj=env[op[2]];contexts=[env[v] for v in op[3]] if kind=='scope' else []
            expected='span' if kind=='scope' else 'frame'
            if obj['kind']!=expected or not in_window(obj,window) or any(c['kind']!='context' for c in contexts):value=unknown('invalid_perception_source')
            else:
                sources=[dict(id=op[2],value=copy.deepcopy(obj))]+[dict(id=v,value=copy.deepcopy(env[v])) for v in (op[3] if kind=='scope' else [])]
                packet=dict(window=copy.deepcopy(window),sources=sources)
                raw=perceive(kind,packet);value=normalize_module(kind,packet,raw)
                calls.append(dict(kind=kind,packet=packet,raw=raw,result=copy.deepcopy(value)))
        env[name]=value;trace.append(dict(index=index,op=op,id=name,value=copy.deepcopy(value)))
    return dict(window=window['id'],status='valid' if has_emit else 'UNKNOWN',
        reason=None if has_emit else 'missing_emit',emitted=emitted,trace=trace,calls=calls,rejected=rejected,
        accepted_operations=count,requested_perception=perception)


def branch_record(executed,kind):
    """Keep factual modality separation; join exposes frame plus source/time reference."""
    out=dict(window=executed['window'],status=executed['status'],reason=executed['reason'],evidence=[])
    for entry in executed['emitted']:
        value=entry['value'];k=value['kind']
        if kind=='visual' and k=='join':
            value=dict(kind='join',frame=value['frame'],span_ref=value['span']['ref'],span_interval=[value['span']['start'],value['span']['end']])
        elif kind=='visual' and k not in ('action','UNKNOWN'):continue
        elif kind=='speech' and k not in ('span','scope','context','UNKNOWN'):continue
        out['evidence'].append(dict(id=entry['id'],value=value))
    return out
