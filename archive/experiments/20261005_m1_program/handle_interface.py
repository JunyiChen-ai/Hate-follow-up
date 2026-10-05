"""Declared source-handle interface B; same original temporal interpreter."""
import copy
import json
import re
from program import CONSTANTS as ORIGINAL_CONSTANTS, canonical, span, local, execute

CACHE_VERSION='R1 source-handle interface B; sources2026-10-05'
CONSTANTS={**ORIGINAL_CONSTANTS,'handle_digits':8,'target_max_words':4,'field_tokens':24}
PLANNER_SYSTEM=('Select actual local sources and interpretation context for source-grounded evidence programs. '
    'Do not classify hate, supply confidence, invent sources or write a rationale. '
    'The decoder restricts your output to the supplied source handles and legal choices. '
    'Select one speech span and one frame whenever that window has actual candidates; '
    'context is optional and never establishes an occurrence in the local window. '
    'Select a temporal join only if these actual witnesses describe the same occurrence.')
PLANNER_END=('For each requested window choose speech, frame, zero to two distinct remote contexts '
    'and whether to join the two local witnesses. Source times are nominal frame coordinates '
    'or proportional character estimates, not verified frame/word timestamps.')
MODULE_SYSTEM=('Describe only the supplied source content. Do not classify hate or infer unsupported facts. '
    'The decoder restricts JSON syntax and source references. Choose UNKNOWN when uncertain. '
    'Support identifies the measured local source and does not certify your interpretation.')
SCOPE_QUESTION=('Read the local utterance using the interpretation-only contexts. Select speaker '
    '(speaker/quoted/reported/UNKNOWN), mode (direct/quoted/rejected/reported/UNKNOWN), '
    'and a target from the supplied exact local character spans or UNKNOWN. '
    'Contexts are not additional local occurrences. Return speaker,mode,target,support.')
ACTION_QUESTION=('Describe the visible actor, action and target in the supplied frame. '
    'Use at most twelve words per description and UNKNOWN where not visible. '
    'Return actor,action,target,support. The supplied frame is the only support.')


def sh(i):return 'S'+str(i).zfill(CONSTANTS['handle_digits'])
def fh(i):return 'F'+str(i).zfill(CONSTANTS['handle_digits'])


def catalog(source,window):
    speech={};contexts={}
    for i,s in enumerate(source['segments']):
        full=span(source,i,0,len(s['text']))
        if full['kind']=='UNKNOWN':continue
        clipped=local(source,full,window)
        if clipped['kind']!='UNKNOWN':speech[sh(i)]=clipped
        if s['end']<=window['start'] or s['start']>=window['end']:
            contexts[sh(i)]=full
    frames={fh(f['id']):dict(kind='frame',ref={'frame':f['id']},time=f['time'],
        time_source=f['time_source'],local=True,local_window=window['id'])
        for f in source['frames'] if window['start']<=f['time']<window['end']}
    return dict(speech=speech,frames=frames,contexts=contexts)


def join_possible(cat,speech,frame):
    if speech is None or frame is None:return False
    s=cat['speech'][speech];f=cat['frames'][frame]
    return s['start']<=f['time']<s['end']


def compile_selection(source,window,selection):
    assert set(selection)=={'window','speech','frame','contexts','join'}
    assert selection['window']==window['id'] and type(selection['join']) is bool
    c=catalog(source,window);s,f=selection['speech'],selection['frame'];ctx=selection['contexts']
    assert (s in c['speech']) if c['speech'] else s is None
    assert (f in c['frames']) if c['frames'] else f is None
    assert isinstance(ctx,list) and len(ctx)<=CONSTANTS['max_context'] and len(set(ctx))==len(ctx)
    assert all(h in c['contexts'] for h in ctx) and (s is not None or not ctx)
    assert not selection['join'] or join_possible(c,s,f)
    ops=[];emitted=[]
    if s is not None:
        r=c['speech'][s]['ref'];ops.extend([['span','s',r['segment'],r['start_char'],r['end_char']],['local','ls','s']])
        names=[]
        for i,h in enumerate(ctx):
            r=c['contexts'][h]['ref'];name='c'+str(i);n='lc'+str(i)
            ops.extend([['span',name,r['segment'],r['start_char'],r['end_char']],['context',n,name]])
            names.append(n)
        ops.append(['scope','scope','ls',names]);emitted.extend(['ls','scope'])
    if f is not None:
        ops.extend([['frame','f',c['frames'][f]['ref']['frame']],['local','lf','f'],['action','action','lf']])
        emitted.append('action')
    if selection['join']:
        ops.append(['join','join','ls','lf']);emitted.append('join')
    ops.append(['emit',emitted]);assert len(ops)<=CONSTANTS['max_ops']
    return dict(status='valid',ops=ops)


def target_choices(packet):
    obj=packet['sources'][0]['value'];text=obj['text'];r=obj['ref']
    words=list(re.finditer(r'\S+',text));choices=['"UNKNOWN"']
    for a in range(len(words)):
        for b in range(a,min(a+CONSTANTS['target_max_words'],len(words))):
            ref=dict(segment=r['segment'],start_char=r['start_char']+words[a].start(),
                end_char=r['start_char']+words[b].end())
            choices.append(canonical(ref))
    return choices


def planner_content(source,requested):
    content=[]
    for f in source['frames']:
        content.extend([dict(type='text',text=f'frame {f["id"]}, nominal t={f["time"]}s\n'),dict(type='image')])
    inventory=copy.deepcopy(source)
    for segment in inventory['segments']:segment['span_handle']=sh(segment['id'])
    for frame in inventory['frames']:frame['frame_handle']=fh(frame['id'])
    requested_material=[]
    for w in requested:
        candidates=catalog(source,w)
        # Full remote text already occurs once in the source inventory, never 8x.
        candidates['contexts']={h:{k:v for k,v in obj.items() if k!='text'}
            for h,obj in candidates['contexts'].items()}
        requested_material.append(dict(window=w,candidates=candidates))
    table=dict(source=inventory,requested_windows=requested_material)
    content.append(dict(type='text',text=canonical(table)+'\n'+PLANNER_END))
    return content


def module_content(kind,packet):
    material=copy.deepcopy(packet)
    if kind=='scope':material['target_options']=[json.loads(x) for x in target_choices(packet)]
    content=[dict(type='text',text=canonical(material)+'\n'+(SCOPE_QUESTION if kind=='scope' else ACTION_QUESTION))]
    if kind=='action':content.insert(0,dict(type='image'))
    return content


def write_plans(stream,source,requested):
    stream.force('[');selections=[]
    for i,w in enumerate(requested):
        if i:stream.force(',')
        c=catalog(source,w)
        stream.force('{"window":'+str(w['id'])+',"speech":')
        s=json.loads(stream.choose([canonical(h) for h in c['speech']] or ['null']))
        stream.force(',"frame":')
        f=json.loads(stream.choose([canonical(h) for h in c['frames']] or ['null']))
        stream.force(',"contexts":[');contexts=[]
        remote=list(c['contexts']) if s is not None else []
        first=stream.choose([']']+[canonical(h) for h in remote])
        if first!=']':
            contexts.append(json.loads(first))
            second=stream.choose([']']+[','+canonical(h)+']' for h in remote if h not in contexts])
            if second!=']':contexts.append(json.loads(second[1:-1]))
        stream.force(',"join":')
        join=json.loads(stream.choose(['false','true'] if join_possible(c,s,f) else ['false']))
        stream.force('}');selections.append(dict(window=w['id'],speech=s,frame=f,contexts=contexts,join=join))
    stream.force(']');return selections


def write_module(stream,kind,packet):
    if kind=='scope':
        stream.force('{"speaker":')
        stream.choose([canonical(x) for x in ('speaker','quoted','reported','UNKNOWN')])
        stream.force(',"mode":')
        stream.choose([canonical(x) for x in ('direct','quoted','rejected','reported','UNKNOWN')])
        stream.force(',"target":');stream.choose(target_choices(packet))
    else:
        stream.force('{"actor":"');stream.description()
        stream.force(',"action":"');stream.description()
        stream.force(',"target":"');stream.description()
    stream.force(',"support":'+canonical([packet['sources'][0]['id']])+'}')
