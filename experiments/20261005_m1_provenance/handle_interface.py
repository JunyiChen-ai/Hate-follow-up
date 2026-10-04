"""Declared B source handles; optional observations and legal current edge endpoints."""
import json
import re
from graph import CONSTANTS as ORIGINAL_CONSTANTS, canonical, parse_ledger, parse_links

VERSION='R1 structured entity-discourse source handles B; sources2026-10-05'
CONSTANTS={**ORIGINAL_CONSTANTS,'field_tokens':64,'entity_words':4,'quote_words':16}


def catalog(window):
    body=window['body'];words=list(re.finditer(r'\S+',body));result={}
    for i in range(len(words)):
        for j in range(i,min(i+16,len(words))):
            a,b=words[i].start(),words[j].end();text=body[a:b]
            if body.find(text)!=a or body.find(text,a+1)>=0:continue
            result[f't{i:04d}_{j+1:04d}']=dict(start=a,end=b,entity=j-i+1<=4)
    return result


def ledger_content(overview,window):
    from inputs import ledger_content as original_content
    content,paths=original_content(overview,window)
    content.append(dict(type='text',text='Structured source interface: entities use id,description,support; '
        'support is a LOCAL frame ID or a catalog exact span handle. Actions retain id,description,actor,target,frame. '
        'utterance has speaker. Quotations use id,span,owner with span a catalog exact source handle. '
        'Empty arrays and UNKNOWN are allowed; do not invent any observation. The decoder supplies literal syntax '
        'and source IDs, but your factual choices/descriptions must be supported by the actual sources. '
        'Text span handles t<start-word:04d>_<exclusive-end-word:04d> refer to the exact u above. '
        'Each handle spans contiguous whitespace words,1-4 for entities or1-16 for quotations, '
        'and the literal substring must occur uniquely. All legal choices remain available to the decoder. '
        'Word character boundaries, indexed from0:\n'+canonical([[m.start(),m.end()] for m in re.finditer(r'\S+',window['body'])])))
    return content,paths


def link_content(graph,anchors):
    from inputs import link_content as original_content
    content=original_content(graph,anchors)
    content.append(dict(type='text',text='Structured source interface: return choices. Each choice has type '
        '(STOP or a supported relation), then from/to existing source nodes. STOP ends the list. '
        'Do not select a relation merely because its endpoints are legal. Unsupported relations must be omitted.'))
    return content


def write_ledger(stream,window):
    spans=catalog(window);frames=[f['id'] for f in window['frames']];entities=[];actions=[];quotes=[]
    stream.force('{"entities":[');closed=False
    supports=frames+[h for h,p in spans.items() if p['entity']]
    for i in range(4):
        if not supports:stream.force(']');closed=True;break
        # Commas must be conditional on an additional actual observation.
        opening=('{"id":' if i==0 else ',{"id":')+canonical(f'e{i}')+','
        if stream.choose([']',opening])==']':closed=True;break
        stream.force('"description":"');stream.description()
        stream.force(',"support":');support=json.loads(stream.choose([canonical(s) for s in supports]))
        stream.force('}');entities.append(f'e{i}')
    if not closed:stream.force(']')
    stream.force(',"actions":[');closed=False
    for i in range(2):
        if not frames:stream.force(']');closed=True;break
        opening=('{"id":' if i==0 else ',{"id":')+canonical(f'a{i}')+','
        if stream.choose([']',opening])==']':closed=True;break
        stream.force('"description":"');stream.description()
        for role in ('actor','target'):
            stream.force(',"'+role+'":');stream.choose([canonical(s) for s in ['UNKNOWN']+entities])
        stream.force(',"frame":');stream.choose([canonical(s) for s in frames]);stream.force('}')
    if not closed:stream.force(']')
    stream.force(',"utterance":{"speaker":')
    stream.choose([canonical(s) for s in ['UNKNOWN']+(entities if window['body'].strip() else [])])
    stream.force('},"quotations":[');closed=False
    for i in range(2):
        if not spans:stream.force(']');closed=True;break
        opening=('{"id":' if i==0 else ',{"id":')+canonical(f'q{i}')+','
        if stream.choose([']',opening])==']':closed=True;break
        stream.force('"span":');stream.choose([canonical(s) for s in spans])
        stream.force(',"owner":');stream.choose([canonical(s) for s in ['UNKNOWN']+entities]);stream.force('}')
    if not closed:stream.force(']')
    stream.force('}')
    return json.loads(stream.j.tok.decode(stream.tokens,skip_special_tokens=True))


def compiled_ledger(g,window):
    if g['truncated']:return parse_ledger(dict(truncated=True,text=''),window)
    value=g['selection'];spans=catalog(window);frames={f['id'] for f in window['frames']}
    original=dict(entities=[],actions=value['actions'],utterance=value['utterance'],quotations=[])
    for entity in value['entities']:
        s=entity['support'];e=dict(id=entity['id'],description=entity['description'])
        if s in frames:e.update(frame=s,quote='')
        else:
            p=spans[s];assert p['entity'];e.update(frame=None,quote=window['body'][p['start']:p['end']])
        original['entities'].append(e)
    for quote in value['quotations']:
        p=spans[quote['span']]
        original['quotations'].append(dict(id=quote['id'],quote=window['body'][p['start']:p['end']],owner=quote['owner']))
    return parse_ledger(dict(text=canonical(original),truncated=False),window)


def legal_pairs(nodes,anchors,used):
    result={};speech={'utterance','quote'}
    for a in nodes:
        for b in nodes:
            if a['id']==b['id'] or not (a['window'] in anchors or b['window'] in anchors):continue
            kinds=[]
            if a['kind']==b['kind']=='entity' and a['id']<b['id']:kinds.append('same_entity')
            if a['kind'] in speech and b['kind'] in speech:
                kinds+=['quotes','responds_to']
                if b['window']<a['window']:kinds.append('retracts')
            if a['kind']==b['kind']=='utterance' and a['window']<b['window']:kinds.append('speaker_change')
            if a['kind'] in speech and b['kind']=='entity':kinds.append('quote_owner')
            for kind in kinds:
                edge=(a['id'],b['id'],kind)
                if edge not in used:result.setdefault(kind,{}).setdefault(a['id'],[]).append(b['id'])
    return result


def write_links(stream,graph,anchors):
    stream.force('{"choices":[');used=set();result=[]
    for i in range(64):
        if i:stream.force(',')
        pairs=legal_pairs(graph['nodes'],anchors,used)
        stream.force('{"type":');kind=json.loads(stream.choose([canonical(s) for s in ['STOP']+sorted(pairs)]))
        if kind=='STOP':stream.force('}');break
        stream.force(',"from":');a=json.loads(stream.choose([canonical(s) for s in sorted(pairs[kind])]))
        stream.force(',"to":');b=json.loads(stream.choose([canonical(s) for s in sorted(pairs[kind][a])]))
        stream.force('}');used.add((a,b,kind));result.append(dict(type=kind,**{'from':a,'to':b}))
    stream.force(']}');return dict(edges=result)


def compiled_links(g,nodes,anchors):
    return parse_links(dict(text=canonical(g['selection']) if not g['truncated'] else '',truncated=g['truncated']),nodes,anchors)
