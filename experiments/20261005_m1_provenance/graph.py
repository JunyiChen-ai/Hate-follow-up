"""Source-bound entity occurrences, typed discourse links and executable retrieval."""
import json
from collections import deque

VERSION='R1 source-bound temporal entity-discourse graph; sources2026-10-05'
CONSTANTS=dict(window_seconds=8,local_images=2,ledger_tokens=512,link_tokens=2048,
    link_windows=8,max_entities=4,max_actions=2,max_quotes=2,description_words=24,
    max_edges=64,semantic_hops=2,remote_each_side=2,seed=0)
LEDGER_SYSTEM=('Describe observable people, actions and discourse sources. Return only the requested JSON. '
    'Do not judge policy violations or invent identities.')
LEDGER_SCHEMA={'entities':[{'id':'e0','description':'person at podium','frame':'p0','quote':''}],
    'actions':[{'id':'a0','description':'speaking at podium','actor':'e0','target':'UNKNOWN','frame':'p0'}],
    'utterance':{'speaker':'UNKNOWN'},'quotations':[{'id':'q0','quote':'exact local substring','owner':'UNKNOWN'}]}
LEDGER_INSTRUCTION=('Record only observable local entities and actions. Use frame IDs only from the local image list. '
    'Copy textual mentions or quotations exactly from u. Infer an utterance speaker or quote owner only when supported; '
    'otherwise use UNKNOWN. Do not add confidence, hate labels, decisions or summaries. '
    'At most4 entities e0-e3,2 actions a0-a1,2 quotations q0-q1; descriptions at most24 whitespace words. '
    'An entity cites a local frame and empty quote, or null frame and a nonempty exact uniquely occurring substring of u. '
    'Actions cite local frames and local entity IDs or UNKNOWN. Quotation owners and utterance speakers are local entity IDs '
    'or UNKNOWN. Empty arrays and UNKNOWN are valid. Return exactly the schema keys without extra fields.')
LINK_SYSTEM=('Link observable entities and discourse sources across video time. Return only the requested JSON. '
    'Do not judge policy violations or invent identities.')
LINK_INSTRUCTION=('Link source observations across time. same_entity means the same observed or named referent, '
    'not merely similar descriptions. quotes, responds_to, retracts and quote_owner preserve their actual directed roles. '
    'speaker_change identifies a change of speaker between source utterances. Use only supplied node IDs; omit unsupported '
    'relations. Return only edges, without descriptions, confidence, labels or decisions. '
    'quotes: quoting utterance/quote -> cited utterance/quote. responds_to: response -> utterance/quote being addressed. '
    'retracts: retraction -> an earlier utterance/quote it retracts. speaker_change: earlier utterance -> later utterance. '
    'quote_owner: utterance/quote -> attributed author entity. same_entity: entity occurrence <-> entity occurrence. '
    'Distinct existing endpoints, at least one in the anchor windows; at most64 edges. Schema: '
    '{"edges":[{"from":"w0000/e0","to":"w0008/e1","type":"same_entity"}]}')
PACKET_HEADER=('Graph-linked sources below are interpretation context. A source from another window does not establish '
    'an occurrence inside the current window. Source-bound descriptions and links are model observations, not '
    'independently certified facts. Use current-window pixels and speech to identify the local act.\n')


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def exact_json(text):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate JSON key')
            result[k]=v
        return result
    return json.loads(text,object_pairs_hook=pairs)


def fields(obj,keys):
    if type(obj) is not dict or set(obj)!=set(keys):raise ValueError('schema fields')


def unique_span(body,quote):
    if type(quote) is not str or not quote:raise ValueError('empty/non-string source quote')
    first=body.find(quote)
    if first<0 or body.find(quote,first+1)>=0:raise ValueError('quote not unique exact source substring')
    return first,first+len(quote)


def utterance_node(w):
    if not w['body'].strip():return []
    return [dict(id=f'w{w["i"]:04d}/u',window=w['i'],kind='utterance',start=0,end=len(w['body']),text=w['body'])]


def parse_ledger(g,w):
    """Any invalid field rejects the entire generated ledger, never its source u."""
    base=utterance_node(w)
    try:
        if g['truncated']:raise ValueError('whole-call token cap')
        value=exact_json(g['text']);fields(value,('entities','actions','utterance','quotations'))
        nodes=[];edges=[];prefix=f'w{w["i"]:04d}/';frames={f['id'] for f in w['frames']}
        lists=(('entities',4),('actions',2),('quotations',2))
        for key,limit in lists:
            if type(value[key]) is not list or len(value[key])>limit:raise ValueError('ledger list cap/type')
        entities={}
        for e in value['entities']:
            fields(e,('id','description','frame','quote'))
            if e['id'] not in [f'e{i}' for i in range(4)] or e['id'] in entities:raise ValueError('entity ID')
            if type(e['description']) is not str or not 0<len(e['description'].split())<=24:raise ValueError('entity description')
            node=dict(id=prefix+e['id'],window=w['i'],kind='entity',description=e['description'])
            if e['frame'] is None:
                a,b=unique_span(w['body'],e['quote']);node.update(start=a,end=b,text=e['quote'],frame=None)
            elif e['frame'] in frames and e['quote']=='':node.update(frame=e['frame'])
            else:raise ValueError('entity witness')
            entities[e['id']]=node['id'];nodes.append(node)
        action_ids=set()
        for a in value['actions']:
            fields(a,('id','description','actor','target','frame'))
            if a['id'] not in ('a0','a1') or a['id'] in action_ids:raise ValueError('action ID')
            if type(a['description']) is not str or not 0<len(a['description'].split())<=24:raise ValueError('action description')
            if a['frame'] not in frames:raise ValueError('action witness')
            node=dict(id=prefix+a['id'],window=w['i'],kind='action',description=a['description'],frame=a['frame'])
            for role in ('actor','target'):
                if a[role]!='UNKNOWN':
                    if a[role] not in entities:raise ValueError('action entity binding')
                    edges.append(dict(**{'from':node['id'],'to':entities[a[role]]},type=role))
            nodes.append(node);action_ids.add(a['id'])
        fields(value['utterance'],('speaker',));speaker=value['utterance']['speaker']
        if speaker!='UNKNOWN':
            if speaker not in entities or not base:raise ValueError('utterance speaker')
            edges.append(dict(**{'from':base[0]['id'],'to':entities[speaker]},type='speaker'))
        quote_ids=set()
        for q in value['quotations']:
            fields(q,('id','quote','owner'))
            if q['id'] not in ('q0','q1') or q['id'] in quote_ids:raise ValueError('quotation ID')
            a,b=unique_span(w['body'],q['quote']);node=dict(id=prefix+q['id'],window=w['i'],kind='quote',start=a,end=b,text=q['quote'])
            if q['owner']!='UNKNOWN':
                if q['owner'] not in entities:raise ValueError('quotation owner')
                edges.append(dict(**{'from':node['id'],'to':entities[q['owner']]},type='quote_owner'))
            edges.append(dict(**{'from':prefix+'u','to':node['id']},type='quotes'))
            nodes.append(node);quote_ids.add(q['id'])
        return dict(nodes=base+nodes,edges=edges,error=None)
    except (ValueError,TypeError,KeyError) as error:return dict(nodes=base,edges=[],error=str(error))


def parse_links(g,nodes,anchors):
    try:
        if g['truncated']:raise ValueError('whole-call token cap')
        value=exact_json(g['text']);fields(value,('edges',))
        if type(value['edges']) is not list or len(value['edges'])>64:raise ValueError('link cap/type')
        ids={n['id']:n for n in nodes};result=[];seen=set();speech={'utterance','quote'}
        for e in value['edges']:
            fields(e,('from','to','type'))
            if e['from'] not in ids or e['to'] not in ids or e['from']==e['to']:raise ValueError('edge endpoints')
            a,b=ids[e['from']],ids[e['to']];kind=e['type'];pair=(a['kind'],b['kind'])
            if a['window'] not in anchors and b['window'] not in anchors:raise ValueError('anchor incidence')
            if kind=='same_entity':
                if pair!=('entity','entity'):raise ValueError('same_entity kinds')
                e={**e,'from':min(a['id'],b['id']),'to':max(a['id'],b['id'])}
            elif kind in ('quotes','responds_to','retracts'):
                if not (a['kind'] in speech and b['kind'] in speech):raise ValueError('discourse kinds')
                if kind=='retracts' and not b['window']<a['window']:raise ValueError('retraction direction')
            elif kind=='speaker_change':
                if pair!=('utterance','utterance') or not a['window']<b['window']:raise ValueError('speaker change direction')
            elif kind=='quote_owner':
                if a['kind'] not in speech or b['kind']!='entity':raise ValueError('owner kinds')
            else:raise ValueError('edge type')
            key=e['from'],e['to'],e['type']
            if key in seen:continue
            seen.add(key);result.append(e)
        return dict(edges=result,error=None)
    except (ValueError,TypeError,KeyError) as error:return dict(edges=[],error=str(error))


def compile_graph(ledgers,link_results):
    nodes=[n for ledger in ledgers for n in ledger['nodes']];edges=[];seen=set()
    assert len({n['id'] for n in nodes})==len(nodes)
    for edge in [e for l in ledgers for e in l['edges']]+[e for r in link_results for e in r['edges']]:
        key=edge['from'],edge['to'],edge['type']
        if key not in seen:seen.add(key);edges.append(edge)
    nodes=sorted(nodes,key=lambda n:n['id']);edges=sorted(edges,key=lambda e:(e['from'],e['to'],e['type']))
    parent={n['id']:n['id'] for n in nodes if n['kind']=='entity'}
    def find(i):
        if parent[i]!=i:parent[i]=find(parent[i])
        return parent[i]
    for edge in edges:
        if edge['type']=='same_entity':
            a,b=find(edge['from']),find(edge['to'])
            if a!=b:parent[max(a,b)]=min(a,b)
    components={i:find(i) for i in sorted(parent)}
    return dict(nodes=nodes,edges=edges,entity_components=components)


def retrieve(graph,w,windows):
    ids={n['id']:n for n in graph['nodes']};adj={i:[] for i in ids}
    for edge in graph['edges']:
        weight=0 if edge['type']=='same_entity' else 1
        adj[edge['from']].append((edge['to'],weight,edge));adj[edge['to']].append((edge['from'],weight,edge))
    seeds=sorted(i for i,n in ids.items() if n['window']==w);distance={i:0 for i in seeds};paths={i:[] for i in seeds};queue=deque(seeds)
    while queue:
        current=queue.popleft()
        for target,weight,edge in sorted(adj[current],key=lambda x:(x[1],x[0],x[2]['type'])):
            d=distance[current]+weight
            if d<=2 and (target not in distance or d<distance[target]):
                distance[target]=d;paths[target]=paths[current]+[edge];queue.appendleft(target) if weight==0 else queue.append(target)
    reached={ids[i]['window'] for i in distance if ids[i]['window']!=w};selected=[]
    center=lambda i:(windows[i]['start']+windows[i]['end'])/2
    for side in (sorted([i for i in reached if i<w],key=lambda i:(abs(center(i)-center(w)),i)),
                 sorted([i for i in reached if i>w],key=lambda i:(abs(center(i)-center(w)),i))):selected.extend(side[:2])
    selected=sorted(selected);witnesses=[]
    for remote in selected:
        target=min((i for i in distance if ids[i]['window']==remote),key=lambda i:(distance[i],i))
        witnesses.append(dict(window=remote,node=target,hops=distance[target],path=paths[target]))
    return dict(window=w,selected_windows=selected,witnesses=witnesses)
