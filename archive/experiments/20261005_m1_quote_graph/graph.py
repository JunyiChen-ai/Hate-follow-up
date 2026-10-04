"""Exact-source discourse graph and deterministic context packets."""
import heapq
import json
import re

VERSION = 'R1 source-bound quotation graph, sources2026-10-05'
CONSTANTS = dict(window_seconds=8, chunk_windows=8, chunk_stride=6,
    max_generation_tokens=2048, max_nodes=32, max_edges=48, max_hops=2, max_context=8, seed=0)
KINDS = ('mention','quote','attribution','rejection','endorsement','negation','correction')
TYPES = ('same_referent','attributed_to','refers_to','quotes','rejects','endorses','negates','corrects')
SYSTEM = ('Extract discourse relations from supplied transcript spans. Return only the requested JSON. '
          'Do not judge policy violations or invent speaker identities.')
INSTRUCTION = ('Find exact spans for mentions, quoted or reported speech, attribution cues, rejection, '
    'endorsement, negation and correction. Mention links indicate the same referent; attributed_to links '
    'a quoted span to its speaker mention. Quotes/rejects/endorses/negates/corrects link their actual cue '
    'or utterance to the affected span. Use UNKNOWN by omitting unsupported edges. Every span must use '
    'exact source-body offsets and every endpoint must be supplied or defined here. Never add free '
    'text, labels, confidence, summaries or conclusions.')
SCHEMA = '{"nodes":[{"id":"x0","window":0,"start":0,"end":3,"kind":"mention"}],"edges":[{"from":"x0","to":"u0000","type":"refers_to"}]}'
HEAD = ('Consider only window {i} of {n}, from {a:.1f}s to {b:.1f}s of this video. '
    'Source-linked spans below are interpretation context. A span from another window is not an '
    'occurrence in the current window.\nRelations:\n')
TAIL = '\n\nIs THIS window one of the segments where speech that violates the above rules occurs?\n\nAnswer "Yes" or "No".'


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def uid(i):return f'u{i:04d}'
def sid(node):return f"s{node['window']:04d}:{node['start']}:{node['end']}:{node['kind']}"


def chunk_indices(n):
    result=[]
    for start in range(0,n,CONSTANTS['chunk_stride']):
        result.append(list(range(start,min(n,start+CONSTANTS['chunk_windows']))))
        if start+CONSTANTS['chunk_windows']>=n:break
    return result


def previous_subset(graph, indices):
    nodes=[n for n in graph['nodes'] if n['window'] in indices]
    ids={n['id'] for n in nodes}|{uid(i) for i in indices}
    return dict(nodes=nodes,edges=[e for e in graph['edges'] if e['from'] in ids and e['to'] in ids])


def prompt(windows, indices, previous):
    source=[dict(id=uid(i),window=i,start=windows[i]['start'],end=windows[i]['end'],
        body=windows[i]['body'],characters=len(windows[i]['body'])) for i in indices]
    return INSTRUCTION+'\nNode kinds: '+','.join(KINDS)+'\nEdge types: '+','.join(TYPES)+(
        f"\nAt most {CONSTANTS['max_nodes']} nodes and {CONSTANTS['max_edges']} edges. Windows are zero-based.\n")+(
        'Schema: '+SCHEMA+'\nSources: '+canonical(source)+'\nPrevious accepted overlapping graph: '+canonical(previous))


def pairs(items):
    result={}
    for k,v in items:
        if k in result:raise ValueError('duplicate JSON key')
        result[k]=v
    return result


def parse_chunk(text, truncated, windows, indices, previous):
    """Any invalid item invalidates the whole generation; never salvage a subset."""
    try:
        if truncated:raise ValueError('truncated')
        raw=json.loads(text,object_pairs_hook=pairs)
        if type(raw) is not dict or set(raw)!= {'nodes','edges'}:raise ValueError('root schema')
        if type(raw['nodes']) is not list or type(raw['edges']) is not list:raise ValueError('lists')
        if len(raw['nodes'])>CONSTANTS['max_nodes'] or len(raw['edges'])>CONSTANTS['max_edges']:raise ValueError('limits')
        known={n['id']:n for n in previous['nodes']}
        known.update({uid(i):dict(id=uid(i),kind='utterance',window=i) for i in indices})
        remap={k:k for k in known};nodes=[]
        for r in raw['nodes']:
            if type(r) is not dict or set(r)!= {'id','window','start','end','kind'}:raise ValueError('node schema')
            if type(r['id']) is not str or not re.fullmatch(r'x[0-9]+',r['id']) or r['id'] in remap:raise ValueError('node id')
            if any(type(r[k]) is not int for k in ('window','start','end')):raise ValueError('integer coordinates')
            if r['window'] not in indices or r['kind'] not in KINDS:raise ValueError('node source/type')
            if not 0<=r['start']<r['end']<=len(windows[r['window']]['body']):raise ValueError('span bounds')
            node={k:r[k] for k in ('window','start','end','kind')};node['id']=sid(node)
            remap[r['id']]=node['id'];known[node['id']]=node;nodes.append(node)
        edges=[]
        for e in raw['edges']:
            if type(e) is not dict or set(e)!= {'from','to','type'}:raise ValueError('edge schema')
            if e['type'] not in TYPES or e['from'] not in remap or e['to'] not in remap:raise ValueError('edge endpoint/type')
            a,b=known[remap[e['from']]],known[remap[e['to']]]
            if e['type']=='same_referent' and (a['kind'],b['kind'])!=('mention','mention'):raise ValueError('same referent types')
            if e['type']=='attributed_to' and (a['kind'],b['kind'])!=('quote','mention'):raise ValueError('attribution types')
            edges.append(dict(type=e['type'],**{'from':a['id'],'to':b['id']}))
        return dict(nodes=nodes,edges=edges),None
    except (ValueError,TypeError,KeyError,IndexError) as exc:
        return dict(nodes=[],edges=[]),str(exc)


def merge(graph, chunk):
    nodes={n['id']:n for n in graph['nodes']}
    for n in chunk['nodes']:nodes.setdefault(n['id'],n)
    edges={(e['from'],e['to'],e['type']):e for e in graph['edges']}
    for e in chunk['edges']:edges.setdefault((e['from'],e['to'],e['type']),e)
    return dict(nodes=sorted(nodes.values(),key=lambda n:(n['window'],n['start'],n['end'],n['kind'])),
        edges=[edges[k] for k in sorted(edges)])


def packet(graph,windows,index):
    nodes={n['id']:n for n in graph['nodes']}
    parent={k:k for k,n in nodes.items() if n['kind']=='mention'}
    def find(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    for e in graph['edges']:
        if e['type']=='same_referent':
            a,b=find(e['from']),find(e['to'])
            if a!=b:parent[max(a,b)]=min(a,b)
    # Zero-cost components implement the declared virtual mention vertices.
    members={}
    for k in parent:members.setdefault(find(k),[]).append(k)
    adjacency={k:[] for k in [*nodes,*[uid(i) for i in range(len(windows))]]}
    for n in nodes.values():adjacency[uid(n['window'])].append((n['id'],0,None))
    for group in members.values():
        first=min(group)
        for k in group:
            if k!=first:
                adjacency[k].append((first,0,None));adjacency[first].append((k,0,None))
    for e in graph['edges']:
        if e['type']=='same_referent':continue
        adjacency[e['from']].append((e['to'],1,e));adjacency[e['to']].append((e['from'],1,e))
    distance={};queue=[(0,uid(index))]+[(0,k) for k,n in nodes.items() if n['window']==index]
    heapq.heapify(queue);traversed=[]
    while queue:
        d,k=heapq.heappop(queue)
        if k in distance and distance[k]<=d:continue
        distance[k]=d
        for to,cost,e in adjacency[k]:
            if d+cost<=CONSTANTS['max_hops']:
                heapq.heappush(queue,(d+cost,to))
                if e is not None:traversed.append(e)
    candidates=sorted([n for k,n in nodes.items() if k in distance and n['window']!=index],
        key=lambda n:(distance[n['id']],abs(n['window']-index),n['window'],n['start'],n['end'],n['kind'],n['id']))
    selected=candidates[:CONSTANTS['max_context']]
    # Include the coreference links that ground the declared component collapse.
    reached={k for k in distance}
    used={(e['from'],e['to'],e['type']):e for e in traversed}
    for e in graph['edges']:
        if e['type']=='same_referent' and e['from'] in reached and e['to'] in reached:
            used[e['from'],e['to'],e['type']]=e
    return dict(window=index,selected=selected,omitted=len(candidates)-len(selected),
        distances={n['id']:distance[n['id']] for n in selected},relations=[used[k] for k in sorted(used)],
        components=[sorted(v) for _,v in sorted(members.items())])


def context_text(node,windows):
    w=windows[node['window']]
    return (f"[source={node['id']}; window={node['window']+1}; t={w['start']:.1f}-{w['end']:.1f}s; "
        f"chars={node['start']}:{node['end']}; kind={node['kind']}]\n"+
        w['body'][node['start']:node['end']]+'\n')


def compile_segments(windows,packet):
    w=windows[packet['window']]
    head=HEAD.format(i=w['i']+1,n=len(windows),a=w['start'],b=w['end'])
    head+=''.join(f"({e['from']})--{e['type']}-->({e['to']})\n" for e in packet['relations'])
    return head,[context_text(n,windows) for n in packet['selected']], 'Local speech:\n'+w['body']+'\n',TAIL
