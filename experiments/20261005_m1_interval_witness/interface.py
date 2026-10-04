"""Literal source ownership and executable interval composition; no scores/GT."""
from pathlib import Path
import json
import re

SPEC=json.loads((Path(__file__).parent/'spec.json').read_text())
VERSION=SPEC['version'];CONSTANTS=SPEC['constants']


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def model_visible(value):
    """Keep real source paths for audit/loading, never show internal names to model."""
    if isinstance(value,dict):return {k:model_visible(v) for k,v in value.items() if k!='path'}
    if isinstance(value,list):return [model_visible(v) for v in value]
    return value


def catalog(window):
    body=window['body'];words=list(re.finditer(r'\S+',body));prefix=f'w{window["i"]:04d}:';sources={}
    for frame in window['frames']:
        handle=prefix+frame['id'];sources[handle]=dict(handle=handle,kind='visual',owner=window['i'],
            start=window['start'],end=window['end'],frame=frame)
    for a in range(len(words)):
        for b in range(a+1,min(a+CONSTANTS['quote_words'],len(words))+1):
            start,end=words[a].start(),words[b-1].end();quote=body[start:end]
            if body.find(quote)!=start or body.find(quote,start+1)!=-1:continue
            handle=prefix+f't{a:04d}_{b:04d}'
            sources[handle]=dict(handle=handle,kind='speech',owner=window['i'],start=window['start'],end=window['end'],
                char_start=start,char_end=end,quote=quote)
    return sources


def source_table(window):
    words=list(re.finditer(r'\S+',window['body']))
    return dict(window=window['i'],start=window['start'],end=window['end'],body=window['body'],
        words=[dict(i=i,start=w.start(),end=w.end()) for i,w in enumerate(words)],frames=model_visible(window['frames']),
        handle_rule=f'w{window["i"]:04d}:p<frame index> or w{window["i"]:04d}:t<start word:04d>_<exclusive end word:04d>; unique contiguous 1-16-word literal speech only')


def coverage(windows):
    return dict(visual=all(bool(w['frames']) for w in windows),speech=all(bool(w['body'].strip()) for w in windows))


def write_record(stream,sources):
    stream.force('{"status":');status=json.loads(stream.choose([canonical(s) for s in ('present','absent','UNKNOWN') if s!='present' or sources]))
    stream.force(',"description":"');stream.description();description=stream.events[-1]['text']
    stream.force(',"witnesses":[');handles=list(sources);witnesses=[]
    choices=[canonical(h) for h in handles]+([] if status=='present' else [']'])
    selected=stream.choose(choices)
    if selected!=']':
        witnesses.append(json.loads(selected));remaining=[h for h in handles if h!=witnesses[0]]
        separator=stream.choose([']']+([','] if remaining else []))
        if separator==',':
            witnesses.append(json.loads(stream.choose([canonical(h) for h in remaining])));stream.force(']')
    stream.force('}')
    return dict(status=status,description=description,witnesses=witnesses)


def compile_record(g,sources,cov):
    if g['truncated']:
        assert g['selection'] is None
        return dict(status='UNKNOWN',description='',witnesses=[],coverage=cov,error='token_cap')
    raw=g['selection'];assert raw==json.loads(g['text']) and set(raw)=={'status','description','witnesses'}
    assert raw['status'] in ('present','absent','UNKNOWN') and type(raw['description']) is str
    assert len(raw['description'].split())<=CONSTANTS['description_words']
    assert len(raw['witnesses'])<=2 and len(set(raw['witnesses']))==len(raw['witnesses'])
    assert all(h in sources for h in raw['witnesses'])
    assert raw['status']!='present' or raw['witnesses']
    normalized='UNKNOWN' if raw['status']=='absent' and not all(cov.values()) else raw['status']
    return dict(status=normalized,description=raw['description'],witnesses=raw['witnesses'],coverage=cov,
        error=None,generated_status=raw['status'])


def partition(n):
    assert type(n) is int and n>0
    nodes=[]
    def visit(lo,hi,depth):
        node=dict(id=f'n{lo:04d}_{hi:04d}',lo=lo,hi=hi,depth=depth)
        if hi-lo>1:
            mid=(lo+hi)//2;node['children']=[visit(lo,mid,depth+1),visit(mid,hi,depth+1)]
        nodes.append(node);return node['id']
    visit(0,n,0);return nodes


def compose(children):
    cov={kind:all(c['coverage'][kind] for c in children) for kind in ('visual','speech')}
    status='present' if any(c['status']=='present' for c in children) else (
        'absent' if all(c['status']=='absent' for c in children) and all(cov.values()) else 'UNKNOWN')
    return status,cov


def tree_records(windows,leaves,proposals):
    assert len(windows)==len(leaves);records={};nodes=partition(len(windows))
    for node in nodes:
        lo,hi=node['lo'],node['hi'];bounds=dict(start=windows[lo]['start'],end=windows[hi-1]['end'])
        if hi-lo==1:r=dict(**node,**bounds,**leaves[lo])
        else:
            p=proposals[node['id']]['record'];status,cov=compose([records[c] for c in node['children']])
            r=dict(**node,**bounds,status=status,coverage=cov,description=p['description'],witnesses=p['witnesses'],
                proposed_status=p['status'],proposal_error=p['error'])
            r['disagreement']=bool(conflicting_leaves(r,leaves));r['resolved']=not r['disagreement']
        records[node['id']]=r
    return records


def conflicting_leaves(parent,leaves):
    lo,hi=parent['lo'],parent['hi'];status=parent['proposed_status'];targets=[]
    if status=='present':
        for handle in parent['witnesses']:
            owner=int(handle.split(':')[0][1:]);assert lo<=owner<hi
            if leaves[owner]['status']!='present':targets.append(owner)
        if not targets and parent['status']!='present':targets=[i for i in range(lo,hi) if leaves[i]['status']=='UNKNOWN']
    elif status=='absent' and parent['status']=='present':
        targets=[i for i in range(lo,hi) if leaves[i]['status']=='present']
    return sorted(set(targets))


def repair_plan(windows,leaves,proposals):
    records=tree_records(windows,leaves,proposals);triggers=[]
    for r in records.values():
        if r['hi']-r['lo']==1:continue
        targets=conflicting_leaves(r,leaves)
        if targets:triggers.append((r,min(targets)))
    triggers.sort(key=lambda x:(-x[0]['depth'],x[0]['lo'],x[0]['hi'],x[1]));plan=[];seen=set()
    for parent,i in triggers:
        if i in seen:continue
        seen.add(i);plan.append(dict(leaf=i,parent=parent['id'],context=repair_context(parent,windows)))
    return plan


def repair_context(parent,windows):
    allsources={h:r for w in windows[parent['lo']:parent['hi']] for h,r in catalog(w).items()}
    return dict(parent_range=dict(lo=parent['lo'],hi=parent['hi'],start=parent['start'],end=parent['end']),
        parent_description=parent['description'],proposed_witness_literal_source_records=[allsources[h] for h in parent['witnesses']])


def final_tree(windows,leaves,proposals):
    records=tree_records(windows,leaves,proposals)
    return sorted(records.values(),key=lambda r:(r['hi']-r['lo'],r['lo'],r['hi']))


def resolve_handle(windows,handle):
    prefix,coord=handle.split(':');owner=int(prefix[1:]);w=windows[owner]
    assert prefix==f'w{owner:04d}' and w['i']==owner
    if coord.startswith('p'):
        frames=[f for f in w['frames'] if f['id']==coord];assert len(frames)==1
        return dict(handle=handle,kind='visual',owner=owner,start=w['start'],end=w['end'],frame=frames[0])
    a,b=map(int,coord[1:].split('_'));words=list(re.finditer(r'\S+',w['body']))
    assert coord==f't{a:04d}_{b:04d}' and 0<=a<b<=len(words) and b-a<=16
    start,end=words[a].start(),words[b-1].end();quote=w['body'][start:end];assert w['body'].find(quote)==start and w['body'].find(quote,start+1)==-1
    return dict(handle=handle,kind='speech',owner=owner,start=w['start'],end=w['end'],char_start=start,char_end=end,quote=quote)


def source_path(metadata,i,kind):
    path=[r for r in metadata['final_tree'] if r['lo']<=i<r['hi']]
    path.sort(key=lambda r:(r['hi']-r['lo'],r['lo'],r['hi']));literal=[];seen=set()
    for r in path:
        for handle in r['witnesses']:
            if handle not in seen:
                seen.add(handle);s=resolve_handle(metadata['windows'],handle)
                literal.append(dict(**s,role='local' if s['owner']==i else 'interpretation_context'))
    return dict(current_window=i,modality=kind,local_final_record=path[0],ancestor_records=path[1:],literal_witness_sources=literal)
