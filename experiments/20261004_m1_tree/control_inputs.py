"""Predeclared R2 input controls; no labels, scores, models or evaluator."""
import copy
import numpy as np
from tree import CONTEXT_HEADER,TREE_HEADER,record,tree_text,window_packet

ARMS=('main','flat','wrong_links','no_depth','no_added_pixels','temporal','temporal_fresh_priority')


def fresh_priority(temporal,scores,logprobs):
    """Hold topology fixed and change only the newly measured root priorities."""
    tree=copy.deepcopy(temporal)
    roots=[n for n in tree['nodes'] if n['parent'] is None]
    assert len(roots)==len(scores)==len(logprobs)
    assert all(type(s) is int and s in (1,2,3) for s in scores)
    by_root={n['id']:(s,lp) for n,s,lp in zip(roots,scores,logprobs)}
    by_id={n['id']:n for n in tree['nodes']}
    for n in tree['nodes']:
        root=n
        while root['parent'] is not None:root=by_id[root['parent']]
        n['root_relevance']=by_root[root['id']][0]
        if n['parent'] is None:n['relevance_logprobs']=copy.deepcopy(by_root[n['id']][1])
    tree['priority']='fresh temporal roots; matched topology remains fixed'
    return tree


def arm_inputs(arm,main,temporal,fresh,features,entries,windows):
    """Exact shared factual turn and local packets for one declared arm."""
    assert arm in ARMS
    tree=temporal if arm=='temporal' else fresh if arm=='temporal_fresh_priority' else main
    if arm=='no_depth':tree={**main,'nodes':[n for n in main['nodes'] if n['parent'] is None]}
    if arm=='flat':
        nodes=sorted(main['nodes'],key=lambda n:(n['time'],n['id']))
        observation=TREE_HEADER+(''.join(f'[t={n["time"]:.3f}s]\n{n["caption"]}\n' for n in nodes) if nodes else '(none)\n')
    else:observation=tree_text(tree)
    packets=[packet_control(arm,main,features,entries,a,b) if arm in ('flat','wrong_links','no_depth','no_added_pixels')
             else window_packet(tree,features,entries,a,b) for a,b in windows]
    return dict(observation=observation,packets=packets)


def temporal_tree(main,features,entries,caption):
    """Time-ordered membership with exactly the main topology/member counts."""
    old={n['id']:n for n in main['nodes']};nodes=[];captions={};events=[];reused=[]
    def build(source,members):
        assert len(members)==len(source['members']) and len(members)>0
        center=features[members].mean(0,dtype=np.float32);rep=members[len(members)//2]
        key=str(rep)
        if key not in captions:
            if key in main['captions']:captions[key]=copy.deepcopy(main['captions'][key]);reused.append(rep)
            else:captions[key]=caption(rep);events.append(rep)
        n=dict(id=source['id'],parent=source['parent'],depth=source['depth'],
            root_relevance=source['root_relevance'],members=members,center=center.tolist(),
            representative=rep,time=entries[rep]['time'],caption=captions[key]['text'],
            caption_tokens=captions[key]['tokens'])
        nodes.append(n);cursor=0
        children=sorted((c for c in old.values() if c['parent']==n['id']),key=lambda c:c['id'])
        for child in children:
            count=len(child['members']);build(child,members[cursor:cursor+count]);cursor+=count
        if children:assert cursor==len(members)
    roots=sorted((n for n in old.values() if n['parent'] is None),key=lambda n:n['id'])
    cursor=0
    for root in roots:
        count=len(root['members']);build(root,list(range(cursor,cursor+count)));cursor+=count
    assert cursor==len(entries) and len(nodes)==len(main['nodes'])
    return dict(nodes=nodes,captions=captions,caption_events=events,reused_main_captions=reused,
        rounds=[],matched_main_rounds=copy.deepcopy(main['rounds']),
        membership='time ordered; exact main node counts/topology')


def packet_control(arm,main,features,entries,start,end):
    """Keep true coordinates while changing one declared representation."""
    packet=window_packet(main,features,entries,start,end)
    nodes={n['id']:n for n in main['nodes']}
    if arm=='flat':
        ancestors=sorted((nodes[i] for i in packet['ancestor_ids']),key=lambda n:(n['time'],n['id']))
        packet['context']=CONTEXT_HEADER+(''.join(f'[t={n["time"]:.3f}s]\n{n["caption"]}\n' for n in ancestors) if ancestors else '(none)\n')
    elif arm=='no_added_pixels':packet['pool_members']=[]
    elif arm=='no_depth':
        roots={**main,'nodes':[n for n in main['nodes'] if n['parent'] is None]}
        packet=window_packet(roots,features,entries,start,end)
    elif arm=='wrong_links':
        parents={n['parent'] for n in main['nodes'] if n['parent'] is not None}
        leaves=sorted(n['id'] for n in main['nodes'] if n['id'] not in parents)
        shifted={i:leaves[(k+len(leaves)//2)%len(leaves)] for k,i in enumerate(leaves)}
        ancestors=[];seen=set()
        for leaf in packet['leaf_ids']:
            chain=[];parent=nodes[shifted[leaf]]['parent']
            while parent is not None:
                chain.append(nodes[parent]);parent=nodes[parent]['parent']
            for n in reversed(chain):
                if n['id'] not in seen:ancestors.append(n);seen.add(n['id'])
        packet['ancestor_ids']=[n['id'] for n in ancestors]
        packet['context']=CONTEXT_HEADER+(''.join(record(n) for n in ancestors) if ancestors else '(none)\n')
    else:raise ValueError(arm)
    assert all(start<=entries[i]['time']<end for i in packet['pool_members'])
    return packet
