"""Predeclared source-ID interventions; no labels, margins or synthetic donors."""
from collections import Counter

ARMS=('R0','L0','C0','L1','C1','L2','D0')


def token_length(block):return int(block['shape'][3])


def remote_candidates(blocks,local):
    excluded=set(local)
    return [i for i in range(len(blocks)) if i not in excluded]


def time_order(blocks,local,bounds):
    start,end=bounds
    def distance(index):
        time=blocks[index]['actual_time']
        assert time<start or time>=end,'remote candidate is inside LOCAL occurrence window'
        return start-time if time<start else time-end
    return sorted(remote_candidates(blocks,local),key=lambda i:(distance(i),blocks[i]['source_index']))


def matched_preference(blocks,original,preference):
    required=Counter(token_length(blocks[i]) for i in original);result=[]
    for index in preference:
        length=token_length(blocks[index])
        if required[length]>0:
            result.append(index);required[length]-=1
    assert not any(required.values()),'original IDs must remain eligible for exact budget matching'
    assert len(result)==len(set(result))==len(original)
    assert Counter(token_length(blocks[i]) for i in result)==Counter(token_length(blocks[i]) for i in original)
    return result


def wrong_donors(blocks,local,original):
    """Maximize nonoriginal IDs separately in exact complete-token buckets."""
    original_set=set(original);candidates=remote_candidates(blocks,local);result=[]
    required=Counter(token_length(blocks[i]) for i in original)
    for length,count in sorted(required.items()):
        old=[i for i in original if token_length(blocks[i])==length]
        greatest=max(blocks[i]['source_index'] for i in old)
        pool=[i for i in candidates if token_length(blocks[i])==length]
        ordered=sorted(pool,key=lambda i:(blocks[i]['source_index']<=greatest,blocks[i]['source_index']))
        alternatives=[i for i in ordered if i not in original_set]
        retained=[i for i in ordered if i in original_set]
        chosen=(alternatives+retained)[:count];assert len(chosen)==count
        assert len(set(chosen)-original_set)==min(count,len(alternatives))
        result.extend(chosen)
    assert len(result)==len(set(result))==len(original)
    assert set(result).isdisjoint(local)
    assert Counter(token_length(blocks[i]) for i in result)==Counter(token_length(blocks[i]) for i in original)
    return result


class Selection:
    """One independent query's intervention, called once in each actual layer."""
    def __init__(self,arm,blocks,local,bounds,r0_layers):
        assert arm in ARMS and local
        self.arm=arm;self.blocks=blocks;self.local=list(local);self.bounds=list(bounds)
        self.reference=r0_layers;self.layer0=None;self.preference=None;self.visited=[]

    def choose(self,layer,default,scores):
        assert layer==len(self.visited),'actual layers must visit in model order once'
        self.visited.append(layer)
        reference=list(self.reference[layer]['remote_ids'])
        assert set(reference).isdisjoint(self.local)
        assert len(reference)==len(default),'same source pool gives same top-k count'
        if layer==0:
            self.layer0=list(default)
            values=scores.tolist()
            self.preference=sorted(remote_candidates(self.blocks,self.local),key=lambda i:(-values[i],self.blocks[i]['source_index']))
        if self.arm=='R0':
            assert default==reference,'fresh source/query replay changed main selection'
            return list(default)
        if self.arm=='L0':return []
        if self.arm=='C0':return time_order(self.blocks,self.local,self.bounds)[:len(reference)]
        if self.arm=='L1':return list(self.layer0)
        if self.arm=='C1':return matched_preference(self.blocks,reference,time_order(self.blocks,self.local,self.bounds))
        if self.arm=='L2':return matched_preference(self.blocks,reference,self.preference)
        if self.arm=='D0':return wrong_donors(self.blocks,self.local,reference)
        raise AssertionError(self.arm)


def exposure(blocks,local,reference,selected):
    r=set(reference);s=set(selected)
    def ancestors(ids):return set(a for i in ids for a in blocks[i]['ancestors'])
    actual=ancestors(local+selected);local_ancestry=ancestors(local)
    return dict(reference_remote_ids=list(reference),remote_ids=list(selected),
        reference_tokens=sum(token_length(blocks[i]) for i in reference),remote_tokens=sum(token_length(blocks[i]) for i in selected),
        retained_original=len(r&s),changed_remote=len(s-r),full_donor_replacement=bool(r) and not r&s,
        source_ancestor_union=len(actual),local_ancestor_union=len(local_ancestry),extra_ancestors=len(actual-local_ancestry),
        source_ancestor_span_seconds=(max(blocks[i]['actual_time'] for i in actual)-min(blocks[i]['actual_time'] for i in actual)) if actual else 0.,
        remote_actual_times=[blocks[i]['actual_time'] for i in selected],
        remote_image_tokens=[len(blocks[i]['image_rows']) for i in selected])
