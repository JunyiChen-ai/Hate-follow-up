"""Maximum cross-window timestamp permutations within exact token buckets."""
from collections import Counter,defaultdict


def permutation(times,lengths,source_indices,window_seconds=8):
    assert len(times)==len(lengths)==len(source_indices)
    groups=defaultdict(list)
    for index,length in enumerate(lengths):groups[int(length)].append(index)
    donors=list(range(len(times)));stats=[]
    for length,ids in sorted(groups.items()):
        ordered=sorted(ids,key=lambda i:(int(times[i]//window_seconds),source_indices[i]))
        owners=Counter(int(times[i]//window_seconds) for i in ordered)
        largest=max(owners.values());n=len(ordered)
        # A rotation by the largest color class achieves the maximum possible
        # different-owner assignment: n if max<=n/2, else 2*(n-max).
        for k,index in enumerate(ordered):donors[index]=ordered[(k+largest)%n]
        changed=sum(int(times[i]//window_seconds)!=int(times[donors[i]]//window_seconds) for i in ordered)
        optimum=min(n,2*(n-largest));assert changed==optimum
        stats.append(dict(block_tokens=length,n=n,largest_owner_group=largest,cross_window=changed,maximum=optimum))
    assert sorted(donors)==list(range(len(times)))
    assert all(lengths[i]==lengths[j] for i,j in enumerate(donors))
    return donors,stats
