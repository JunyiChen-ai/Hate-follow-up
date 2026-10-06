"""Frozen dual-path grouping, with explicit deterministic official-DPC adaptation."""
import math
import torch
import torch.nn.functional as F

BUDGET=50
SIMILARITY=.9
NEIGHBORS=7
CENTER_WEIGHT=.6


def rank(values,indices,largest=True):
    assert values.device.type=='cpu' and len(values)==len(indices)
    return sorted(range(len(indices)),key=lambda i:((-float(values[i]) if largest else float(values[i])),indices[i]))


def group(features,saliency,grid,previous=None,previous_grid=None):
    """Every group records exact roots/members/weights for all DeepStack levels."""
    assert features.device.type==saliency.device.type=='cpu' and features.ndim==2
    x=features.float();n=len(x);assert n>0 and saliency.shape==(n,)
    assert torch.isfinite(x).all() and torch.isfinite(saliency).all()
    if previous is None or list(grid)!=list(previous_grid):
        similarity=None;static=torch.zeros(n,dtype=torch.bool)
    else:
        assert previous.device.type=='cpu' and previous.shape==x.shape
        similarity=(F.normalize(x,dim=-1,eps=1e-12)*F.normalize(previous.float(),dim=-1,eps=1e-12)).sum(-1)
        static=similarity>SIMILARITY
    ss=static.nonzero().flatten().tolist();dd=(~static).nonzero().flatten().tolist();g=min(BUDGET,n)
    ks=math.floor(g*len(ss)/n);kd=g-ks
    if kd>len(dd):kd=len(dd);ks=g-kd
    if ks>len(ss):ks=len(ss);kd=g-ks
    groups=[];dpc=None
    if ks:
        y=x[ss];m=len(ss)
        if ks==m:centers=list(range(m));distance=None
        else:
            distance=torch.cdist(y,y)/math.sqrt(x.shape[1])
            k=min(NEIGHBORS,m-1)
            # The official nearest-distance pool includes the diagonal.
            near=distance.sort(dim=-1,stable=True).values[:,:k]
            density=torch.exp(-near.square().mean(-1))
            # Official code compares density[row] > density[column], so these
            # are LOWER-density neighbors. Equal density uses source-index order
            # instead of upstream random 1e-6 jitter; this is declared adaptation.
            lower=(density[:,None]>density[None,:])|((density[:,None]==density[None,:])&(torch.arange(m)[:,None]<torch.arange(m)[None,:]))
            separation=torch.where(lower,distance,distance.max()).min(-1).values
            score=density*separation;centers=sorted(rank(score,ss)[:ks])
            dpc=dict(neighbors=k,density=density.tolist(),separation=separation.tolist(),score=score.tolist(),centers=[ss[i] for i in centers],direction='official-lower-density',random_jitter=False)
        noncenters=[i for i in range(m) if i not in centers];assigned={c:[] for c in centers}
        if noncenters:
            for i in noncenters:
                # Centers sorted by actual token ID: argmin ties are explicit.
                selected=int(distance[i,centers].argmin());assigned[centers[selected]].append(i)
        for center in centers:
            rest=assigned[center];members=[ss[center]]+[ss[i] for i in rest]
            weights=[CENTER_WEIGHT]+[(1-CENTER_WEIGHT)/len(rest)]*len(rest) if rest else [1.]
            groups.append(dict(root=ss[center],members=members,weights=weights,kind='static'))
    if kd:
        for i in rank(saliency[dd].float(),dd)[:kd]:
            token=dd[i];groups.append(dict(root=token,members=[token],weights=[1.],kind='dynamic'))
    groups.sort(key=lambda r:r['root'])
    assert len(groups)==g and len({r['root'] for r in groups})==g
    assert all(abs(sum(r['weights'])-1)<1e-12 for r in groups)
    static_members=[i for r in groups if r['kind']=='static' for i in r['members']]
    assert len(set(static_members))==len(static_members)
    if ks:assert sorted(static_members)==ss
    return dict(original_tokens=n,retained_tokens=g,static_ids=ss,dynamic_ids=dd,static_budget=ks,dynamic_budget=kd,
        reset_adjacency=similarity is None,similarity=None if similarity is None else similarity.tolist(),groups=groups,dpc=dpc)


def aggregate(features,plan):
    assert features.ndim==2 and len(features)==plan['original_tokens']
    return torch.stack([(features[r['members']].float()*torch.tensor(r['weights'],device=features.device,dtype=torch.float32)[:,None]).sum(0).to(features.dtype)
        for r in plan['groups']])
