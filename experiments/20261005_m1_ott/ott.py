"""Complete weighted facility-location, LOO-mass OT, and temporal component graph."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())
sys.path.insert(0,str(ROOT))
from src.sparse_visual_prefix import capture_native,aggregate,pack,prefill
SPEC=json.loads((Path(__file__).parent/'spec.json').read_text())


def spatial(unit,weights,k):
    sim=unit@unit.T
    coverage=torch.zeros(len(unit)); chosen=[]; mask=torch.zeros(len(unit),dtype=torch.bool)
    for _ in range(k):
        gain=((sim-coverage[:,None]).clamp_min(0)*weights[:,None]).sum(0)
        gain[mask]=-torch.inf; index=int(gain.argmax())
        chosen.append(index); mask[index]=True; coverage=torch.maximum(coverage,sim[:,index])
    chosen=sorted(chosen)
    selected=sim[:,chosen]
    # Equal best similarities have zero removal contribution, so their owner tie is immaterial.
    order=torch.argsort(selected,dim=1,descending=True,stable=True)
    best=selected.gather(1,order[:,:1]).squeeze(1)
    second=selected.gather(1,order[:,1:2]).squeeze(1)
    contribution=torch.zeros(k)
    contribution.scatter_add_(0,order[:,0],(best-second).clamp_min(0)*weights)
    scaled=contribution/(contribution.max()+SPEC['mass_epsilon'])
    mass=torch.softmax(-scaled/SPEC['mass_temperature'],0)
    return chosen,mass,contribution


def sinkhorn(a,b,cost):
    a=a/a.sum(-1,keepdim=True).clamp_min(1e-12)
    b=b/b.sum(-1,keepdim=True).clamp_min(1e-12)
    la=a.clamp_min(1e-30).log(); lb=b.clamp_min(1e-30).log()
    logk=-cost/SPEC['sinkhorn_epsilon']; u=torch.zeros_like(la); v=torch.zeros_like(lb)
    for _ in range(SPEC['sinkhorn_iterations']):
        v=lb-torch.logsumexp(logk+u[:,:,None],dim=1)
        u=la-torch.logsumexp(logk+v[:,None,:],dim=2)
    plan=(u[:,:,None]+logk+v[:,None,:]).exp()
    residual=torch.maximum((plan.sum(2)-a).abs().amax(1),(plan.sum(1)-b).abs().amax(1))
    return plan,residual


def budgets(cost,total,cap):
    allocation=torch.zeros(len(cost),dtype=torch.long); active=torch.ones(len(cost),dtype=torch.bool)
    remaining=int(total)
    while remaining and active.any():
        idx=active.nonzero().flatten()
        target=torch.softmax(-cost[idx]/SPEC['budget_temperature'],0)*remaining
        integer=target.floor().long(); left=remaining-int(integer.sum())
        if left:
            ties=torch.argsort(target-target.floor(),descending=True,stable=True)
            integer[ties[:left]]+=1
        room=cap-allocation[idx]; applied=torch.minimum(integer,room)
        allocation[idx]+=applied; remaining=int((integer-applied).sum())
        active[idx[allocation[idx]>=cap]]=False
        if not int(applied.sum()): break
    assert int(allocation.sum())==min(total,len(cost)*cap)
    return allocation.tolist(),remaining


def matching(plan,budget):
    # Raw coupling; flatten order is destination then source. Destination may repeat.
    order=torch.argsort(plan.flatten(),descending=True,stable=True).tolist()
    used=set(); matches=[]; k=plan.shape[1]
    for flat in order:
        destination,source=divmod(flat,k)
        if source in used: continue
        used.add(source); matches.append([destination,source])
        if len(matches)==budget: break
    return matches if budget else []


def components(selected,matches,costs,n):
    f=len(selected); source_ids=[t*n+i for t,local in enumerate(selected) for i in local]
    parent={i:i for i in source_ids}; pruned=set(); edges=[]
    for t,mm in enumerate(matches):
        for destination,source in mm:
            dst=t*n+selected[t][destination]; src=(t+1)*n+selected[t+1][source]
            c=float(costs[t,destination,source]); keep=c<=SPEC['cost_threshold']
            if keep: parent[src]=dst
            else: pruned.add(src)
            edges.append(dict(source=src,destination=dst,cost=c,merge=keep))
    groups={}
    for source in source_ids:
        root=source; visits=0
        while parent[root]!=root:
            root=parent[root]; visits+=1; assert visits<f
        groups.setdefault(root,[]).append(source)
    roots=sorted(root for root in groups if root not in pruned)
    members=[groups[root] for root in roots]
    removed=sorted(source for root,group in groups.items() if root in pruned for source in group)
    assert roots and len(set(s for group in members for s in group))+len(removed)==len(source_ids)
    assert all(root==min(group) for root,group in zip(roots,members))
    return roots,members,removed,edges


@torch.no_grad()
def select(features,saliency,grids,merge):
    assert features.device.type=='cpu' and saliency.device.type=='cpu'
    gg=grids.tolist(); f=len(gg)
    assert f>=2 and all(g==gg[0] for g in gg) and gg[0][0]==1
    h,w=gg[0][1]//merge,gg[0][2]//merge; n=h*w
    assert features.shape[0]==len(saliency)==f*n and n>=2
    x=features.float().reshape(f,n,-1); weights=saliency.float().reshape(f,n)
    assert torch.isfinite(x).all() and torch.isfinite(weights).all() and (weights>=0).all() and (weights.sum(1)>0).all()
    weights=weights/(weights.sum(1,keepdim=True)+SPEC['mass_epsilon'])
    unit=torch.nn.functional.normalize(x,dim=-1,eps=SPEC['cosine_epsilon'])
    k=min(n,max(2,round(n*SPEC['retention']**(1-SPEC['temporal_share']))))
    selected=[]; masses=[]; contributions=[]
    for t in range(f):
        sel,mass,contribution=spatial(unit[t],weights[t],k)
        selected.append(sel); masses.append(mass); contributions.append(contribution.tolist())
    inds=torch.tensor(selected)
    kept=unit.gather(1,inds[:,:,None].expand(-1,-1,x.shape[-1]))
    yy=torch.arange(n)//w; xx=torch.arange(n)%w
    coords=torch.stack([yy.float()/max(1,h-1),xx.float()/max(1,w-1)],1)[inds]
    alpha=1-(unit[:-1]*unit[1:]).sum(-1).mean(-1).clamp(0,1)/2
    semantic=1-torch.bmm(kept[:-1],kept[1:].transpose(1,2)).clamp_min(0)
    distance=(coords[:-1,:,None]-coords[1:,None,:]).norm(dim=-1)/(2**.5)
    cost=alpha[:,None,None]*semantic+(1-alpha[:,None,None])*distance
    mass=torch.stack(masses); coupling,residual=sinkhorn(mass[:-1],mass[1:],cost)
    transport_cost=(coupling*cost).sum((1,2))
    target=round(f*n*SPEC['retention']); total=max(0,f*k-target)
    bb,unused=budgets(transport_cost,total,k-1)
    mm=[matching(coupling[t],bb[t]) for t in range(f-1)]
    assert all(len(m)==b for m,b in zip(mm,bb))
    roots,members,removed,edges=components(selected,mm,cost,n)
    return dict(grid=[h,w],tokens_per_frame=n,frames=f,spatial_budget=k,selected=selected,
        masses=mass.tolist(),contributions=contributions,alpha=alpha.tolist(),transport_cost=transport_cost.tolist(),
        marginal_residual=residual.tolist(),target_count=target,temporal_budget=total,budgets=bb,unused_budget=unused,
        matches=mm,edges=edges,roots=roots,members=members,pruned_sources=removed,
        counts=[sum(r//n==t for r in roots) for t in range(f)],original_count=f*n,
        spatial_count=f*k,retained_count=len(roots))


def dense_plan(plan):
    n=plan['original_count']; return dict(roots=list(range(n)),members=[[i] for i in range(n)])


@contextmanager
def capture(j):
    """Observe actual last vision QKV/rotary/chunks; native attention unchanged."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
    attn=j.model.model.visual.blocks[-1].attn; state={}
    def before(module,args,kwargs):
        state['cu']=kwargs.get('cu_seqlens',args[1] if len(args)>1 else None)
        state['pos']=kwargs.get('position_embeddings',args[2] if len(args)>2 else None)
        assert state['cu'] is not None and state['pos'] is not None
    def qkv(module,args,out):
        assert 'saliency' not in state
        q,k,_=out.reshape(len(out),3,attn.num_heads,-1).permute(1,0,2,3).unbind(0)
        q,k=apply_rotary_pos_emb_vision(q,k,*state['pos'])
        raw=[]
        cu=state['cu'].tolist()
        for a,b in zip(cu[:-1],cu[1:]):
            qq=q[a:b].float().transpose(0,1); kk=k[a:b].float().transpose(0,1)
            attention=torch.softmax((qq@kk.transpose(1,2))*attn.scaling,-1)
            raw.append(attention.mean((0,1)))
        raw=torch.cat(raw); merge=j.model.config.vision_config.spatial_merge_size
        assert len(raw)%merge**2==0
        state['saliency']=raw.reshape(-1,merge**2).sum(1).detach()
    hooks=[attn.register_forward_pre_hook(before,with_kwargs=True),attn.qkv.register_forward_hook(qkv)]
    try:
        with capture_native(j) as result:
            yield result
            assert 'saliency' in state
            result['saliency']=state['saliency']
            assert len(result['saliency'])==len(result['features'])
    finally:
        for hook in hooks: hook.remove()
