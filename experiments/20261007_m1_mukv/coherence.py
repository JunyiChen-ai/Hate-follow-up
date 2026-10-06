"""Unscored same-model multigrain candidate selection with source-index ties."""
import torch


def cosine(query,keys):
    assert query.ndim==1 and keys.ndim==2 and keys.shape[1]==len(query)
    assert query.device.type==keys.device.type=='cpu'
    q=query.float();k=keys.float();den=k.norm(dim=1)*q.norm();safe=torch.where(den>0,den,torch.ones_like(den))
    result=torch.where(den>0,(k@q)/safe,torch.zeros_like(den));assert torch.isfinite(result).all()
    return result


def retrieve(query,representatives,source_order,excluded):
    assert set(representatives)==set(source_order)==set(excluded)=={'segment','frame','patch'}
    original={};candidates={}
    for grain in representatives:
        original[grain]=cosine(query,representatives[grain]);order=source_order[grain]
        assert len(order)==len(original[grain])
        allowed=[i for i in range(len(order)) if i not in excluded[grain]]
        candidates[grain]=sorted(allowed,key=lambda i:(-float(original[grain][i]),order[i]))[:4]
    coarse=candidates['segment'][:2]
    context=representatives['segment'][coarse].float().mean(0) if coarse else query.float()
    scores={};chosen={};consistency={}
    for grain in representatives:
        consistency[grain]=cosine(context,representatives[grain]);weight=0. if grain=='segment' else .3
        scores[grain]=(1-weight)*original[grain]+weight*consistency[grain]
        chosen[grain]=sorted(candidates[grain],key=lambda i:(-float(scores[grain][i]),source_order[grain][i]))[:2]
    return dict(selected=chosen,candidates=candidates,coarse_ids=coarse,coarse_vector=context,
        original_scores=original,consistency_scores=consistency,reranked_scores=scores)
