"""Actual per-image Qwen feature extraction and source-bound reuse in suffixes."""
import torch
from src.stance_cache import positions


@torch.no_grad()
def extract(j,encoded):
    output=j.model.model.get_image_features(encoded['pixel_values'].to(j.device),encoded['image_grid_thw'].to(j.device),return_dict=True)
    features=[v.detach().cpu() for v in output.pooler_output];counts=[len(v) for v in features]
    assert sum(counts)==int((encoded['input_ids']==j.image_token_id).sum())
    deep=[]
    for value in output.deepstack_features:
        parts=list(value.detach().cpu().split(counts));assert len(parts)==len(features);deep.append(parts)
    return features,[[deep[l][i] for l in range(len(deep))] for i in range(len(features))]


def reuse(j,encoded,features,deepstack):
    ids=encoded['input_ids'].to(j.device);grids=encoded['image_grid_thw'].to(j.device)
    counts=[int(g.prod())//j.processor.image_processor.merge_size**2 for g in grids]
    assert len(counts)==len(features)==len(deepstack) and all(len(f)==n for f,n in zip(features,counts))
    visual=ids==j.image_token_id;assert int(visual.sum())==sum(counts)
    p,_=positions(j,ids,grids);embedding=j.model.model.get_input_embeddings()(ids)
    embedding[visual]=torch.cat(features).to(j.device,j.dtype)
    levels=len(deepstack[0]);assert all(len(d)==levels and all(v.shape==f.shape for v in d) for f,d in zip(features,deepstack))
    ds=[torch.cat([d[l] for d in deepstack]).to(j.device,j.dtype) for l in range(levels)]
    return dict(inputs_embeds=embedding,visual_pos_masks=visual,deepstack_visual_embeds=ds),p
