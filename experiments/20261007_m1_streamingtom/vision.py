"""Actual Qwen projector/DeepStack and chunked last-attention observation."""
import time
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision

CHUNK=64


def tick(device):
    if device.type=='cuda':torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def extract(j,encoded):
    model=j.model.model;attention=model.visual.blocks[-1].attn;state={};hooks=[];start=tick(j.device)
    def before(module,args,kwargs):
        state['cu']=kwargs.get('cu_seqlens',args[1] if len(args)>1 else None)
        state['pos']=kwargs.get('position_embeddings',args[2] if len(args)>2 else None)
        assert state['cu'] is not None and state['pos'] is not None
    def observe(module,args,output):
        assert 'saliency' not in state
        started=tick(j.device)
        q,k,_=output.reshape(len(output),3,attention.num_heads,-1).permute(1,0,2,3).unbind(0)
        q,k=apply_rotary_pos_emb_vision(q,k,*state['pos']);scores=[]
        bounds=state['cu'].tolist()
        for a,b in zip(bounds[:-1],bounds[1:]):
            keys=k[a:b].float().transpose(0,1);total=torch.zeros(b-a,device=keys.device,dtype=torch.float32)
            for s in range(a,b,CHUNK):
                queries=q[s:min(s+CHUNK,b)].float().transpose(0,1)
                weights=torch.softmax((queries@keys.transpose(1,2))*attention.scaling,-1)
                total+=weights.sum((0,1))/(attention.num_heads*(b-a))
            scores.append(total)
        raw=torch.cat(scores);merge=model.visual.spatial_merge_size
        assert len(raw)%merge**2==0
        state['saliency']=raw.reshape(-1,merge**2).sum(1).cpu()
        state['saliency_seconds']=tick(j.device)-started
    try:
        hooks=[attention.register_forward_pre_hook(before,with_kwargs=True),attention.qkv.register_forward_hook(observe)]
        result=model.get_image_features(encoded['pixel_values'].to(j.device),encoded['image_grid_thw'].to(j.device),return_dict=True)
        features=torch.cat(tuple(result.pooler_output),0).detach().cpu();deep=[v.detach().cpu() for v in result.deepstack_features]
        assert len(features)==len(state['saliency']) and all(v.shape==features.shape for v in deep)
        return dict(features=features,deepstack=deep,saliency=state['saliency'],grid=encoded['image_grid_thw'].tolist(),
            seconds=tick(j.device)-start,saliency_seconds=state['saliency_seconds'],cost_scope='full actual vision + additional chunked attention observation/transfer; saliency is paid, not zero-cost')
    finally:
        for hook in hooks:hook.remove()
