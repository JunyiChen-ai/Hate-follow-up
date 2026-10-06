"""Per-head/channel visual-group uint4; true token count, FP32 ranges, no digest."""
import torch


def compress(x):
    assert x.device.type=='cpu' and x.ndim>=3 and x.shape[-2]>0 and torch.isfinite(x).all()
    y=x.float();minimum=y.amin(-2,keepdim=True);scale=(y.amax(-2,keepdim=True)-minimum)/15
    safe=torch.where(scale==0,torch.ones_like(scale),scale)
    codes=((y-minimum)/safe).round().clamp(0,15).to(torch.uint8)
    if codes.shape[-2]%2:
        codes=torch.cat([codes,torch.zeros_like(codes[...,:1,:])],-2)
    packed=codes[...,0::2,:]|(codes[...,1::2,:]<<4)
    return dict(packed=packed,minimum=minimum,scale=scale,shape=list(x.shape),dtype=str(x.dtype))


def decompress(blob,device='cpu'):
    packed=blob['packed'];minimum=blob['minimum'];scale=blob['scale'];shape=blob['shape']
    assert packed.dtype==torch.uint8 and packed.device.type==minimum.device.type==scale.device.type=='cpu'
    assert minimum.dtype==scale.dtype==torch.float32 and list(minimum.shape)==shape[:-2]+[1,shape[-1]]
    assert list(packed.shape)==shape[:-2]+[(shape[-2]+1)//2,shape[-1]]
    codes=torch.stack([packed&15,packed>>4],-2).flatten(-3,-2)[...,:shape[-2],:]
    assert list(codes.shape)==shape
    x=codes.float()*scale+minimum
    dtype={'torch.float32':torch.float32,'torch.bfloat16':torch.bfloat16}[blob['dtype']]
    return x.to(device=device,dtype=dtype)


def storage_bytes(blob):
    return sum(blob[k].numel()*blob[k].element_size() for k in ('packed','minimum','scale'))
