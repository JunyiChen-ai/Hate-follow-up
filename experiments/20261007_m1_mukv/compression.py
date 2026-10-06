"""Paper-literal attention/FFT-bin compression; no labels or claimed saliency."""
import math
import torch


def normalized(values):
    assert values.ndim==1 and values.device.type=='cpu' and torch.isfinite(values).all()
    span=values.max()-values.min()
    return (values-values.min())/span if span>0 else torch.zeros_like(values)


def indicators(last_rotated_keys,attention):
    """[KVhead,currentvisualtoken,channel] -> FFT-bin-associated score."""
    assert last_rotated_keys.device.type==attention.device.type=='cpu' and last_rotated_keys.ndim==3
    x=last_rotated_keys.float();assert attention.shape==(x.shape[1],) and torch.isfinite(x).all()
    frequency=torch.fft.fft(x,dim=1).abs().mean((0,2))
    a=normalized(attention.float());f=normalized(frequency)
    return dict(attention=attention.float(),frequency=frequency,normalized_attention=a,normalized_frequency=f,score=.5*a+.5*f)


def selection(last_rotated_keys,attention,ratio):
    assert 0<ratio<=1
    result=indicators(last_rotated_keys,attention);n=len(attention);assert n>0
    count=max(1,math.floor(ratio*n));rank=sorted(range(n),key=lambda i:(-float(result['score'][i]),i))
    return sorted(rank[:count]),result
