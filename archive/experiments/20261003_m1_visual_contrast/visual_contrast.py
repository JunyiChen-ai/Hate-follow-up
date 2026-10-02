"""Source-scheduled visual corruption with consistent still-image temporal copies."""
import torch


def schedule(step=500):
    beta=1e-5+(.005-1e-5)*torch.sigmoid(torch.linspace(-6,6,1000,dtype=torch.float32))
    assert 0<=step<1000
    return float(torch.cumprod(1-beta,dim=0)[step])


def corrupt_pixels(pixels,patch_size,temporal_patch_size,abar=None):
    assert pixels.device.type=='cpu' and pixels.ndim==2
    p,t=int(patch_size),int(temporal_patch_size)
    assert pixels.shape[1]==3*t*p*p
    x=pixels.float().reshape(-1,3,t,p,p)
    assert torch.equal(x,x[:,:,:1].expand_as(x)), 'not repeated still-image temporal copies'
    a=schedule() if abar is None else float(abar);assert 0<=a<=1
    gen=torch.Generator(device='cpu').manual_seed(0)
    noise=torch.randn((len(x),3,1,p,p),generator=gen,dtype=torch.float32)
    z=(a**.5*x+(1-a)**.5*noise).reshape_as(pixels)
    assert torch.isfinite(z).all()
    return z,{'abar':a,'noise_step':500,'seed':0,'patch_size':p,'temporal_patch_size':t,
        'patches':len(x),'same_temporal_noise':True,'pixel_rms_change':float((z-pixels.float()).square().mean().sqrt())}


def class_contrast(clean,corrupted,coefficient=1.):
    return (1+coefficient)*clean-coefficient*corrupted


def mix_window_pixels(clean,corrupted,grid,frame_indices):
    """Replace complete patch blocks of selected images; retain all other pixels."""
    assert clean.device.type==corrupted.device.type=='cpu' and clean.shape==corrupted.shape
    assert grid.ndim==2 and grid.shape[1]==3
    sizes=grid.long().prod(1).tolist();assert sum(sizes)==len(clean)
    selected=list(frame_indices)
    assert len(set(selected))==len(selected) and all(0<=i<len(sizes) for i in selected)
    boundaries=[0]
    for n in sizes:boundaries.append(boundaries[-1]+n)
    mixed=clean.float().clone();ranges=[]
    for i in selected:
        a,b=boundaries[i:i+2];mixed[a:b]=corrupted[a:b];ranges.append([a,b])
    return mixed,{'frames':selected,'patch_ranges':ranges,'pixel_rows':sum(b-a for a,b in ranges)}
