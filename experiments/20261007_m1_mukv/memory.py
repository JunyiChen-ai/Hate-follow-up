"""Full-precision sparse source KV, layer-wise offload and truthful source IDs."""
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]


def save_tensor(path,value):
    x=value.detach().cpu().contiguous();assert x.dtype in (torch.bfloat16,torch.float32)
    a=x.view(torch.uint16).numpy() if x.dtype==torch.bfloat16 else x.numpy();np.save(path,a,allow_pickle=False)
    return dict(path=str(Path(path).resolve().relative_to(ROOT)),shape=list(x.shape),dtype=str(x.dtype),bytes=a.nbytes)


def load_tensor(meta):
    path=ROOT/meta['path'];assert path.resolve().is_relative_to(ROOT)
    a=np.load(path,allow_pickle=False);assert list(a.shape)==meta['shape']
    if meta['dtype']=='torch.bfloat16':assert a.dtype==np.uint16;return torch.from_numpy(a.copy()).view(torch.bfloat16)
    assert a.dtype==np.float32;return torch.from_numpy(a.copy())


class Memory:
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.blocks=[];self.reps=[]

    def append(self,window,grain,quadrant,frames,positions,visual_rows,keys,values,keep,history):
        i=len(self.blocks);folder=self.folder/str(i);folder.mkdir(exist_ok=True)
        data=torch.stack([torch.cat([keys[l] for l in range(len(keys))]),torch.cat([values[l] for l in range(len(values))])]).cpu()
        assert positions.shape==(3,1,data.shape[-2]) and keep==sorted(set(keep)) and keep
        keep_rows=torch.tensor(keep);reduced=data[:,:,:,keep_rows,:];meta=save_tensor(folder/'kv.npy',reduced)
        vr=[j for j,k in enumerate(keep) if k in visual_rows];assert vr
        rep=reduced[0][-1,:,vr,:].float().mean(1).flatten();ancestors=set(history)
        for h in history:ancestors.update(self.blocks[h]['ancestors'])
        block=dict(id=i,window=window,grain=grain,quadrant=quadrant,source_frames=[{k:f[k] for k in ('index','time','path')} for f in frames],
            positions=positions[:,:,keep_rows].cpu(),visual_rows=vr,kept_sequence_rows=keep,shape=list(reduced.shape),storage=meta,
            direct_ancestors=list(history),ancestors=sorted(ancestors))
        self.blocks.append(block);self.reps.append(rep);return block

    def layer(self,index,layer,device):
        b=self.blocks[index];a=np.load(ROOT/b['storage']['path'],allow_pickle=False,mmap_mode='r')
        assert list(a.shape)==b['shape'];t=torch.from_numpy(a[:,layer].copy())
        if b['storage']['dtype']=='torch.bfloat16':assert a.dtype==np.uint16;t=t.view(torch.bfloat16)
        else:assert a.dtype==np.float32
        return t[0].unsqueeze(0).to(device),t[1].unsqueeze(0).to(device),b['positions'].to(device)

    def earlier(self,window,grain):return [b['id'] for b in self.blocks if b['window']<window and b['grain']==grain]

    def metadata(self):return [{k:v[:,0].tolist() if k=='positions' else v for k,v in b.items()} for b in self.blocks]

    def close(self,release):
        if release:
            for b in self.blocks:
                p=ROOT/b['storage']['path'];p.unlink();p.parent.rmdir()
            self.folder.rmdir()
        self.blocks.clear();self.reps.clear()
