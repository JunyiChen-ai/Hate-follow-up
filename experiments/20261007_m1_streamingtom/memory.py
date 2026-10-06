"""Per-video quantized visual/full-precision text memory and temporary features."""
from pathlib import Path
import numpy as np
import torch
from quantization import compress,decompress,storage_bytes
ROOT=Path(__file__).resolve().parents[2]


def save_tensor(path,x):
    x=x.detach().cpu().contiguous()
    assert x.dtype in (torch.bfloat16,torch.float32)
    bits=x.view(torch.uint16).numpy() if x.dtype==torch.bfloat16 else x.numpy()
    np.save(path,bits,allow_pickle=False)
    return dict(path=str(Path(path).resolve().relative_to(ROOT)),shape=list(x.shape),dtype=str(x.dtype),bytes=bits.nbytes)


def read_tensor(meta):
    path=ROOT/meta['path'];assert path.resolve().is_relative_to(ROOT)
    a=np.load(path,allow_pickle=False)
    expected=np.uint16 if meta['dtype']=='torch.bfloat16' else np.float32
    assert a.dtype==expected and list(a.shape)==meta['shape']
    t=torch.from_numpy(a.copy())
    return t.view(torch.bfloat16) if meta['dtype']=='torch.bfloat16' else t


class Memory:
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True);self.blocks=[];self.reps=[]

    def append(self,frame,position,visual_rows,keys,values,history,full_features,deepstack):
        i=len(self.blocks);folder=self.folder/str(i);folder.mkdir(exist_ok=True)
        data=torch.stack([torch.cat([keys[l] for l in range(len(keys))]),torch.cat([values[l] for l in range(len(values))])]).cpu()
        assert data.ndim==5 and data.shape[1]==len(keys) and position.shape==(3,1,data.shape[-2])
        text=[t for t in range(data.shape[-2]) if t not in visual_rows];blob=compress(data[:,:,:,visual_rows,:])
        for name in ('packed','minimum','scale'):np.save(folder/(name+'.npy'),blob[name].numpy(),allow_pickle=False)
        text_meta=save_tensor(folder/'text.npy',data[:,:,:,text,:])
        features=save_tensor(folder/'full_features.npy',full_features)
        ds=[save_tensor(folder/f'full_deepstack_{l}.npy',v) for l,v in enumerate(deepstack)]
        ancestors=set(history)
        for ancestor in history:ancestors.update(self.blocks[ancestor]['ancestors'])
        block=dict(id=i,source_index=frame['index'],actual_time=frame['time'],positions=position.cpu(),visual_rows=list(visual_rows),text_rows=text,
            quantization={k:blob[k] for k in ('shape','dtype')},text=text_meta,features=features,deepstack=ds,
            direct_ancestors=list(history),ancestors=sorted(ancestors),storage_bytes=storage_bytes(blob)+text_meta['bytes'],
            shape=list(data.shape),folder=folder)
        self.blocks.append(block);self.reps.append(data[0][:,:,visual_rows,:].float().mean(-2).reshape(data.shape[1],-1))
        # Verify physical pack/depack and constant-channel reconstruction before
        # releasing the exact current source tensor; no zero-cost precision claim.
        restored=decompress({**blob,'dtype':'torch.float32'})
        assert ((restored-data[:,:,:,visual_rows,:].float()).abs()<=blob['scale']/2+2e-5).all()
        return block

    def layer(self,index,layer,device):
        b=self.blocks[index];folder=b['folder']
        blob=dict(shape=[2,*b['quantization']['shape'][2:]],dtype=b['quantization']['dtype'])
        for name in ('packed','minimum','scale'):
            a=np.load(folder/(name+'.npy'),allow_pickle=False,mmap_mode='r');blob[name]=torch.from_numpy(a[:,layer].copy())
        visual=decompress(blob)
        # Read only the requested layer. Loading all36 text-KV layers for every
        # layer/source query would multiply actual disk/CPU work by36.
        a=np.load(ROOT/b['text']['path'],allow_pickle=False,mmap_mode='r')
        assert list(a.shape)==b['text']['shape']
        text=torch.from_numpy(a[:,layer].copy())
        if b['text']['dtype']=='torch.bfloat16':
            assert a.dtype==np.uint16;text=text.view(torch.bfloat16)
        else:assert a.dtype==np.float32
        result=torch.empty((2,*b['shape'][2:]),dtype=visual.dtype)
        result[:, :,b['visual_rows'],:]=visual;result[:, :,b['text_rows'],:]=text
        return result[0].unsqueeze(0).to(device),result[1].unsqueeze(0).to(device),b['positions'].to(device)

    def representatives(self):
        assert self.reps
        return torch.stack(self.reps,dim=1)

    def local_features(self,index):
        b=self.blocks[index];return read_tensor(b['features']),[read_tensor(m) for m in b['deepstack']]

    def metadata(self):
        return [{k:(v[:,0].tolist() if k=='positions' else str(v.resolve().relative_to(ROOT)) if k=='folder' else v) for k,v in b.items()} for b in self.blocks]

    def close(self,release):
        self.reps.clear()
        if release:
            # Only this instance's per-video temporary payload is removed, after
            # the caller has atomically persisted the complete record/proofs.
            for folder in [b['folder'] for b in self.blocks]:
                for p in folder.iterdir():p.unlink()
                folder.rmdir()
            self.folder.rmdir()
        self.blocks.clear()
