"""Per-video rebuildable dense KV, without all-corpus persistence or digests."""
from pathlib import Path
import numpy as np
import torch
from retrieval import ROOT, mean_image_key


def tensor_array(value):
    cpu = value.detach().to('cpu').contiguous()
    if cpu.dtype == torch.bfloat16:
        return cpu.view(torch.uint16).numpy(), 'bfloat16_bits'
    assert cpu.dtype == torch.float32
    return cpu.numpy(), 'float32'


def array_tensor(value, encoding, device):
    # A copy avoids exposing a writable tensor backed by read-only mmap storage.
    result = torch.from_numpy(np.array(value, copy=True))
    if encoding == 'bfloat16_bits':
        assert result.dtype == torch.uint16
        result = result.view(torch.bfloat16)
    else:
        assert encoding == 'float32' and result.dtype == torch.float32
    return result.to(device)


class VideoMemory:
    """Caller supplies actual model tensors; this class does not synthesize KV."""
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.folder.relative_to((ROOT/'runs/20261006_m1_rekv').resolve())
        self.folder.mkdir(parents=True, exist_ok=True)
        self.blocks = []
        self.arrays = []
        self.representatives = []

    def append(self, index, time, positions, image_rows, layer_keys, layer_values, direct_ancestors, input_evidence):
        number = len(self.blocks)
        assert number == 0 or self.blocks[-1]['source_index'] < index
        assert len(layer_keys) == len(layer_values) > 0
        assert positions.ndim == 3 and positions.shape[:2] == (3,1)
        length = positions.shape[2]
        assert length > 0 and image_rows and len(set(image_rows)) == len(image_rows)
        assert all(0 <= row < length for row in image_rows)
        assert all(0 <= ancestor < number for ancestor in direct_ancestors)
        shapes = {tuple(value.shape) for value in [*layer_keys, *layer_values]}
        assert len(shapes) == 1
        shape = next(iter(shapes))
        assert len(shape) == 4 and shape[0] == 1 and shape[2] == length
        dtype = {value.dtype for value in [*layer_keys, *layer_values]}
        assert len(dtype) == 1
        keys = torch.stack([value[0].detach().to('cpu') for value in layer_keys])
        values = torch.stack([value[0].detach().to('cpu') for value in layer_values])
        storage, encoding = tensor_array(torch.stack([keys, values]))
        path = self.folder/f'block_{number:06d}.npy'
        assert not path.exists(), 'temporary source block cannot overwrite earlier input'
        np.save(path, storage, allow_pickle=False)
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        assert array.shape == storage.shape and array.dtype == storage.dtype
        representative = torch.stack([mean_image_key(value,image_rows).to('cpu') for value in layer_keys])
        assert torch.isfinite(representative).all()
        inherited = set(direct_ancestors)
        for ancestor in direct_ancestors:
            inherited.update(self.blocks[ancestor]['ancestors'])
        block = dict(block_id=number,source_index=int(index),actual_time=float(time),
            positions=positions.to('cpu').clone(),image_rows=list(image_rows),
            direct_ancestors=list(direct_ancestors),ancestors=sorted(inherited),
            input_evidence=input_evidence,encoding=encoding,shape=list(array.shape),
            storage_bytes=int(array.nbytes),path=str(path.relative_to(ROOT)))
        self.blocks.append(block)
        self.arrays.append(array)
        self.representatives.append(representative)
        return block

    def layer(self, block_id, layer, device):
        block = self.blocks[block_id]
        assert 0 <= layer < block['shape'][1]
        data = self.arrays[block_id]
        key = array_tensor(data[0,layer],block['encoding'],device)[None]
        value = array_tensor(data[1,layer],block['encoding'],device)[None]
        assert key.shape == value.shape == (1,*block['shape'][2:])
        return key,value,block['positions'].to(device)

    def representative_layers(self, device):
        assert self.blocks
        return torch.stack(self.representatives,dim=1).to(device)

    def metadata(self):
        result=[]
        for block in self.blocks:
            row={key:value for key,value in block.items() if key not in ('positions','path')}
            row['positions']=block['positions'][:,0].tolist()
            row['temporary_storage']='owned per-video denseKV; released after atomic video completion'
            result.append(row)
        return result

    def close(self, release=False):
        for array in self.arrays:
            array._mmap.close()
        self.arrays=[]
        if release:
            for block in self.blocks:
                path=ROOT/block['path']
                assert path.parent == self.folder
                path.unlink()
        self.representatives=[]
