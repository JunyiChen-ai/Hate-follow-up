"""Arithmetic bounds and independent weighted-member oracle, not full method tests."""
import json
import socket
from pathlib import Path
import torch
from ctr import group,aggregate
from quantization import compress,decompress,storage_bytes


def main():
    torch.set_num_threads(4);torch.manual_seed(0);cases=[];print(socket.gethostname(),flush=True)
    for dtype in (torch.float32,torch.bfloat16):
        for n in (1,2,49,50,51,196,320):
            x=torch.randn(2,8,n,16).to(dtype);x[...,0]=3
            blob=compress(x);y=decompress(blob);scale=blob['scale']
            assert torch.equal(y[...,0],x[...,0])
            exact=decompress({**blob,'dtype':'torch.float32'})
            assert ((exact-x.float()).abs()<=scale/2+2e-6).all()
            f=torch.randn(n,16).to(dtype);sal=torch.rand(n)
            for prior,grid in ((None,None),(f,[1,16,16]),(torch.zeros_like(f),[1,16,16]),(f,[1,32,16])):
                p=group(f,sal,[1,16,16],prior,grid);z=aggregate(f,p)
                assert z.shape==(min(n,50),16)
                assert len({g['root'] for g in p['groups']})==min(n,50)
                assert p==group(f,sal,[1,16,16],prior,grid)
                for g,v in zip(p['groups'],z):
                    oracle=sum(f[i].double()*w for i,w in zip(g['members'],g['weights'])).to(dtype)
                    assert torch.allclose(v.float(),oracle.float(),atol=.01 if dtype==torch.bfloat16 else 1e-6,rtol=0)
            cases.append(dict(dtype=str(dtype),tokens=n,constant_channel_exact=True,float32_half_step_error_bound=True,quant_bytes=storage_bytes(blob),dual_path_cases=4))
    out=Path(__file__).resolve().parents[2]/'runs/20261007_m1_streamingtom/algorithm_cpu_checks'
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,
        scope='isolated grouping/uint4 arithmetic only; no complete source/model/renderer/GPU or quality claim',cases=cases),indent=2)+'\n')
    print('ALGORITHMS_PASS',flush=True)


if __name__=='__main__':main()
