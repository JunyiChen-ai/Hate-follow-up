"""Physical layer/row identity, quantization bound and exact BF16 text storage."""
import json
import socket
from pathlib import Path
import torch
from memory import Memory


def main():
    torch.set_num_threads(4);torch.manual_seed(0);print(socket.gethostname(),flush=True);cases=[]
    out=Path(__file__).resolve().parents[2]/'runs/20261007_m1_streamingtom/memory_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    for dtype in (torch.float32,torch.bfloat16):
        for n in (5,50,51):
            folder=out/(str(dtype).split('.')[-1]+str(n));m=Memory(folder);t=n+9
            keys={l:torch.randn(1,8,t,128).to(dtype) for l in range(36)};values={l:torch.randn(1,8,t,128).to(dtype) for l in range(36)}
            visual=list(range(3,3+n));text=[i for i in range(t) if i not in visual]
            position=torch.arange(t)[None,None].expand(3,1,-1).clone();features=torch.randn(196,64).to(dtype);deep=[torch.randn_like(features) for _ in range(3)]
            try:
                m.append(dict(index=12,time=2.),position,visual,keys,values,[],features,deep)
                rep=m.representatives();assert rep.shape==(36,1,1024)
                for l in range(36):
                    assert torch.equal(rep[l,0],keys[l][0,:,visual,:].float().mean(-2).flatten())
                    k,v,p=m.layer(0,l,'cpu');assert k.shape==v.shape==keys[l].shape and torch.equal(p,position)
                    assert torch.equal(k[:,:,text,:],keys[l][:,:,text,:]) and torch.equal(v[:,:,text,:],values[l][:,:,text,:])
                    for actual,source in ((k,keys[l]),(v,values[l])):
                        scale=(source[:,:,visual,:].float().amax(-2,keepdim=True)-source[:,:,visual,:].float().amin(-2,keepdim=True))/15
                        # BF16 cast rounds the reconstructed value, whose
                        # magnitude can exceed |source| by the half-step.
                        bound=scale/2+2e-5+((source[:,:,visual,:].float().abs()+scale/2)*.008 if dtype==torch.bfloat16 else 0)
                        assert ((actual[:,:,visual,:].float()-source[:,:,visual,:].float()).abs()<=bound).all()
                ff,ds=m.local_features(0);assert torch.equal(ff,features) and all(torch.equal(a,b) for a,b in zip(ds,deep))
                cases.append(dict(dtype=str(dtype),visual_tokens=n,layers=36,kv_heads=8,dimension=128,representative_layer_identity=True,
                    text_rawbits_exact=True,all_visual_quantization_bounds=True,all_LOCAL_fullfeatures_DeepStack_exact=True))
                print('MEMORY_CASE_PASS',dtype,n,flush=True)
            finally:m.close(release=True)
            assert not folder.exists()
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,scope='isolated physical source memory; no complete model/reader or semantics',cases=cases),indent=2)+'\n');print('MEMORY_CPU_PASS',flush=True)


if __name__=='__main__':main()
