"""Independent DFT and coherence arithmetic references, not method quality."""
import json
import socket
from pathlib import Path
import numpy as np
import torch
from compression import indicators,selection
from coherence import retrieve


def main():
    torch.set_num_threads(4);torch.manual_seed(0);print(socket.gethostname(),flush=True);cases=[]
    for n in (1,2,5,8,13):
        x=torch.randn(8,n,16);attention=torch.rand(n)
        data=indicators(x,attention);a=x.numpy().astype(np.float64)
        phase=np.exp(-2j*np.pi*np.arange(n)[:,None]*np.arange(n)[None,:]/n)
        dft=np.einsum('kn,hnd->hkd',phase,a);oracle=np.abs(dft).mean((0,2))
        error=float(np.max(np.abs(oracle-data['frequency'].numpy())));assert error<1e-5
        for ratio in (.1,.8,1.):
            ids,_=selection(x,attention,ratio);assert len(ids)==max(1,int(np.floor(n*ratio))) and ids==sorted(set(ids))
        same=torch.ones(8,n,16);equal=indicators(same,torch.ones(n))
        assert not equal['normalized_attention'].any()
        if n>1:assert float(equal['frequency'][0])>0 and float(equal['frequency'][1:].abs().max())<1e-5
        cases.append(dict(tokens=n,independent_DFT_max_error=error,literal_bin_to_token_association=True,constant_signal_DC_case=True))
    query=torch.tensor([1.,0.,0.]);keys={g:torch.tensor([[1.,0.,0.],[0.,1.,0.],[-1.,0.,0.]]) for g in ('segment','frame','patch')}
    order={g:[0,1,2] for g in keys};excluded={g:set() for g in keys};result=retrieve(query,keys,order,excluded)
    assert result['coarse_ids']==[0,1] and torch.equal(result['coarse_vector'],torch.tensor([.5,.5,0.]))
    assert all(ids==[0,1] for ids in result['selected'].values())
    for grain in keys:
        expected=keys[grain].numpy()@np.array([1.,0.,0.]);cons=(keys[grain].numpy()@np.array([.5,.5,0.]))/np.sqrt(.5)
        weight=0 if grain=='segment' else .3
        assert np.allclose(result['reranked_scores'][grain].numpy(),(1-weight)*expected+weight*cons,atol=1e-7,rtol=0)
    empty={g:torch.empty(0,3) for g in keys};none=retrieve(query,empty,{g:[] for g in keys},{g:set() for g in keys})
    assert all(not ids for ids in none['selected'].values()) and torch.equal(none['coarse_vector'],query)
    out=Path(__file__).resolve().parents[2]/'runs/20261007_m1_mukv/algorithm_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,cases=cases,coherence_independent_numpy_reference=True,empty_candidate_fallback=True,
        scope='isolated literalFFT/coherence arithmetic; not full3grain source/model/realprocessor/GPU/performance'),indent=2)+'\n');print('ALGORITHMS_PASS',flush=True)


if __name__=='__main__':main()
