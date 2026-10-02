#!/usr/bin/env python3
"""No-GT actual-model tests of depth indexing, state preservation and scoring."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from contraster import DepthContraster,contrast_logits
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CLAUDE.md').is_file())


class TinyJudge:
    family='qwen3_vl';yes_ids=[3,4];no_ids=[5,6]
    def __init__(self,lm,head):
        self.lm=lm;self.head=head
        self.model=SimpleNamespace(model=SimpleNamespace(language_model=lm),get_output_embeddings=lambda:head)
    def _step(self,cache,ids):
        return self.lm(input_ids=torch.tensor([ids]),past_key_values=cache,use_cache=True).last_hidden_state[0,-1]


def main():
    torch.manual_seed(0);torch.set_num_threads(1);results=[]
    cfg=Qwen3VLTextConfig(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=6,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,max_position_embeddings=128,
        rope_scaling={'rope_type':'default','mrope_section':[2,3,3]})
    cfg._attn_implementation='sdpa'
    for dtype in (torch.float32,torch.bfloat16):
        lm=Qwen3VLTextModel(cfg).eval().to(dtype);head=torch.nn.Linear(64,128,bias=False).to(dtype)
        saved={k:v.clone() for k,v in lm.state_dict().items()};j=TinyJudge(lm,head);e=DepthContraster(j)
        with torch.no_grad():
            prefix=lm(input_ids=torch.tensor([[10,11,12,13,14,15]]),use_cache=True).past_key_values
            native_cache=copy.deepcopy(prefix);native=lm(input_ids=torch.tensor([[20,21,22]]),past_key_values=native_cache,use_cache=True,output_hidden_states=True)
            hooked_cache=copy.deepcopy(prefix);h=e.forward(hooked_cache,[20,21,22])
            assert torch.equal(h,native.last_hidden_state[0,-1])
            assert e.layers==[2,4]
            for n in e.layers:assert torch.equal(e.rows[n],native.hidden_states[n][0,-1])
            for a,b in zip(hooked_cache.layers,native_cache.layers):
                assert torch.equal(a.keys,b.keys) and torch.equal(a.values,b.values)
            states=torch.stack([lm.norm(native.hidden_states[n][0,-1]) for n in e.layers]+[h])
            expected=states.float()@head.weight.float().T
            z,d=e.readout(h)
            assert torch.equal(torch.tensor(d['label_logits']),expected[:,j.yes_ids+j.no_ids])
            lp=expected.double().log_softmax(-1);p=lp.exp();mix=(p[:-1]+p[-1:])/2
            reference=.5*((p[:-1]*(lp[:-1]-mix.log())).sum(-1)+(p[-1:]*(lp[-1:]-mix.log())).sum(-1))
            torch.testing.assert_close(torch.tensor(d['js']).double(),reference,atol=2e-7,rtol=1e-5)
            assert d['selected_index']==int(reference.argmax()) and d['logprob_equivalence_error']<2e-6
            assert torch.equal(h,j._step(copy.deepcopy(prefix),[20,21,22]))
            e.close();assert not e.handles
            assert torch.equal(h,j._step(copy.deepcopy(prefix),[20,21,22]))
        assert all(torch.equal(v,saved[k]) for k,v in lm.state_dict().items())
        results.append({'dtype':str(dtype),'captured_depth_matches_hidden_states':True,'native_hidden_and_KV_exact':True,
            'norm_once_and_full_projection_exact':True,'standard_JS_reference':True,'selected_layer':d['selected_layer'],
            'logprob_equivalence_error':d['logprob_equivalence_error'],'weights_and_hook_removal_exact':True})
    # Strongly separated and near-zero probabilities remain finite; ties choose first.
    x=torch.tensor([[1000.,-1000.,2.,3.],[1000.,-1000.,2.,3.],[-1000.,1000.,3.,2.]])
    z,d=contrast_logits(x,[2,4],[0,2],[1,3]);assert d['selected_layer']==2 and torch.isfinite(torch.tensor(z))
    assert max(d['js'])<=.693148 and min(d['js'])>=0
    path=ROOT/'runs/20261003_m1_contraster/selfcheck';path.mkdir(parents=True,exist_ok=True)
    (path/'invariance.json').write_text(json.dumps({'tests':results,'extreme_logits_and_tie':True,'GT_read':False},indent=2)+'\n')
    print(json.dumps(results,indent=2));print('SELFCHECK_DONE')

if __name__=='__main__':main()
