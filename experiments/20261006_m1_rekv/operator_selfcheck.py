"""Independent arithmetic references for GQA retrieval and Qwen position packing."""
import json
import numpy as np
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb
from retrieval import ROOT, mean_query, mean_image_key, remote_selection
from layout import pack_positions, rotate_key, suffix_positions, causal_mask


def main():
    torch.set_num_threads(4)
    generator = torch.Generator().manual_seed(0)
    cases = []
    for kv in (1, 2, 8):
        q = torch.randn(1, kv * 4, 7, 16, generator=generator)
        rows = [0, 2, 6]
        result = mean_query(q, rows, kv).numpy()
        raw = q.numpy()
        expected = np.concatenate([
            np.mean([raw[0, h, r] for h in range(g * 4, (g + 1) * 4) for r in rows], axis=0)
            for g in range(kv)])
        assert np.allclose(result, expected, atol=2e-7, rtol=0)
        key = torch.randn(1, kv, 7, 16, generator=generator)
        expected_key = np.concatenate([np.mean([key.numpy()[0, h, r] for r in rows], axis=0) for h in range(kv)])
        assert np.allclose(mean_image_key(key, rows).numpy(), expected_key, atol=2e-7, rtol=0)
        source = [2, 5, 8, 10, 17, 22]
        representatives = torch.ones(6, kv * 16)
        for query in (torch.ones(kv * 16), torch.zeros(kv * 16)):
            chosen, scores = remote_selection(query, representatives, source, [1, 4], budget=2)
            assert chosen == [0, 2] and torch.isfinite(scores).all()
        none, _ = remote_selection(result := torch.from_numpy(result), representatives, source, list(range(6)))
        assert none == []
        cases.append(dict(kind='GQA/image-only independent means, zero-norm and stable remote ties',kv_heads=kv,q_heads=kv*4))
    cfg = Qwen3VLTextConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=36,
        num_attention_heads=4, num_key_value_heads=1, head_dim=16,
        rope_parameters={'rope_type':'default','rope_theta':5000000.,'mrope_section':[4,2,2],'mrope_interleaved':True})
    rotary = Qwen3VLTextRotaryEmbedding(cfg)
    original = torch.stack([torch.arange(9)+10,torch.arange(9)+13,torch.arange(9)+17])[:,None]
    packed, end = pack_positions([original, original+50], 101)
    for old, new in zip([original,original+50],packed):
        difference = new-old
        assert len(np.unique(difference.numpy())) == 1
        assert torch.equal(new[1]-new[0],old[1]-old[0]) and torch.equal(new[2]-new[0],old[2]-old[0])
    assert int(packed[0].min()) == 101 and int(packed[1].min()) == int(packed[0].max())+1
    assert int(suffix_positions(5,end,'cpu').min()) == end
    for dtype in (torch.float32,torch.bfloat16):
        key = torch.randn(1,1,9,16,generator=generator).to(dtype)
        for position in (original, packed[0]):
            cos,sin = rotary(key,position)
            _,reference = apply_rotary_pos_emb(key,key,cos,sin)
            assert torch.equal(rotate_key(key,position,rotary),reference)
        cases.append(dict(kind='actual Qwen interleaved3axis identity/translation operator',dtype=str(dtype)))
    for past in (0,5,333):
        for length in (1,7,31):
            mask = causal_mask(length,past,'cpu')
            expected = np.asarray([[column<=past+row for column in range(past+length)] for row in range(length)])
            assert np.array_equal(mask[0,0].numpy(),expected)
            assert mask.shape == (1,1,length,past+length)
    cases.append(dict(kind='independent rectangular prefix+suffix causality',cases=9))
    out=ROOT/'runs/20261006_m1_rekv/operator_cpu_checks'
    out.mkdir(parents=True,exist_ok=True)
    summary=dict(PASS=True,scope='software arithmetic only; not source-image/36-layer forward/GPU/pretrained/GT/performance evidence',cases=cases)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
