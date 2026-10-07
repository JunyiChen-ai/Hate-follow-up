"""CPU fixture check of the R1 controls against the production R1 reader; no annotations."""
import json
import socket
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from inputs import ROOT,SPEC,windows_for
from cpu_fixture import fixture
from measure import read_video
from validate import validate_bundle
from controls import read,uniform_group,nearest
from src.pre_rotary_memory import select


def untimed(p):
    return {**p,'extra':{k:v for k,v in p['extra'].items() if k!='standalone_seconds'}}


def main():
    torch.set_num_threads(4);torch.manual_seed(0);print(socket.gethostname(),flush=True)
    out=ROOT/'runs/20261007_m1_streamingtom/controls_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    for n in (1,2,49,50,51,64,90,196,320):
        plan=uniform_group(torch.zeros(n,4),torch.zeros(n),[1,2,2]);roots=[g['root'] for g in plan['groups']]
        assert len(roots)==min(50,n) and roots==sorted(set(roots)) and roots[0]>=0 and roots[-1]<n
    native=out/'native.png';Image.fromarray(np.full((8,8,3),100,np.uint8)).save(native);rng=np.random.default_rng(0);frames=[]
    for i in range(7):
        if i%2==0:image=rng.integers(0,256,(32,32,3),dtype=np.uint8)
        path=out/f'source_{i}.png';Image.fromarray(image).save(path);frames.append(dict(index=16*i,time=2.*i,path=str(path.relative_to(ROOT))))
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=512,hidden_size=64,intermediate_size=128,num_hidden_layers=36,num_attention_heads=32,num_key_value_heads=8,head_dim=128,
        max_position_embeddings=4096,rope_parameters={'rope_type':'default','rope_theta':5000000.,'mrope_section':[24,20,20],'mrope_interleaved':True}),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,
            out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa'
    row=dict(dataset='HateMM',video_id='fixture',duration=24.);segments=[(0.,8.,'Fixture speech'),(8.,16.,'Later speech')]
    source=dict(decode_seconds=0.,fixture=True);cases=[]
    with torch.no_grad():
        for dtype,count in ((torch.float32,18),(torch.bfloat16,20)):
            model=Qwen3VLModel(cfg).eval().to(dtype);model.requires_grad_(False);j,ff,hooks=fixture(model,count,dtype,native);opened=[]
            folder=out/(str(dtype).split('.')[-1]+str(count))
            try:
                with (patch('measure.frames_for',return_value=(source,frames)),patch('validate.frames_for',return_value=(source,frames)),
                    patch('controls.frames_for',return_value=(source,frames)),patch('measure.frame_paths',return_value=ff),
                    patch('controls.frame_paths',return_value=ff),patch('src.native_input_binding.frame_paths',return_value=ff)):
                    # Production R1 (with the default-preserving hooks) still passes its own strict replay.
                    reference,memory=read_video(j,row,segments,folder/'r1',True);opened.append(memory)
                    validate_bundle(j,row,segments,reference,True);memory.close(release=True);opened.remove(memory)
                    windows=windows_for(row,segments)
                    dual,memory=read(j,row,segments,folder/'dualpath','dualpath',reference);opened.append(memory);memory.close(release=True);opened.remove(memory)
                    c=dual['comparison'];assert c['features_exact'] and c['deepstack_exact'] and c['saliency_exact'] and c['plan_exact'] and c['representatives_exact'] and c['replay_queries_exact'],c
                    assert untimed(dual['base'])==untimed(reference['base']),'native reads must be identical'
                    differs=0
                    for t,r,w in zip(dual['traces'],reference['traces'],windows):
                        if not t['LOCAL']:
                            assert r['branch'] is None and all(v is None for v in t['arms'].values());continue
                        replay=t['arms']['replay'];assert replay['z']==r['new_visual'],'replay must equal R1'
                        assert replay['remote_ids']==[l['remote_ids'] for l in r['branch']['layers']]
                        assert all(ids==[] for ids in t['arms']['no_remote']['remote_ids']) and set(t['arms']['no_remote']['source_tokens'])=={0}
                        assert t['nearest_ids']==nearest(frames,t['LOCAL'],w) and all(ids==t['nearest_ids'] for ids in t['arms']['nearest']['remote_ids'])
                        differs+=t['arms']['no_remote']['z']!=replay['z']
                    assert differs>0,'removing remote blocks must change the fixture read'
                    assert [p['score_curve'] for p in (dual['predictions']['replay'],)]==[reference['optimized']['score_curve']]
                    uni,memory=read(j,row,segments,folder/'uniform','uniform',reference);opened.append(memory);memory.close(release=True);opened.remove(memory)
                    c=uni['comparison'];assert c['features_exact'] and c['deepstack_exact'] and c['saliency_exact'],c
                    assert untimed(uni['base'])==untimed(reference['base'])
                    for p in uni['plans']:
                        n=p['original_tokens'];g=min(50,n);assert p['roots']==[(2*k+1)*n//(2*g) for k in range(g)] and p['static_budget']==0
                    reps=np.load(ROOT/uni['vectors']['source_keys']);queries=np.load(ROOT/uni['vectors']['question_vectors'])
                    for t in uni['traces']:
                        if not t['LOCAL']:continue
                        for l,ids in enumerate(t['arms']['uniform']['remote_ids']):
                            chosen,_=select(torch.from_numpy(queries[t['i'],l]),torch.from_numpy(reps[l]),[f['index'] for f in frames],set(t['LOCAL']),SPEC['remote_frames_per_layer'])
                            assert ids==chosen
                    assert any(a['z']!=b['arms']['replay']['z'] for a,b in zip([t['arms']['uniform'] for t in uni['traces'] if t['LOCAL']],[t for t in dual['traces'] if t['LOCAL']]))
                    (folder/'controls.json').write_text(json.dumps(dict(dualpath=dual,uniform=uni))+'\n')
                    cases.append(dict(dtype=str(dtype),native_frames=count,r1_strict_replay_with_hooks=True,replay_equals_r1=True,no_remote_empty=True,
                        nearest_rule=True,uniform_rule=True,uniform_retrieval_recomputed=True,rebuilt_features_exact=True))
                    print('CONTROLS_CASE_PASS',dtype,count,flush=True)
            finally:
                for memory in opened:memory.close(release=True)
                for hook in hooks:hook.remove()
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,pretrained=False,cases=cases,
        scope='random-weight 36-layer fixture; production R1 read_video/validate and controls.read; not real 8B/semantic/performance evidence'),indent=2)+'\n')
    print('CONTROLS_CPU_PASS',flush=True)


if __name__=='__main__':main()
