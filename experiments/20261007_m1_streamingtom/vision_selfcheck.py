"""Exact unchanged vision and independent FP64 saliency oracle; random CPU."""
import json
import socket
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel,apply_rotary_pos_emb_vision
from vision import extract


def main():
    torch.set_num_threads(4);torch.manual_seed(0);print(socket.gethostname(),flush=True);cases=[]
    cfg=Qwen3VLConfig(text_config=dict(vocab_size=512,hidden_size=64,intermediate_size=128,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2),
        vision_config=dict(depth=3,hidden_size=32,intermediate_size=64,num_heads=4,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,
            out_hidden_size=64,num_position_embeddings=16,deepstack_visual_indexes=[0,1,2]),image_token_id=127,video_token_id=124,vision_start_token_id=125,vision_end_token_id=126)
    cfg._attn_implementation=cfg.text_config._attn_implementation=cfg.vision_config._attn_implementation='sdpa'
    with torch.no_grad():
        for dtype in (torch.float32,torch.bfloat16):
            model=Qwen3VLModel(cfg).eval().to(dtype);j=SimpleNamespace(model=SimpleNamespace(model=model),device=torch.device('cpu'))
            for width in (8,24):
                grid=torch.tensor([[1,width,width]]);pixels=torch.randn(width*width,24).to(dtype)
                encoded=dict(pixel_values=pixels,image_grid_thw=grid);actual=extract(j,encoded)
                state={};att=model.visual.blocks[-1].attn
                def before(module,args,kwargs):state.update(cu=kwargs['cu_seqlens'],pos=kwargs['position_embeddings'])
                def qkv(module,args,out):state['qkv']=out.detach().clone()
                hooks=[att.register_forward_pre_hook(before,with_kwargs=True),att.qkv.register_forward_hook(qkv)]
                try:original=model.get_image_features(pixels,grid,return_dict=True)
                finally:
                    for hook in hooks:hook.remove()
                assert torch.equal(actual['features'],torch.cat(tuple(original.pooler_output)))
                assert all(torch.equal(a,b) for a,b in zip(actual['deepstack'],original.deepstack_features))
                q,k,_=state['qkv'].reshape(width*width,3,att.num_heads,-1).permute(1,0,2,3).unbind(0)
                q,k=apply_rotary_pos_emb_vision(q,k,*state['pos']);q=q.double().transpose(0,1);k=k.double().transpose(0,1)
                oracle=torch.softmax((q@k.transpose(1,2))*att.scaling,-1).mean((0,1)).reshape(-1,4).sum(1)
                difference=float((actual['saliency'].double()-oracle).abs().max());assert difference<2e-6
                assert abs(float(actual['saliency'].sum())-1)<2e-6
                assert actual['seconds']>=actual['saliency_seconds']>0
                cases.append(dict(dtype=str(dtype),raw_patches=width*width,merged_tokens=width*width//4,unchanged_projector_DeepStack_exact=True,
                    saliency_oracle_max_error=difference,chunked_attention_paid=True))
                print('VISION_CASE_PASS',dtype,width,difference,flush=True)
    out=Path(__file__).resolve().parents[2]/'runs/20261007_m1_streamingtom/vision_cpu_checks';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(PASS=True,GT_read=False,pretrained=False,scope='isolated actual Qwen vision/DeepStack observer and independent dense FP64 attention oracle; not full method',cases=cases),indent=2)+'\n')
    print('VISION_CPU_PASS',flush=True)


if __name__=='__main__':main()
