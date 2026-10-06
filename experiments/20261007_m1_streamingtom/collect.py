"""Fresh vision -> dual-path sparse source -> quantized frame-bound LM memory."""
import time
import torch
from transformers.cache_utils import DynamicCache
from inputs import SPEC
from source_encoding import encode,reduce
from src.pre_rotary_memory import translate,rotate,observe


def tick(j):
    if j.device.type=='cuda':torch.cuda.synchronize()
    return time.perf_counter()


def context_cache(j,native,ctx,memory,history):
    packed,end=translate([memory.blocks[i]['positions'] for i in history],ctx['stance_cache_logical_start'])
    cache=DynamicCache();rotary=j.model.model.language_model.rotary_emb
    for layer,prefix in enumerate(native.layers):
        keys=[prefix.keys];values=[prefix.values]
        for i,p in zip(history,packed):
            k,v,_=memory.layer(i,layer,j.device);keys.append(rotate(k,p,rotary));values.append(v)
        cache.update(torch.cat(keys,2),torch.cat(values,2),layer)
    return cache,end


@torch.no_grad()
def collect(j,native,ctx,memory,frames,proof_folder):
    start=tick(j);records=[];previous=None;previous_grid=None;saved=j.model.model.rope_deltas.clone()
    first=j.forward_calls;vision=j.vision_calls;parts=dict(encoding=0.,vision=0.,saliency=0.,grouping_pack=0.,history_dequant=0.,source_LM=0.,storage_proof=0.)
    try:
        for i,frame in enumerate(frames):
            stamp=tick(j);encoded,input_evidence=encode(j,ctx,frame);parts['encoding']+=tick(j)-stamp
            stamp=tick(j);packed,captured,proof=reduce(j,encoded,previous,previous_grid)
            elapsed=tick(j)-stamp;parts['vision']+=proof['vision_seconds'];parts['saliency']+=proof['saliency_seconds']
            parts['grouping_pack']+=max(0.,elapsed-proof['vision_seconds'])
            history=list(range(max(0,i-SPEC['source_previous_frames']),i));stamp=tick(j)
            cache,end=context_cache(j,native,ctx,memory,history);parts['history_dequant']+=tick(j)-stamp
            pos=packed['position_ids']+end;kw={k:v for k,v in packed.items() if k not in ('position_ids','attention_mask')}
            stamp=tick(j)
            with observe(j) as (keys,values):
                out=j.model.model.language_model(**kw,position_ids=pos,past_key_values=cache,use_cache=True)
                assert out.past_key_values is cache;del out
            parts['source_LM']+=tick(j)-stamp;stamp=tick(j)
            visual=packed['visual_pos_masks'][0].nonzero().flatten().tolist()
            block=memory.append(frame,pos,visual,keys,values,history,captured['features'],captured['deepstack'])
            # Persistent compressed observations are output proofs, not inputs
            # to future native/global/speech branches or free per-video work.
            from memory import save_tensor
            folder=proof_folder/str(i);folder.mkdir(parents=True,exist_ok=True)
            feature=save_tensor(folder/'compressed_features.npy',proof['compressed_features'])
            original_feature=save_tensor(folder/'full_projector.npy',captured['features'])
            original_ds=[save_tensor(folder/f'full_deepstack_{l}.npy',v) for l,v in enumerate(captured['deepstack'])]
            ds=[save_tensor(folder/f'compressed_deepstack_{l}.npy',v) for l,v in enumerate(proof['compressed_deepstack'])]
            records.append(dict(block_id=i,history_ids=history,physical_start=ctx['stance_cache_tokens']+sum(memory.blocks[h]['shape'][-2] for h in history),
                logical_start=end,suffix_tokens=packed['inputs_embeds'].shape[1],input=input_evidence,plan=proof['plan'],packing=proof['packing'],
                full_projector=original_feature,full_deepstack=original_ds,compressed_features=feature,compressed_deepstack=ds,
                saliency=proof['saliency'].tolist(),temporary_KV_bytes=block['storage_bytes']))
            parts['storage_proof']+=tick(j)-stamp
            previous=captured['features'];previous_grid=captured['grid'][0]
            del captured,proof,packed,encoded,cache,keys,values
        assert native.get_seq_length()==ctx['stance_cache_tokens']
        assert j.forward_calls-first==len(frames) and j.vision_calls-vision==len(frames)
        return dict(records=records,actual_forwards=len(frames),actual_vision=len(frames),seconds=tick(j)-start,parts=parts,
            timing_scope='vision includes saliency; saliency is nested, not added again; actual full source wall includes I/O')
    finally:j.model.model.rope_deltas=saved
