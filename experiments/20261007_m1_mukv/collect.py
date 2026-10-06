"""Full three-grain source histories, explicit six-prefill accounting and DCP."""
import time
import torch
from inputs import SPEC
from memory import save_tensor
from encoding import acquire_features,packed_source
from source_attention import factory
from compression import selection
from src.pre_rotary_memory import attention_scope


def tick(j):
    if j.device.type=='cuda':torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def collect(j,native,ctx,memory,window_frames,proof_folder):
    start=tick(j);first=j.forward_calls;vision=j.vision_calls;records=[];feature_records=[];cached={};saved=j.model.model.rope_deltas.clone()
    times=dict(vision=0.,input_pack=0.,source_LM_attention=0.,FFT_selection=0.,storage_proof=0.)
    try:
        for window,frames in enumerate(window_frames):
            if not frames:continue
            for f in frames:
                stamp=tick(j);c=acquire_features(j,ctx,f);cached[f['index']]=c;times['vision']+=tick(j)-stamp
                stamp=tick(j);folder=proof_folder/'vision'/str(f['index']);folder.mkdir(parents=True,exist_ok=True)
                ff=save_tensor(folder/'full_projector.npy',c['feature']);dd=[save_tensor(folder/f'full_deepstack_{l}.npy',v) for l,v in enumerate(c['deepstack'])]
                feature_records.append(dict(frame=f,feature=ff,deepstack=dd,input=c['input'],grid=c['grid']));times['storage_proof']+=tick(j)-stamp
            middle=[frames[(len(frames)-1)//2]]
            blocks=[('segment',None,frames),('frame',None,middle)]+[('patch',q,middle) for q in range(4)]
            for grain,quadrant,media in blocks:
                stamp=tick(j);packed,relative,evidence=packed_source(j,ctx,media,grain,cached,quadrant);times['input_pack']+=tick(j)-stamp
                if packed is None:continue
                history=memory.earlier(window,grain);capture=dict(keys={},values={});make,p,end=factory(j,native,ctx,memory,history,relative,evidence['visual_rows'],capture)
                stamp=tick(j)
                with attention_scope(j,make):
                    out=j.model.model.language_model(**packed,position_ids=p,past_key_values=native,use_cache=True)
                    assert out.past_key_values is native;del out
                times['source_LM_attention']+=tick(j)-stamp;stamp=tick(j)
                roots,indicator=selection(capture['last_rotated_keys'],capture['attention'],SPEC['retention'][grain])
                vis=evidence['visual_rows'];selected_visual={vis[i] for i in roots}
                keep=[i for i in range(len(evidence['packed_ids'])) if i not in vis or i in selected_visual]
                times['FFT_selection']+=tick(j)-stamp;stamp=tick(j)
                block=memory.append(window,grain,quadrant,media,p,vis,capture['keys'],capture['values'],keep,history)
                folder=proof_folder/'sources'/str(block['id']);folder.mkdir(parents=True,exist_ok=True)
                last=save_tensor(folder/'last_rotated_visual_keys.npy',capture['last_rotated_keys'])
                records.append(dict(id=block['id'],window=window,grain=grain,quadrant=quadrant,input=evidence,history=history,
                    logical_start=end,positions=p[:,0].tolist(),attention=capture['attention'].tolist(),last_rotated_visual_keys=last,
                    frequency=indicator['frequency'].tolist(),normalized_attention=indicator['normalized_attention'].tolist(),normalized_frequency=indicator['normalized_frequency'].tolist(),
                    scores=indicator['score'].tolist(),retained_visual_roots=roots,retained_sequence_rows=keep,KV_storage_bytes=block['storage']['bytes']))
                times['storage_proof']+=tick(j)-stamp;del capture,packed
            assert native.get_seq_length()==ctx['stance_cache_tokens']
        source_calls=len(records);vision_calls=sum(len(f) for f in window_frames)
        assert j.forward_calls-first==source_calls and j.vision_calls-vision==vision_calls
        return dict(records=records,vision_records=feature_records,actual_LM=source_calls,actual_vision=vision_calls,seconds=tick(j)-start,times=times,
            call_scope='up to6 sourceLM/window; each actual sourceframe vision once; allprior samegrain history included, quadrant siblings excluded; additional attention/FFT/I/O paid'),cached
    finally:j.model.model.rope_deltas=saved
