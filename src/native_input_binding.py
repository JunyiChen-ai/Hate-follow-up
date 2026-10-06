"""Readable model/runtime and actual native pixel-array bindings for new readers.

Promoted from the reviewed ReKV binding on2026-10-06; active ReKV keeps its
already-reviewed local implementation. No hashes or model revision digests.
"""
import av
import numpy as np
import torch
import transformers
from pathlib import Path
from src.mllm_judge import MODEL,VIDEO_QUESTION
from src.video_inputs import frame_paths
ROOT=Path(__file__).resolve().parents[1]


def runtime(j,spec,implementation):
    cfg=j.model.model.config;text=cfg.text_config;vision=cfg.vision_config
    return dict(model=MODEL,implementation=implementation,spec=spec,dtype=str(j.dtype),
        torch=torch.__version__,transformers=transformers.__version__,av=av.__version__,numpy=np.__version__,
        image_kwargs=getattr(j,'img_kw',{}),
        text_config={key:getattr(text,key,None) for key in ('hidden_size','intermediate_size','num_hidden_layers',
            'num_attention_heads','num_key_value_heads','head_dim','vocab_size','rms_norm_eps','rope_parameters','layer_types')},
        vision_config={key:getattr(vision,key,None) for key in ('hidden_size','depth','num_heads','patch_size',
            'temporal_patch_size','spatial_merge_size','out_hidden_size','deepstack_visual_indexes')})


def native(j,row,segments,ctx,frames):
    msgs,files=j.prefix_messages(frame_paths(row['dataset'],row['video_id'],frames),segments)
    text,encoded=j.encode_prefix(msgs,files)
    qids,qtext=j.branch_ids(msgs,VIDEO_QUESTION,head_text=text);aids,atext=j.answer_ids(msgs,VIDEO_QUESTION,ctx['stance'])
    assert msgs==ctx['msgs'] and text+qtext+atext==ctx['head'] and encoded['input_ids'].shape[1]==ctx['prefix_tokens']
    evidence=dict(prefix_ids=encoded['input_ids'][0].tolist(),image_grid=encoded['image_grid_thw'].tolist(),
        paths=[str(path.relative_to(ROOT)) for path in files],global_question_ids=qids,stance_ids=aids)
    pixels=encoded['pixel_values'].to('cpu').float().numpy();assert np.isfinite(pixels).all()
    return evidence,pixels


def save(j,row,segments,ctx,spec,implementation,path):
    evidence,pixels=native(j,row,segments,ctx,spec['native_frames']);np.save(path,pixels,allow_pickle=False)
    return dict(runtime=runtime(j,spec,implementation),native=evidence,pixel_path=str(path.relative_to(ROOT)),
        pixel_shape=list(pixels.shape),pixel_dtype='float32')


def validate(j,row,segments,ctx,spec,implementation,binding):
    assert binding['runtime']==runtime(j,spec,implementation),'completed record runtime/config changed'
    evidence,pixels=native(j,row,segments,ctx,spec['native_frames']);assert evidence==binding['native']
    saved=np.load(ROOT/binding['pixel_path'],allow_pickle=False)
    assert binding['pixel_dtype']=='float32' and saved.dtype==np.float32 and list(saved.shape)==binding['pixel_shape']==list(pixels.shape)
    assert np.array_equal(saved,pixels),'completed record native pixel input changed'
