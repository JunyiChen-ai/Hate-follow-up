"""CPU-only native Judge rendering/encoding, without loading model weights."""
from src.mllm_judge import Judge,MODEL,FAMILY_IMAGE_KW


def cpu_renderer(model_id=MODEL):
    import torch
    from transformers import AutoProcessor,AutoConfig
    j=Judge.__new__(Judge);j.model_id=model_id
    j.processor=AutoProcessor.from_pretrained(model_id,local_files_only=True);j.tok=j.processor.tokenizer
    config=AutoConfig.from_pretrained(model_id,local_files_only=True)
    j.family=config.model_type;assert j.family=='qwen3_vl'
    j.img_kw=dict(FAMILY_IMAGE_KW[j.family]);j.list_content=False
    j.same_turn=False;j.loose_stance_seam=False;j.device=torch.device('cpu');j.dtype=torch.bfloat16
    j.image_token_id=getattr(config,'image_token_id',None)
    return j
