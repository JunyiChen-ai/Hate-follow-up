"""Fresh single-model FP32 greedy source observations, with recorded input tokens."""
from pathlib import Path
import time
import math
import torch
from PIL import Image


def clock(j):
    if j.device.type=='cuda':torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def generate(j,root,system,content,paths,max_tokens):
    before=j.forward_calls;before_vision=getattr(j,'vision_calls',0);start=clock(j)
    rendered=j.render([j.turn('system',system),dict(role='user',content=content)],True)
    images=[Image.open(Path(root)/p).convert('RGB') for p in paths]
    try:enc=j.encode(rendered,images)
    finally:
        for image in images:image.close()
    if hasattr(j,'source_W32'):del j.source_W32
    j.model.model.rope_deltas=None
    output=j.model.model(**j.model_inputs(enc),use_cache=True)
    cache=output.past_key_values;hidden=output.last_hidden_state[0,-1].clone();del output
    j.source_W32=j.model.get_output_embeddings().weight.float()
    generated=[];stopped=False
    for _ in range(max_tokens):
        logits=hidden.float()@j.source_W32.T
        if j.softcap:logits=torch.tanh(logits/j.softcap)*j.softcap
        next_token=int(logits.argmax())
        if next_token in j.eos_ids:stopped=True;break
        generated.append(next_token);hidden=j._step(cache,[next_token]).clone()
    del cache,hidden
    forwards=j.forward_calls-before;assert forwards==1+len(generated)
    return dict(text=j.tok.decode(generated,skip_special_tokens=True).strip(),tokens=generated,
        truncated=not stopped,seconds=clock(j)-start,actual_forwards=forwards,
        actual_vision_forwards=getattr(j,'vision_calls',0)-before_vision,
        system=system,prompt=rendered,max_tokens=max_tokens,image_paths=paths,
        input_tokens=enc['input_ids'][0].tolist(),
        image_grid=enc['image_grid_thw'].tolist() if 'image_grid_thw' in enc else [])


def validate_generation(j,root,g,system,content,paths,max_tokens):
    rendered=j.render([j.turn('system',system),dict(role='user',content=content)],True)
    assert (g['system'],g['prompt'],g['max_tokens'],g['image_paths'])==(system,rendered,max_tokens,paths)
    images=[Image.open(Path(root)/p).convert('RGB') for p in paths]
    try:enc=j.encode(rendered,images)
    finally:
        for image in images:image.close()
    assert g['input_tokens']==enc['input_ids'][0].tolist()
    assert g['image_grid']==(enc['image_grid_thw'].tolist() if 'image_grid_thw' in enc else [])
    assert all(type(t) is int and 0<=t<len(j.tok) for t in g['tokens'])
    assert g['text']==j.tok.decode(g['tokens'],skip_special_tokens=True).strip()
    assert type(g['truncated']) is bool and len(g['tokens'])<=max_tokens
    assert g['truncated']==(len(g['tokens'])==max_tokens)
    assert g['actual_forwards']==1+len(g['tokens']) and math.isfinite(g['seconds']) and g['seconds']>=0
    assert g['actual_vision_forwards']==int(bool(paths))
