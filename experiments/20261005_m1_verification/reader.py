"""Final fresh independent multimodal readings under the unchanged native stance."""
import copy
import math
from PIL import Image
import torch
from inputs import ROOT
from interface import SPEC, canonical, model_visible, branch_record
from src.mllm_judge import yesno_question
from src.source_generation import clock


def encoding(j, ctx, m, i, kind):
    w = m['windows'][i]
    record = branch_record(m, i, kind)
    content = [dict(type='text', text=SPEC['branch_header'] + canonical(model_visible(record)) + '\n')]
    paths = []
    for f in w['frames']:
        content.extend([dict(type='text', text=f'[LOCAL frame={f["id"]}; actual_t={f["time"]:.6f}s; index={f["index"]}]\n'), dict(type='image')])
        paths.append(f['path'])
    if kind == 'speech':
        content.append(dict(type='text', text='Literal LOCAL speech:\n' + w['body'] + '\n'))
    content.append(dict(type='text', text=yesno_question(i, len(m['windows']), w['start'], w['end'], w['body'], kind)))
    messages = copy.deepcopy(ctx['msgs']) + copy.deepcopy(ctx['history']) + [dict(role='user', content=content)]
    text = j.render(messages, True)
    allpaths = [str(p.relative_to(ROOT)) for p in ctx['files']] + paths
    images = [Image.open(ROOT / p).convert('RGB') for p in allpaths]
    try:
        enc = j.encode(text, images)
    finally:
        for im in images:
            im.close()
    trace = dict(messages=messages, prompt=text, image_paths=allpaths, source_record=record, input_tokens=enc['input_ids'][0].tolist(), image_grid=enc['image_grid_thw'].tolist())
    assert trace['input_tokens'][:ctx['stance_cache_tokens']] == ctx['native_stance_ids']
    return enc, trace


@torch.no_grad()
def read(j, ctx, m, i, kind):
    before, before_vision = j.forward_calls, j.vision_calls
    start = clock(j)
    enc, trace = encoding(j, ctx, m, i, kind)
    j.model.model.rope_deltas = None
    out = j.model.model(**j.model_inputs(enc), use_cache=False)
    z = j.margins_fp32(out.last_hidden_state[0, -1:])[0]
    del out
    trace.update(margin=z, seconds=clock(j) - start, actual_forwards=j.forward_calls - before, actual_vision_forwards=j.vision_calls - before_vision)
    assert trace['actual_forwards'] == trace['actual_vision_forwards'] == 1
    return z, trace


def validate_trace(j, ctx, m, i, kind, trace):
    _, expected = encoding(j, ctx, m, i, kind)
    assert all(trace[k] == v for k, v in expected.items())
    assert trace['actual_forwards'] == trace['actual_vision_forwards'] == 1 and math.isfinite(trace['margin'])
    assert math.isfinite(trace['seconds']) and trace['seconds'] >= 0
