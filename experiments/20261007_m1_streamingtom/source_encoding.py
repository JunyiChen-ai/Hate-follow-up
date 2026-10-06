"""Actual timestamped source vision and source-bound sparse Qwen LM inputs."""
from PIL import Image
import torch
from inputs import ROOT,SPEC
from src.sparse_visual_prefix import pack
from src.stance_cache import positions
from vision import extract
from ctr import group,aggregate


def encode(j,ctx,frame):
    message=dict(role='user',content=[dict(type='text',text=SPEC['source_time_text'].format(time=frame['time'])),dict(type='image',image=frame['path'])])
    full=j.render(ctx['msgs']+ctx['history']+[message],False);assert full.startswith(ctx['head']);text=full[len(ctx['head']):]
    with Image.open(ROOT/frame['path']) as original:
        image=original.convert('RGB')
        try:encoded=j.encode(text,[image])
        finally:image.close()
    relative,_=positions(j,encoded['input_ids'].to(j.device),encoded['image_grid_thw'].to(j.device))
    rows=(encoded['input_ids'][0]==j.image_token_id).nonzero().flatten().tolist()
    assert len(encoded['image_grid_thw'])==1 and rows
    evidence=dict(message=message,suffix_text=text,suffix_ids=encoded['input_ids'][0].tolist(),image_grid=encoded['image_grid_thw'].tolist(),
        visual_rows=rows,relative_positions=relative[:,0].tolist())
    return encoded,evidence


def reduce(j,encoded,previous,previous_grid):
    captured=extract(j,encoded);assert len(captured['grid'])==1
    plan=group(captured['features'],captured['saliency'],captured['grid'][0],previous,previous_grid)
    feature=aggregate(captured['features'],plan);deep=[aggregate(v,plan) for v in captured['deepstack']]
    packed,evidence=pack(j,{'encoded':encoded,'deepstack':captured['deepstack']},[r['root'] for r in plan['groups']],feature,deep)
    assert torch.equal(packed['inputs_embeds'][packed['visual_pos_masks']],feature.to(j.device,j.dtype))
    assert all(torch.equal(a,b.to(j.device,j.dtype)) for a,b in zip(packed['deepstack_visual_embeds'],deep))
    proof=dict(plan=plan,packing=evidence,compressed_features=feature,compressed_deepstack=deep,
        saliency=captured['saliency'],vision_seconds=captured['seconds'],saliency_seconds=captured['saliency_seconds'])
    return packed,captured,proof
