"""Actual multigrain image message and source-preserving prefill layout."""
from PIL import Image
import torch
from inputs import ROOT,SPEC,quadrants
from src.stance_cache import positions
from src.cached_image_features import extract,reuse
from src.sparse_visual_prefix import pack


def source_input(j,ctx,frames,grain):
    times=[f['time'] for f in frames];content=[dict(type='text',text=SPEC['source_time_text'].format(times=times,grain=grain))]
    for frame in frames:content.append(dict(type='image',image=frame['path']))
    message=dict(role='user',content=content);full=j.render(ctx['msgs']+ctx['history']+[message],False)
    assert full.startswith(ctx['head']);text=full[len(ctx['head']):];images=[]
    try:
        for f in frames:
            with Image.open(ROOT/f['path']) as image:images.append(image.convert('RGB'))
        encoded=j.encode(text,images)
    finally:
        for image in images:image.close()
    p,_=positions(j,encoded['input_ids'].to(j.device),encoded['image_grid_thw'].to(j.device))
    evidence=dict(message=message,suffix_text=text,suffix_ids=encoded['input_ids'][0].tolist(),grid=encoded['image_grid_thw'].tolist(),positions=p[:,0].tolist())
    return encoded,p,evidence


def acquire_features(j,ctx,frame):
    encoded,p,evidence=source_input(j,ctx,[frame],'frame');f,ds=extract(j,encoded)
    assert len(f)==len(ds)==1
    return dict(feature=f[0],deepstack=ds[0],grid=encoded['image_grid_thw'][0].tolist(),input=evidence)


def source_geometry(j,ctx,frames,grain,quadrant=None):
    encoded,p,evidence=source_input(j,ctx,frames,grain)
    original=encoded['input_ids'][0];visual_full=(original==j.image_token_id).nonzero().flatten().tolist()
    if quadrant is None:roots=list(range(len(visual_full)))
    else:
        assert grain=='patch' and len(frames)==1 and 0<=quadrant<4
        roots=quadrants(p,visual_full)[quadrant]
        if not roots:return None,None,None
    keep=original!=j.image_token_id;keep[torch.tensor(visual_full)[roots]]=True;sequence=keep.nonzero().flatten()
    selected=original[sequence];position=p[:,:,sequence]
    visual=(selected==j.image_token_id).nonzero().flatten().tolist()
    evidence.update(grain=grain,quadrant=quadrant,visual_roots=roots,sequence_rows=sequence.tolist(),visual_rows=visual,packed_ids=selected.tolist(),packed_positions=position[:,0].tolist())
    return encoded,position,evidence


def packed_source(j,ctx,frames,grain,cached,quadrant=None):
    encoded,p,evidence=source_geometry(j,ctx,frames,grain,quadrant)
    if encoded is None:return None,None,None
    features=[cached[f['index']]['feature'] for f in frames];deep=[cached[f['index']]['deepstack'] for f in frames]
    counts=[len(v) for v in features]
    assert evidence['grid']==[cached[f['index']]['grid'] for f in frames]
    if quadrant is None:
        packed,_=reuse(j,encoded,features,deep)
    else:
        assert grain=='patch' and len(frames)==1 and 0<=quadrant<4
        roots=evidence['visual_roots']
        full=torch.cat(features);all_ds=[torch.cat([d[l] for d in deep]) for l in range(len(deep[0]))]
        packed,layout=pack(j,dict(encoded=encoded,deepstack=all_ds),roots,full[roots],[v[roots] for v in all_ds])
        assert layout['sequence_indices']==evidence['sequence_rows'] and packed['position_ids'][:,0].tolist()==evidence['packed_positions']
    packed={k:v for k,v in packed.items() if k not in ('attention_mask','position_ids')}
    return packed,p,evidence
