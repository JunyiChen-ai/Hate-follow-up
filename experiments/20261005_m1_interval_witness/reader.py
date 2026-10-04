"""Fresh global/own-stance and independent full multimodal source-path readings."""
import copy
import math
from PIL import Image
import torch
from inputs import ROOT
from interface import SPEC,canonical,model_visible,source_path
from src.mllm_judge import VIDEO_QUESTION,yesno_question
from src.stance_cache import positions
from src.source_generation import clock


@torch.no_grad()
def build_tree(j,frames,segments,m):
    j.model.model.rope_deltas=None
    native,paths=j.prefix_messages(frames,segments)
    messages=j._append_text(native,'\n'+SPEC['global_header']+canonical(m['final_tree'])+'\n')
    text,enc=j.encode_prefix(messages,paths);cache=j.prefix_cache(enc);qid,qtext=j.branch_ids(messages,VIDEO_QUESTION)
    z=j.cached_margin(cache,qid,in_place=True);stance='Yes' if z>0 else 'No'
    aid,atext=j.answer_ids(messages,VIDEO_QUESTION,stance);j.extend_cache(cache,aid)
    history=[dict(role='user',content=[dict(type='text',text=VIDEO_QUESTION)]),j.turn('assistant',stance)]
    ids=enc['input_ids'][0].tolist()+qid+aid
    p,delta=positions(j,torch.tensor([ids],device=j.device),enc['image_grid_thw'].to(j.device))
    assert len(ids)==cache.get_seq_length() and torch.equal(delta,j.model.model.rope_deltas)
    ctx=dict(msgs=messages,files=paths,history=history,head=text+qtext+atext,native_stance_ids=ids,
        global_margin=z,stance=stance,stance_cache_tokens=len(ids),stance_cache_logical_start=int(p.max())+1,
        prefix_tokens=len(enc['input_ids'][0]),image_counts=list(j.img_tokens))
    del cache
    return ctx


def binding(j,frames,segments,m,stance):
    native,paths=j.prefix_messages(frames,segments);messages=j._append_text(native,'\n'+SPEC['global_header']+canonical(m['final_tree'])+'\n')
    text,enc=j.encode_prefix(messages,paths);qid,qtext=j.branch_ids(messages,VIDEO_QUESTION);aid,atext=j.answer_ids(messages,VIDEO_QUESTION,stance)
    return dict(msgs=messages,files=paths,history=[dict(role='user',content=[dict(type='text',text=VIDEO_QUESTION)]),j.turn('assistant',stance)],
        head=text+qtext+atext,native_stance_ids=enc['input_ids'][0].tolist()+qid+aid,stance_cache_tokens=len(enc['input_ids'][0])+len(qid)+len(aid),prefix_tokens=len(enc['input_ids'][0]))


def encoding(j,ctx,m,i,kind):
    w=m['windows'][i];record=source_path(m,i,kind)
    content=[dict(type='text',text=SPEC['branch_header']+canonical(model_visible(record))+'\n')];paths=[]
    for f in w['frames']:
        content.extend([dict(type='text',text=f'[LOCAL frame={f["id"]}; actual_t={f["time"]:.6f}s; index={f["index"]}]\n'),dict(type='image')]);paths.append(f['path'])
    if kind=='speech':content.append(dict(type='text',text='Literal LOCAL speech:\n'+w['body']+'\n'))
    content.append(dict(type='text',text=yesno_question(i,len(m['windows']),w['start'],w['end'],w['body'],kind)))
    messages=copy.deepcopy(ctx['msgs'])+copy.deepcopy(ctx['history'])+[dict(role='user',content=content)]
    text=j.render(messages,True);allpaths=[str(p.relative_to(ROOT)) for p in ctx['files']]+paths
    images=[Image.open(ROOT/p).convert('RGB') for p in allpaths]
    try:enc=j.encode(text,images)
    finally:
        for image in images:image.close()
    trace=dict(messages=messages,prompt=text,image_paths=allpaths,source_path=record,input_tokens=enc['input_ids'][0].tolist(),image_grid=enc['image_grid_thw'].tolist())
    assert trace['input_tokens'][:ctx['stance_cache_tokens']]==ctx['native_stance_ids'];return enc,trace


@torch.no_grad()
def read(j,ctx,m,i,kind):
    before=j.forward_calls;before_vision=j.vision_calls;start=clock(j);enc,trace=encoding(j,ctx,m,i,kind)
    j.model.model.rope_deltas=None;out=j.model.model(**j.model_inputs(enc),use_cache=False)
    z=j.margins_fp32(out.last_hidden_state[0,-1:])[0];del out
    trace.update(margin=z,seconds=clock(j)-start,actual_forwards=j.forward_calls-before,actual_vision_forwards=j.vision_calls-before_vision)
    assert trace['actual_forwards']==trace['actual_vision_forwards']==1
    return z,trace


def validate_trace(j,ctx,m,i,kind,trace):
    _,expected=encoding(j,ctx,m,i,kind);assert all(trace[k]==v for k,v in expected.items())
    assert trace['actual_forwards']==trace['actual_vision_forwards']==1 and math.isfinite(trace['margin'])
    assert math.isfinite(trace['seconds']) and trace['seconds']>=0
