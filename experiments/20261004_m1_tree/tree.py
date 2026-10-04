"""Real-frame semantic breadth/depth acquisition, without annotation input."""
import copy
import math
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans

CACHE_VERSION='R1 Qwen-only semantic breadth-depth tree, sources 2026-10-04'
CONSTANTS=dict(pool_fps=1,coverage_windows=8,feature_batch=8,root_widths=[4,8,16],
    high_roots=2,branch_width=2,max_depth_below_root=2,seed=0,n_init=10,
    max_iter=300,tol=1e-4,caption_tokens=96,local_images=2,norm_floor=1e-12)
CAP_SYSTEM='Describe observable visual evidence accurately.'
CAP_QUESTION=('Describe the visible people, actions, gestures, symbols, and any legible '
    'on-screen text in this frame. Do not infer information outside the image. Use at most two sentences.')
TREE_HEADER='Tree observations (frame descriptions, not independent moderation decisions):\n'
CONTEXT_HEADER=('Interpretation context for the selected in-window images. Context descriptions '
    'may show frames outside this window; their times remain explicit. They do not establish '
    'an occurrence inside the window:\n')
LOCAL_HEADER='Local observed frames:\n'


def resolve_video(row):
    given=Path(row['video_path'])
    choices=[given]+[p for d in ('video','videos') for p in sorted(
        (Path.home()/'data'/row['dataset']/d).glob(row['video_id']+'.*'))]
    p=next((p for p in choices if p.is_file()),None)
    if p is None:raise FileNotFoundError((row['dataset'],row['video_id']))
    return p


def pixel_check(enc):
    from src.mllm_judge import MAX_PIXELS
    assert 'image_grid_thw' in enc
    assert all(int(h*w)*16**2<=MAX_PIXELS for t,h,w in enc['image_grid_thw'].tolist())


@torch.no_grad()
def image_features(j,images):
    enc=j.processor.image_processor(images=images,return_tensors='pt',**j.img_kw)
    pixel_check(enc)
    out=j.model.model.get_image_features(enc['pixel_values'].to(j.device,j.dtype),
        enc['image_grid_thw'].to(j.device))
    rows=out.pooler_output if hasattr(out,'pooler_output') else out[0]
    assert len(rows)==len(images)
    values=[]
    for row,grid in zip(rows,enc['image_grid_thw']):
        assert len(row)==int(grid.prod())//j.processor.image_processor.merge_size**2
        v=row.float().mean(0);v=v/v.norm().clamp_min(1e-12)
        values.append(v.cpu().numpy())
    return np.stack(values)


@torch.no_grad()
def decode_pool(j,row):
    import av
    from src.video_inputs import fixed_windows
    duration=float(row['duration']);windows=fixed_windows(duration,8)
    source=resolve_video(row);entries=[];features=[];batch=[];backups={};covered=set()
    targets=np.arange(.5,duration,1.);cursor=0;decoded=0;previous=-math.inf
    def add(frame,entry):
        entries.append(entry);batch.append(frame.to_image().convert('RGB'))
        if len(batch)==8:flush()
    def flush():
        if batch:
            features.extend(image_features(j,batch))
            for im in batch:im.close()
            batch.clear()
    with av.open(str(source)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        origin=container.start_time/av.time_base if container.start_time is not None else None
        for index,frame in enumerate(container.decode(stream)):
            decoded+=1
            if frame.pts is None:raise ValueError('source frame lacks actual PTS')
            absolute=float(frame.pts*frame.time_base)
            if origin is None:origin=absolute
            t=absolute-origin
            assert t>previous,('nonmonotone source PTS',source,index)
            previous=t
            if not 0<=t<duration:continue
            entry=dict(index=index,pts=int(frame.pts),time=float(t),absolute=absolute,
                time_base=[frame.time_base.numerator,frame.time_base.denominator],
                width=frame.width,height=frame.height)
            w=min(int(t//8),len(windows)-1)
            chosen=cursor<len(targets) and targets[cursor]<=t
            if chosen:
                while cursor<len(targets) and targets[cursor]<=t:cursor+=1
                add(frame,entry);covered.add(w)
                if w in backups:backups.pop(w)[2].close()
            elif w not in covered:
                a,b=windows[w];dist=abs(t-(a+b)/2)
                if w not in backups or (dist,t,index)<backups[w][0]:
                    if w in backups:backups[w][2].close()
                    backups[w]=((dist,t,index),entry,frame.to_image().convert('RGB'))
        flush()
    repairs=[]
    for w,(dist,entry,im) in sorted(backups.items()):
        entries.append(entry);features.extend(image_features(j,[im]));im.close()
        repairs.append(dict(window=w,index=entry['index'],time=entry['time']));covered.add(w)
    d=j.model.config.text_config.hidden_size
    f=np.asarray(features,dtype=np.float32).reshape(len(entries),d)
    order=sorted(range(len(entries)),key=lambda i:(entries[i]['time'],entries[i]['index']))
    entries=[entries[i] for i in order];f=f[order]
    assert len({e['index'] for e in entries})==len(entries)
    assert np.isfinite(f).all()
    return f,dict(entries=entries,input_video=str(source),manifest_video_path=row['video_path'],
        origin=origin,decoded_frames=decoded,coverage_repairs=repairs,
        windows_without_decoded_frame=[i for i in range(len(windows)) if i not in covered])


class Witness:
    def __init__(self,metadata,folder):
        import av
        self.metadata=metadata;self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.container=av.open(metadata['input_video']);self.stream=self.container.streams.video[0]
        self.stream.thread_type='AUTO'

    def image(self,entry):
        from PIL import Image
        path=self.folder/f'frame_{entry["index"]:08d}.png'
        if path.exists():
            with Image.open(path) as im:
                assert im.size==(entry['width'],entry['height'])
                return im.convert('RGB'),path
        self.container.seek(int(entry['absolute']/float(self.stream.time_base)),
            stream=self.stream,backward=True,any_frame=False)
        for frame in self.container.decode(self.stream):
            t=float(frame.pts*frame.time_base)
            if abs(t-entry['absolute'])<1e-8:
                assert frame.pts==entry['pts'] and frame.width==entry['width'] and frame.height==entry['height']
                im=frame.to_image().convert('RGB');im.save(path);return im,path
            if t>entry['absolute']+1e-8:raise ValueError('seek skipped indexed witness')
        raise ValueError('indexed witness missing')

    def close(self):self.container.close()


def clusters(features,entries,members,k):
    members=np.asarray(members,dtype=int)
    assert len(members)>0
    x=features[members];k=min(int(k),len(members))
    if k==1:labels=np.zeros(len(members),dtype=int)
    else:labels=KMeans(n_clusters=k,random_state=0,n_init=10,max_iter=300,tol=1e-4).fit_predict(x)
    groups=[]
    for label in sorted(set(labels.tolist())):
        ids=members[labels==label];center=features[ids].mean(0,dtype=np.float32)
        distances=np.sum((features[ids]-center)**2,axis=1)
        rep=min(range(len(ids)),key=lambda i:(float(distances[i]),entries[ids[i]]['time'],entries[ids[i]]['index']))
        groups.append(dict(members=ids.tolist(),center=center.tolist(),representative=int(ids[rep])))
    return sorted(groups,key=lambda g:(entries[g['representative']]['time'],entries[g['representative']]['index']))


class Builder:
    def __init__(self,features,entries,caption,relevance):
        self.features=features;self.entries=entries;self.caption=caption;self.relevance=relevance
        self.captions={};self.caption_events=[]

    def describe(self,node):
        rep=node['representative']
        if rep not in self.captions:
            self.captions[rep]=self.caption(rep);self.caption_events.append(rep)
        node['caption']=self.captions[rep]['text']
        node['caption_tokens']=self.captions[rep]['tokens']
        node['time']=self.entries[rep]['time']

    def build(self):
        if not len(self.entries):return dict(nodes=[],rounds=[],caption_events=[],captions={})
        members=list(range(len(self.entries)));rounds=[];roots=[];last_count=0
        for k in (4,8,16):
            roots=clusters(self.features,self.entries,members,k)
            for i,node in enumerate(roots):
                node.update(id=f'r{i:03d}',parent=None,depth=0);self.describe(node)
            relevance=self.relevance(roots)
            assert len(relevance)==len(roots) and all(r in (1,2,3) for r in relevance)
            for node,r in zip(roots,relevance):node['root_relevance']=r
            rounds.append(dict(requested_width=k,actual_width=len(roots),
                representatives=[n['representative'] for n in roots],relevance=relevance))
            if relevance.count(3)>=2 or k==16 or len(roots)<=last_count or len(roots)==len(members):break
            last_count=len(roots)
        nodes=[]
        def expand(node,remaining):
            nodes.append(node)
            if remaining<=0 or len(node['members'])<=1:return
            children=clusters(self.features,self.entries,node['members'],2)
            if len(children)<=1:return
            for i,ch in enumerate(children):
                ch.update(id=f'{node["id"]}.c{i:03d}',parent=node['id'],depth=node['depth']+1,
                    root_relevance=node['root_relevance']);self.describe(ch);expand(ch,remaining-1)
        for root in roots:expand(root,{1:0,2:1,3:2}[root['root_relevance']])
        assert len({n['id'] for n in nodes})==len(nodes)<=112
        return dict(nodes=nodes,rounds=rounds,caption_events=self.caption_events,
            captions={str(k):v for k,v in self.captions.items()})


def record(node,parent=True):
    return (f'[node={node["id"]}; parent={node["parent"] or "ROOT" if parent else "NONE"}; '
        f'depth={node["depth"] if parent else 0}; t={node["time"]:.3f}s]\n{node["caption"]}\n')


def tree_text(tree,parent=True):
    nodes=sorted(tree['nodes'],key=lambda n:(n['time'],n['id']))
    return TREE_HEADER+(''.join(record(n,parent) for n in nodes) if nodes else '(none)\n')


def window_packet(tree,features,entries,a,b):
    nodes={n['id']:n for n in tree['nodes']};parents={n['parent'] for n in tree['nodes'] if n['parent']}
    candidates=[]
    for node in tree['nodes']:
        if node['id'] in parents:continue
        members=[i for i in node['members'] if a<=entries[i]['time']<b]
        if not members:continue
        center=np.asarray(node['center'],dtype=np.float32)
        rep=min(members,key=lambda i:(float(np.sum((features[i]-center)**2)),entries[i]['time'],entries[i]['index']))
        dist=float(np.sum((features[rep]-center)**2))
        candidates.append(((-node['root_relevance'],dist,entries[rep]['time'],entries[rep]['index'],node['id']),rep,node))
    selected=[];used=set()
    for rank,rep,node in sorted(candidates,key=lambda p:p[0]):
        if entries[rep]['index'] not in used:
            selected.append((rep,node));used.add(entries[rep]['index'])
        if len(selected)==2:break
    selected.sort(key=lambda p:(entries[p[0]]['time'],entries[p[0]]['index']))
    ancestors=[];seen=set()
    for rep,node in selected:
        chain=[];parent=node['parent']
        while parent is not None:
            chain.append(nodes[parent]);parent=nodes[parent]['parent']
        for n in reversed(chain):
            if n['id'] not in seen:ancestors.append(n);seen.add(n['id'])
    return dict(pool_members=[rep for rep,node in selected],leaf_ids=[node['id'] for rep,node in selected],
        ancestor_ids=[n['id'] for n in ancestors],context=CONTEXT_HEADER+
        (''.join(record(n) for n in ancestors) if ancestors else '(none)\n'))


@torch.no_grad()
def caption(j,witness,entry):
    im,path=witness.image(entry)
    msgs=[j.turn('system',CAP_SYSTEM),{'role':'user','content':[{'type':'image'},{'type':'text','text':CAP_QUESTION}]}]
    text=j.render(msgs,True)
    try:enc=j.encode(text,[im]);pixel_check(enc)
    finally:im.close()
    j.model.model.rope_deltas=None
    out=j.model.model(**j.model_inputs(enc),use_cache=True)
    cache=out.past_key_values;h=out.last_hidden_state[0,-1];del out
    if not hasattr(j,'caption_W32'):j.caption_W32=j.model.get_output_embeddings().weight.float()
    gen=[];stopped=False
    for _ in range(96):
        logits=h.float()@j.caption_W32.T
        if j.softcap:logits=torch.tanh(logits/j.softcap)*j.softcap
        nxt=int(logits.argmax())
        if nxt in j.eos_ids:stopped=True;break
        gen.append(nxt);h=j._step(cache,[nxt])
    del cache
    return dict(text=j.tok.decode(gen,skip_special_tokens=True).strip(),tokens=gen,
        truncated=not stopped,path=str(path),system=CAP_SYSTEM,question=CAP_QUESTION)


@torch.no_grad()
def relevance(j,nodes):
    from src.mllm_judge import YOUTUBE_RULES,VIDEO_QUESTION,SYSTEM_MESSAGE
    choices=[j.tok.encode(str(i),add_special_tokens=False) for i in (1,2,3)]
    assert all(len(v)==1 for v in choices) and len({v[0] for v in choices})==3
    body='Observations:\n'+''.join(record(n) for n in nodes)+'\nPlatform rules:\n'+YOUTUBE_RULES+'\nQuestion:\n'+VIDEO_QUESTION+'\n'
    msgs=[j.turn('system',SYSTEM_MESSAGE),j.turn('user',body)]
    j.model.model.rope_deltas=None;text,enc=j.encode_prefix(msgs,[])
    cache=j.prefix_cache(enc);n=cache.get_seq_length();delta=j.model.model.rope_deltas.clone();scores=[]
    for node in nodes:
        q=(f'Rate how informative observation {node["id"]} is for answering the question, '
            'including evidence for either answer. 1 = not informative; 2 = somewhat informative; '
            '3 = highly informative. Answer with one digit: 1, 2, or 3.')
        bids,_=j.branch_ids(msgs,q,head_text=text);j.model.model.rope_deltas=delta.clone()
        try:lp=j.cached_choices(cache,bids,choices,in_place=True)
        finally:cache.crop(n);j.model.model.rope_deltas=delta.clone()
        node['relevance_logprobs']=lp;scores.append(int(np.argmax(lp))+1)
    del cache
    return scores
