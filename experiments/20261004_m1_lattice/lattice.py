"""Independent-slot confusion DAG and its conditional predecessor operators."""
import math
import re
import numpy as np

CACHE_VERSION='R1 temporal beam word lattice, sources 2026-10-05; first-observed audio overlap'
ASR_MODEL='openai/whisper-large-v3'
CONSTANTS=dict(window_seconds=8,sample_rate=16000,beam_size=5,return_sequences=5,
    decoder_max_length=448,length_penalty=1,temperature=1,seed=0,
    language_detection_seconds=30,edit_costs=[0,1,1,1],edit_ties=['diagonal','deletion','insertion'],
    audio_overlap='first observed resampled sample in decode order; no shift or averaging')
SCAFFOLD=('Consider only window {i} of {n}, from {a:.1f}s to {b:.1f}s of this video.\n'
    'The following speech lattice contains alternative transcriptions of the same audio. '
    'Alternatives in one slot are mutually exclusive; a null edge means no word. '
    'The complete video transcript is interpretation context.\nSpeech lattice:\n')
TAIL='\n\nIs THIS window one of the segments where speech that violates the above rules occurs?\n\nAnswer "Yes" or "No".'


def align_words(anchor,words):
    """Star alignment; insertion phrases belong to an anchor boundary."""
    n,m=len(anchor),len(words);d=np.zeros((n+1,m+1),dtype=int)
    d[:,0]=np.arange(n+1);d[0,:]=np.arange(m+1)
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost=int(anchor[i-1].casefold()!=words[j-1].casefold())
            d[i,j]=min(d[i-1,j-1]+cost,d[i-1,j]+1,d[i,j-1]+1)
    edits=[];i,j=n,m
    while i or j:
        if i and j and d[i,j]==d[i-1,j-1]+int(anchor[i-1].casefold()!=words[j-1].casefold()):
            edits.append(('word',i-1,words[j-1]));i-=1;j-=1
        elif i and d[i,j]==d[i-1,j]+1:
            edits.append(('word',i-1,''));i-=1
        else:
            assert j and d[i,j]==d[i,j-1]+1
            edits.append(('insert',i,words[j-1]));j-=1
    values=['']*n;insertions=[[] for _ in range(n+1)]
    for kind,index,text in reversed(edits):
        if kind=='word':values[index]=text
        else:insertions[index].append(text)
    return values,[' '.join(x) for x in insertions]


def confusion(beams):
    assert len(beams)==5 and all(np.isfinite(b['score']) for b in beams)
    scores=np.asarray([b['score'] for b in beams],dtype=np.float32)
    weights=np.exp(scores-scores.max(),dtype=np.float32);weights/=weights.sum(dtype=np.float32)
    ordered=sorted(range(5),key=lambda i:(-beams[i]['score'],i))
    anchor=next((re.findall(r'\S+',beams[i]['text']) for i in ordered if beams[i]['text'].strip()),[])
    aligned=[align_words(anchor,re.findall(r'\S+',b['text'])) for b in beams]
    slots=[]
    def add(kind,index,values):
        if kind=='insert' and all(not v for v in values):return
        options={}
        for i,value in enumerate(values):
            if value not in options:options[value]=dict(text=value,mass=0.,beams=[])
            options[value]['mass']+=float(weights[i]);options[value]['beams'].append(i)
        options=sorted(options.values(),key=lambda v:(-v['mass'],min(v['beams']),v['text']))
        assert np.isclose(sum(x['mass'] for x in options),1.,atol=1e-6)
        slots.append(dict(kind=kind,anchor_index=index,alternatives=options))
    for i in range(len(anchor)+1):
        add('insert',i,[x[1][i] for x in aligned])
        if i<len(anchor):add('word',i,[x[0][i] for x in aligned])
    return dict(anchor_words=anchor,slots=slots,beam_weights=weights.tolist(),
        onebest=beams[ordered[0]]['text'],approximation='independent slot recombination; not full beam posterior')


def tokenize_graph(confusion,tokenizer):
    """Physical node identity and logical longest-path positions are distinct."""
    nodes=[];ids=[];offset=0
    for slot_index,slot in enumerate(confusion['slots']):
        width=0
        for alt_index,alt in enumerate(slot['alternatives']):
            token_ids=tokenizer.encode(' '+alt['text'],add_special_tokens=False) if alt['text'] and alt['mass']>0 else []
            width=max(width,len(token_ids))
            for subword,token in enumerate(token_ids):
                nodes.append(dict(slot=slot_index,alternative=alt_index,subword=subword,
                    logical=offset+subword,mass=alt['mass'],physical=len(ids)))
                ids.append(token)
        offset+=width
    return dict(ids=ids,nodes=nodes,path_length=offset,slots=confusion['slots'])


def graph_bias(graph,head_length,tail_length,prefix_length,arm='full'):
    """No cached prefix query rows; suffix queries see the full immutable prefix."""
    assert arm in ('full','binary','flat')
    g=len(graph['ids']);length=head_length+g+tail_length
    bias=np.full((length,prefix_length+length),-np.inf,dtype=np.float32)
    bias[:,:prefix_length]=0.
    for i in range(head_length):bias[i,prefix_length:prefix_length+i+1]=0.
    if g:
        slot=np.asarray([n['slot'] for n in graph['nodes']]);alt=np.asarray([n['alternative'] for n in graph['nodes']])
        mass=np.asarray([n['mass'] for n in graph['nodes']],dtype=np.float32)
        logmass=np.zeros(g,dtype=np.float32) if arm=='binary' else np.log(mass)
        predecessor=slot[None,:]<slot[:,None]
        own=(slot[None,:]==slot[:,None])&(alt[None,:]==alt[:,None])&np.tri(g,dtype=bool)
        bias[head_length:head_length+g,prefix_length:prefix_length+head_length]=0.
        bias[head_length:head_length+g,prefix_length+head_length:prefix_length+head_length+g]=np.where(
            predecessor,logmass[None,:],np.where(own,0.,-np.inf))
    for i in range(tail_length):
        row=head_length+g+i
        bias[row,prefix_length:prefix_length+head_length]=0.
        if g:bias[row,prefix_length+head_length:prefix_length+head_length+g]=logmass
        bias[row,prefix_length+head_length+g:prefix_length+row+1]=0.
    if arm=='flat':
        bias[:,prefix_length:]=np.where(np.tri(length,dtype=bool),0.,-np.inf)
    assert np.isfinite(bias[:,:prefix_length]).all()
    assert all(np.isfinite(bias[i,prefix_length+i]) for i in range(length))
    return bias


def graph_positions(graph,head_length,tail_length,start,arm='full'):
    if arm=='flat':return list(range(start,start+head_length+len(graph['ids'])+tail_length))
    return list(range(start,start+head_length))+[
        start+head_length+n['logical'] for n in graph['nodes']]+list(range(
        start+head_length+graph['path_length'],start+head_length+graph['path_length']+tail_length))


def validate_graph_trace(trace,current_confusion,tokenizer,prefix_tokens,logical_start):
    """Bind a saved score to current lexical alternatives and their actual tokens."""
    expected=tokenize_graph(current_confusion,tokenizer)
    assert trace['graph']==expected,'saved graph differs from current confusion/tokenization'
    assert trace['arm']=='full' and trace['prefix_tokens']==prefix_tokens
    assert trace['prefix_logical_start']==logical_start
    head=tokenizer.encode(trace['head_text'],add_special_tokens=False)
    tail=tokenizer.encode(trace['tail_text'],add_special_tokens=False)
    assert trace['head_tokens']==head and trace['tail_tokens']==tail
    assert trace['graph_tokens']==len(expected['ids'])
    assert trace['slot_count']==len(expected['slots'])
    assert trace['physical_tokens']==len(head)+len(expected['ids'])+len(tail)
    assert trace['logical_positions']==graph_positions(expected,len(head),len(tail),logical_start)
