"""Word-identity mapping and one-reader support-conditioned attention."""
from contextlib import contextmanager
import re
import numpy as np
import torch

ARMS = ('soft','hard','proportional','unweighted','shifted')


def arm_support(alignment, arm):
    assert arm in ARMS
    p = np.asarray(alignment['soft'],dtype=float)
    if p.size == 0: return np.zeros((0,len(alignment['windows'])))
    if arm == 'proportional': p = np.asarray(alignment['proportional'],dtype=float)
    if arm == 'hard':
        p = np.asarray(alignment['viterbi'],dtype=float)
        hard = np.zeros_like(p); hard[np.arange(len(p)),p.argmax(1)] = 1; p = hard
    if arm == 'shifted': p = np.roll(p,p.shape[1]//2,axis=1)
    assert p.shape == (len(alignment['words']),len(alignment['windows']))
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1),1)
    return p


def prefix_word_spans(j, text, enc, segments, words):
    token = j.processor.image_token; parts = text.split(token)
    assert len(parts) == len(j.img_tokens)+1
    expanded = parts[0]
    for tail, count in zip(parts[1:],j.img_tokens): expanded += token*count+tail
    aligned = j.tok(expanded,add_special_tokens=False,return_offsets_mapping=True)
    assert aligned['input_ids'] == enc['input_ids'][0].tolist()
    offsets = np.asarray(aligned['offset_mapping'])
    cursor = expanded.index('\nTranscript:\n')+len('\nTranscript:\n')
    end = expanded.index('\nBased on this platform',cursor)
    spans = []
    for si,(s,e,body) in enumerate(segments):
        if not body.strip(): continue
        line = f'[{s:.1f}s-{e:.1f}s] {body.strip()}'
        start = expanded.index(line,cursor); cursor = start+len(line)
        assert cursor <= end
        body_start = start+len(f'[{s:.1f}s-{e:.1f}s] ')
        ws = [w for w in words if w['segment']==si]
        matches = list(re.finditer(r'\S+',body.strip()))
        assert len(matches) == len(ws)
        for k,(m,w) in enumerate(zip(matches,ws)):
            assert m.group() == w['text'] and m.start() == w['char_start'] and m.end() == w['char_end']
            # Each interword space belongs to the following word; first space after timestamp is scaffold.
            lo = body_start+(matches[k-1].end() if k else m.start())
            spans.append((w['id'],lo,body_start+m.end()))
    assert len(spans) == len(words) and [w for w,a,b in spans] == list(range(len(words)))
    return offsets, spans


def token_support(offsets, spans, support):
    """Cross-word tokens take character-overlap weighted support; unbound keys retain 1."""
    offsets = np.asarray(offsets); weighted = np.zeros(len(offsets)); count = np.zeros(len(offsets))
    for word, a, b in spans:
        overlap = np.maximum(0,np.minimum(offsets[:,1],b)-np.maximum(offsets[:,0],a))
        weighted += overlap*support[word]; count += overlap
    out = np.ones(len(offsets)); ids = count > 0; out[ids] = weighted[ids]/count[ids]
    assert np.isfinite(out).all() and (out >= 0).all() and (out <= 1+1e-8).all()
    return out, ids


def question_word_spans(j, suffix, bids, question, words, selected):
    aligned = j.tok(suffix,add_special_tokens=False,return_offsets_mapping=True)
    assert aligned['input_ids'] == bids
    if not selected: return np.asarray(aligned['offset_mapping']), []
    qstart = suffix.index(question)
    body_start = qstart+question.index('Judge only what is spoken in this window: ')+len('Judge only what is spoken in this window: ')
    cursor = body_start; spans = []
    for k,wid in enumerate(selected):
        w = words[wid]['text']; start = cursor+(1 if k else 0)
        assert suffix[start:start+len(w)] == w
        spans.append((wid,cursor,start+len(w))); cursor = start+len(w)
    return np.asarray(aligned['offset_mapping']), spans


class PriorReader:
    def __init__(self,judge):
        self.j = judge; self.active = None; self.visited = []
        self.layers = judge.model.model.language_model.layers
        self.hooks = [layer.self_attn.register_forward_pre_hook(self._hook(i),with_kwargs=True)
                      for i,layer in enumerate(self.layers)]

    def _hook(self,i):
        def hook(module,args,kwargs):
            if self.active is None: return
            self.visited.append(i)
            assert 'attention_mask' in kwargs
            return args,{**kwargs,'attention_mask':self.active}
        return hook

    @contextmanager
    def condition(self,prefix,queries,key_support):
        assert self.active is None and len(key_support)==prefix+queries
        j=self.j; keys=torch.arange(prefix+queries,device=j.device)
        rows=torch.arange(queries,device=j.device)+prefix
        weight=torch.as_tensor(key_support,device=j.device,dtype=torch.float32)
        mask=weight.clamp_min(1e-6).log()[None,:].expand(queries,-1).clone()
        mask.masked_fill_(keys[None,:]>rows[:,None],float('-inf'))
        self.active=mask.to(j.dtype)[None,None]; self.visited=[]
        try: yield
        finally: self.active=None
        assert self.visited == list(range(len(self.layers)))

    def margin(self,cache,ids,support,delta):
        n=cache.get_seq_length(); self.j.model.model.rope_deltas=delta.clone()
        try:
            with self.condition(n,len(ids),support):
                z=self.j.cached_margin(cache,ids,in_place=True)
        finally: cache.crop(n); self.j.model.model.rope_deltas=delta.clone()
        assert cache.get_seq_length()==n
        return z

    def close(self):
        for h in self.hooks: h.remove()
