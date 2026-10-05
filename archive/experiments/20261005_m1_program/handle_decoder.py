"""Measured constrained greedy stream; literal grammar, no output repair."""
import json
import time
import torch
from PIL import Image
from handle_interface import CONSTANTS


class Capped(Exception):pass


def valid_content(text):
    return all(ord(c)>=32 and c not in ('"','\\','\ufffd') for c in text)


class Stream:
    def __init__(self,j,limit,hidden=None,cache=None,tokens=None):
        self.j=j;self.limit=limit;self.hidden=hidden;self.cache=cache
        self.replay=tokens is not None;self.saved=tokens or [];self.tokens=[];self.events=[]

    def append(self,token):
        if len(self.tokens)>=self.limit:raise Capped()
        if self.replay:
            assert len(self.tokens)<len(self.saved) and self.saved[len(self.tokens)]==token
        else:self.hidden=self.j._step(self.cache,[token])
        self.tokens.append(token)

    def logits(self):
        z=self.hidden.float()@self.j.generation_W32.T
        return torch.tanh(z/self.j.softcap)*self.j.softcap if self.j.softcap else z

    def greedy(self,allowed):
        allowed=sorted(allowed)
        if self.replay:
            if len(self.tokens)>=len(self.saved):raise Capped()
            nxt=self.saved[len(self.tokens)];assert nxt in allowed;return nxt
        ids=torch.tensor(allowed,device=self.hidden.device,dtype=torch.long)
        return allowed[int(self.logits()[ids].argmax())]

    def force(self,text):
        start=len(self.tokens);ids=self.j.tok.encode(text,add_special_tokens=False)
        for token in ids:self.append(token)
        self.events.append(dict(kind='forced',text=text,start=start,end=len(self.tokens)))

    def choose(self,options):
        assert options and len(set(options))==len(options)
        sequences=[self.j.tok.encode(x,add_special_tokens=False) for x in options]
        assert all(sequences) and len({tuple(x) for x in sequences})==len(sequences)
        assert not any(len(a)<len(b) and b[:len(a)]==a for a in sequences for b in sequences)
        start=len(self.tokens);prefix=[];pool=list(range(len(options)))
        while True:
            n=len(prefix);allowed={sequences[i][n] for i in pool}
            token=self.greedy(allowed);self.append(token);prefix.append(token)
            pool=[i for i in pool if sequences[i][:len(prefix)]==prefix]
            ended=[i for i in pool if len(sequences[i])==len(prefix)]
            if ended:
                assert len(pool)==len(ended)==1;value=options[ended[0]];break
        self.events.append(dict(kind='choice',options=options,selected=value,start=start,end=len(self.tokens)))
        return value

    def description(self):
        start=len(self.tokens);content=[];text='';reason='model_quote'
        quote=self.j.tok.encode('"',add_special_tokens=False);assert len(quote)==1
        q=quote[0]
        if not hasattr(self.j,'description_tokens'):
            vocab_size=len(self.j.tok)
            self.j.description_tokens=[i for i in range(vocab_size)
                if i not in self.j.tok.all_special_ids and valid_content(self.j.tok.decode([i]))
                and self.j.tok.decode([i])]
        while True:
            if len(content)>=CONSTANTS['field_tokens'] or len(text.split())>=CONSTANTS['description_words']:
                reason='content_token_cap' if len(content)>=CONSTANTS['field_tokens'] else 'word_cap'
                self.append(q);break
            if self.replay:
                if len(self.tokens)>=len(self.saved):raise Capped()
                token=self.saved[len(self.tokens)]
                if token!=q:
                    assert token in self.j.description_tokens
                    candidate=self.j.tok.decode(content+[token])
                    assert valid_content(candidate) and len(candidate.split())<=CONSTANTS['description_words']
            else:
                allowed=self.j.description_tokens+[q];z=self.logits()
                ids=torch.tensor(allowed,device=z.device,dtype=torch.long)
                ranked=z[ids].clone()
                while True:
                    k=int(ranked.argmax());token=allowed[k]
                    if token==q:break
                    candidate=self.j.tok.decode(content+[token])
                    if valid_content(candidate) and len(candidate.split())<=CONSTANTS['description_words']:break
                    ranked[k]=float('-inf')
            self.append(token)
            if token==q:break
            content.append(token);text=self.j.tok.decode(content)
        self.events.append(dict(kind='description',text=text,reason=reason,start=start,end=len(self.tokens)))


@torch.no_grad()
def generate(j,root,system,content,paths,limit,writer):
    torch.cuda.synchronize();start=time.perf_counter();before=j.forward_calls
    rendered=j.render([j.turn('system',system),dict(role='user',content=content)],True)
    images=[Image.open(root/p).convert('RGB') for p in paths]
    try:enc=j.encode(rendered,images)
    finally:
        for image in images:image.close()
    # The FP32 vocabulary copy is scratch; prefill does not use it. Rebuild the
    # same values after prefill instead of keeping 2.3GiB live during long MLPs.
    if hasattr(j,'generation_W32'):del j.generation_W32
    j.model.model.rope_deltas=None
    output=j.model.model(**j.model_inputs(enc),use_cache=True)
    cache=output.past_key_values;hidden=output.last_hidden_state[0,-1].clone();del output
    if not hasattr(j,'generation_W32'):j.generation_W32=j.model.get_output_embeddings().weight.float()
    stream=Stream(j,limit,hidden,cache);del hidden;complete=True;value=None
    try:value=writer(stream)
    except Capped:complete=False
    finally:del cache
    torch.cuda.synchronize()
    return dict(text=j.tok.decode(stream.tokens,skip_special_tokens=True).strip(),tokens=stream.tokens,
        truncated=not complete,events=stream.events,selection=value,seconds=time.perf_counter()-start,
        actual_forwards=j.forward_calls-before,prompt=rendered,system=system,max_tokens=limit,
        image_paths=paths,input_tokens=enc['input_ids'][0].tolist(),
        image_grid=enc['image_grid_thw'].tolist() if 'image_grid_thw' in enc else [])


def validate_generation(j,root,g,system,content,paths,limit,writer):
    rendered=j.render([j.turn('system',system),dict(role='user',content=content)],True)
    assert g['system']==system and g['prompt']==rendered and g['image_paths']==paths and g['max_tokens']==limit
    images=[Image.open(root/p).convert('RGB') for p in paths]
    try:enc=j.encode(rendered,images)
    finally:
        for image in images:image.close()
    assert enc['input_ids'][0].tolist()==g['input_tokens']
    assert (enc['image_grid_thw'].tolist() if 'image_grid_thw' in enc else [])==g['image_grid']
    stream=Stream(j,limit,tokens=g['tokens']);complete=True;value=None
    try:value=writer(stream)
    except Capped:complete=False
    assert stream.tokens==g['tokens'] and stream.events==g['events'] and value==g['selection']
    assert g['truncated']==(not complete) and (complete or len(g['tokens'])==limit)
    assert j.tok.decode(g['tokens'],skip_special_tokens=True).strip()==g['text']
    assert g['actual_forwards']==1+len(g['tokens'])
