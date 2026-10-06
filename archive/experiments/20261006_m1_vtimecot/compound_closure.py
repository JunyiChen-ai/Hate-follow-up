"""C-only complete quoted-field syntax, actual tokens and scoped restoration."""
from contextlib import contextmanager
import torch
import src.structured_source_generation as base
from timeline import INTERFACE
OriginalStream=base.Stream


def prefix_consumed(pending,literal):
    """Match already-generated syntax, allowing JSON whitespace outside strings."""
    index=0;inside=False
    for char in pending:
        if char in ' \t\r\n' and not inside:continue
        if index>=len(literal) or literal[index]!=char:return None
        if char=='"':inside=not inside
        index+=1
    return index


class CompleteQuotedStream(OriginalStream):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.pending='';self.following_literal=None

    def force(self,text):
        if not self.pending:return super().force(text)
        consumed=prefix_consumed(self.pending,text);assert consumed is not None,'compound closure conflicts with required JSON literal'
        pending=self.pending;self.pending='';start=len(self.tokens)
        for token in self.j.tok.encode(text[consumed:],add_special_tokens=False):self.append(token)
        self.events.append(dict(kind='forced',text=text,start=start,end=len(self.tokens),
            compound_emitted=pending,compound_literal_chars=consumed))

    def closing_tokens(self):
        assert self.following_literal is not None
        cache=getattr(self.j,'complete_quote_tokens',None)
        if cache is None:cache={};self.j.complete_quote_tokens=cache
        if self.following_literal not in cache:
            result={}
            for token in range(len(self.j.tok)):
                if token in self.j.tok.all_special_ids:continue
                decoded=self.j.tok.decode([token])
                if decoded.startswith('"') and prefix_consumed(decoded[1:],self.following_literal) is not None:result[token]=decoded[1:]
            cache[self.following_literal]=result
        return cache[self.following_literal]

    def description(self):
        assert not self.pending
        start=len(self.tokens);content=[];text='';reason='model_quote';closing=self.closing_tokens()
        quote=self.j.tok.encode('"',add_special_tokens=False);assert len(quote)==1 and quote[0] in closing
        if not hasattr(self.j,'description_tokens'):
            self.j.description_tokens=[i for i in range(len(self.j.tok)) if i not in self.j.tok.all_special_ids
                and base.valid_content(self.j.tok.decode([i])) and self.j.tok.decode([i])]
        close_token=None
        while True:
            if len(content)>=self.field_tokens or len(text.split())>=self.description_words:
                reason='content_token_cap' if len(content)>=self.field_tokens else 'word_cap';self.append(quote[0]);close_token=quote[0];break
            if self.replay:
                if len(self.tokens)>=len(self.saved):raise base.Capped()
                token=self.saved[len(self.tokens)]
                if token not in closing:
                    assert token in self.j.description_tokens
                    candidate=self.j.tok.decode(content+[token]);assert base.valid_content(candidate) and len(candidate.split())<=self.description_words
            else:
                allowed=sorted(set(self.j.description_tokens)|set(closing));z=self.logits()
                ids=torch.tensor(allowed,device=z.device,dtype=torch.long);ranked=z[ids].clone()
                while True:
                    k=int(ranked.argmax());token=allowed[k]
                    if token in closing:break
                    candidate=self.j.tok.decode(content+[token])
                    if base.valid_content(candidate) and len(candidate.split())<=self.description_words:break
                    ranked[k]=float('-inf')
            self.append(token)
            if token in closing:self.pending=closing[token];close_token=token;break
            content.append(token);text=self.j.tok.decode(content)
        self.events.append(dict(kind='description',text=text,reason=reason,start=start,end=len(self.tokens),
            closing_token=close_token,emitted_next_literal=self.pending))


@contextmanager
def stream_scope():
    original=base.Stream
    if INTERFACE=='C':base.Stream=CompleteQuotedStream
    try:yield
    finally:base.Stream=original


def generate(*args,**kwargs):
    with stream_scope():return base.generate(*args,**kwargs)


def validate_generation(*args,**kwargs):
    with stream_scope():return base.validate_generation(*args,**kwargs)
