"""Neutral captions and hypothetical conditions; never factual final evidence."""
import re
from retrieval import SPEC


def unavailable(text):
    return not text.strip() or bool(re.match(r'^(UNKNOWN|NONE)\b',text.lstrip(),re.I))


def field(stream,words,tokens):
    old=stream.description_words,stream.field_tokens
    stream.description_words=words;stream.field_tokens=tokens
    try:
        stream.description();event=stream.events[-1]
    finally:stream.description_words,stream.field_tokens=old
    return 'UNKNOWN' if event['reason']!='model_quote' or unavailable(event['text']) else event['text']


def caption_writer(stream):
    stream.force('{"caption":"');text=field(stream,SPEC['caption_words'],SPEC['caption_tokens']);stream.force('}')
    return dict(caption=text)


def slot_writer(indices):
    def write(stream):
        result=[];stream.force('[')
        for ordinal,index in enumerate(indices):
            if ordinal:stream.force(',')
            stream.force('{"id":'+str(index));row=dict(id=index)
            for role in SPEC['slot_order']:
                stream.force(',"'+role+'":"');row[role]=field(stream,SPEC['slot_words'],SPEC['slot_tokens'])
            stream.force('}');result.append(row)
        stream.force(']');return result
    return write


def compile_caption(g):
    return dict(caption='UNKNOWN') if g['truncated'] or len(g['tokens'])>=g['max_tokens'] else g['selection']


def compile_slots(g,indices):
    if g['truncated'] or len(g['tokens'])>=g['max_tokens']:
        return [dict(id=i,**{role:'UNKNOWN' for role in SPEC['slot_order']}) for i in indices]
    assert [r['id'] for r in g['selection']]==indices
    return g['selection']
