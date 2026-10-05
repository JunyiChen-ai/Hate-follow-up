"""Measured bounded fields and neutral relevance, never hate scores."""
import re
from events import SPEC


def unavailable(text):return not text.strip() or bool(re.match(r'^(UNKNOWN|NONE)\b',text.lstrip(),re.I))


def description_writer(key):
    def write(stream):
        stream.force('{"'+key+'":"');old=stream.description_words,stream.field_tokens
        stream.description_words=SPEC['description_words'];stream.field_tokens=SPEC['description_tokens']
        try:stream.description();event=stream.events[-1]
        finally:stream.description_words,stream.field_tokens=old
        text='UNKNOWN' if event['reason']!='model_quote' or unavailable(event['text']) else event['text'];stream.force('}')
        return {key:text}
    return write


def compile_description(g,key):return 'UNKNOWN' if g['truncated'] or len(g['tokens'])>=g['max_tokens'] else g['selection'][key]


def relevance_writer(available):
    def write(stream):
        values=[];stream.force('{"relevance":[')
        for i,valid in enumerate(available):
            if i:stream.force(',')
            options=['"UNKNOWN" ']+[str(v)+' ' for v in range(1,11)] if valid else ['"UNKNOWN" ']
            value=stream.choose(options).strip();values.append(None if value=='"UNKNOWN"' else int(value))
        stream.force(']}');return dict(relevance=values)
    return write


def compile_relevance(g,count):
    return [None]*count if g['truncated'] or len(g['tokens'])>=g['max_tokens'] else g['selection']['relevance']
