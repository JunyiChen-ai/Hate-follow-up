"""Neutral retrieval keys and source-ID filtering, never hate predictions."""
import re
from retrieval import SPEC


def unavailable(text):
    return not text.strip() or bool(re.match(r'^UNKNOWN\b',text.lstrip(),re.I))


def field(stream,words,tokens):
    old=(stream.description_words,stream.field_tokens)
    stream.description_words=words;stream.field_tokens=tokens
    try:stream.description();event=stream.events[-1]
    finally:stream.description_words,stream.field_tokens=old
    return 'UNKNOWN' if event['reason']!='model_quote' or unavailable(event['text']) else event['text']


def key_writer(stream):
    result={};stream.force('{"caption":"')
    result['caption']=field(stream,SPEC['caption_words'],SPEC['caption_tokens'])
    for key in SPEC['key_order']:
        stream.force(',"'+key+'":"');result[key]=field(stream,SPEC['key_words'],SPEC['key_tokens'])
    stream.force('}');return result


def filter_writer(candidates):
    def write(stream):
        stream.force('{"status":"');status=stream.choose(['SUPPORTED",','INSUFFICIENT",','UNKNOWN",'])[:-2]
        stream.force('"ids":[');ids=[]
        closed=False
        while len(ids)<SPEC['max_remote_windows']:
            available=[i for i in candidates if i not in ids]
            lead='' if not ids else ','
            options=[']']+[lead+str(i)+' ' for i in available]
            picked=stream.choose(options)
            if picked==']':closed=True;break
            ids.append(int(picked[len(lead):]))
        if not closed:stream.force(']')
        stream.force(',"new_query":"')
        query=field(stream,SPEC['query_words'],SPEC['query_tokens']) if status=='INSUFFICIENT' else 'UNKNOWN'
        if status!='INSUFFICIENT':stream.force('UNKNOWN"')
        stream.force('}');return dict(status=status,ids=ids,new_query=query)
    return write


def compile_keys(g):
    if g['truncated'] or len(g['tokens'])>=g['max_tokens']:
        return {key:'UNKNOWN' for key in ['caption',*SPEC['key_order']]}
    return g['selection']


def compile_filter(g):
    if g['truncated'] or len(g['tokens'])>=g['max_tokens']:
        return dict(status='UNKNOWN',ids=[],new_query='UNKNOWN')
    return g['selection']
