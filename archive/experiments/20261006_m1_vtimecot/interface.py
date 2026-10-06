"""Bounded neutral queries/plans; typed tools, no generated factual evidence."""
import re
from timeline import SPEC,INTERFACE


def unavailable(text):return not text.strip() or bool(re.match(r'^(UNKNOWN|NONE)\b',text.lstrip(),re.I))


def field(stream,words,tokens,following=None):
    old=stream.description_words,stream.field_tokens;stream.description_words=words;stream.field_tokens=tokens
    if INTERFACE=='C':stream.following_literal=following
    try:stream.description();event=stream.events[-1]
    finally:stream.description_words,stream.field_tokens=old
    return 'UNKNOWN' if event['reason']!='model_quote' or unavailable(event['text']) else event['text']


def query_writer(stream):
    result=[];stream.force('[')
    for i in range(SPEC['query_count']):
        stream.force((',' if i else '')+'"');result.append(field(stream,SPEC['query_words'],SPEC['query_tokens'],',"' if i+1<SPEC['query_count'] else ']'))
    stream.force(']');return result


def compile_queries(g):
    if g['truncated'] or len(g['tokens'])>=g['max_tokens']:return []
    return list(dict.fromkeys(q for q in g['selection'] if not unavailable(q)))


def relevance_writer(stream):
    stream.force('{"relevance":');value=stream.choose(['"UNKNOWN"}']+[str(i)+'}' for i in range(11)])[:-1]
    return dict(relevance=None if value=='"UNKNOWN"' else int(value))


def compile_relevance(g):
    return None if g['truncated'] or len(g['tokens'])>=g['max_tokens'] else g['selection']['relevance']


def plan_writer(state,tables):
    def write(stream):
        stream.force('{"reason":"');reason=field(stream,SPEC['planner_words'],SPEC['planner_tokens'],',"action":"')
        allowed=['TERMINATE','UNKNOWN']
        if not state['progress']:allowed.append('PROGRESS_BAR')
        elif state['query_id'] is None and any(t['intervals'] for t in tables):allowed.append('HIGHLIGHT')
        elif state['query_id'] is not None and not state['cut']:allowed.append('CUT')
        stream.force(',"action":"');action=stream.choose([a+'"' for a in allowed])[:-1];result=dict(reason=reason,action=action)
        if action=='HIGHLIGHT':
            stream.force(',"query_id":');result['query_id']=int(stream.choose([str(i)+' ' for i,t in enumerate(tables) if t['intervals']]))
        elif action=='CUT':
            stream.force(',"interval_id":');result['interval_id']=int(stream.choose([str(r['id'])+' ' for r in tables[state['query_id']]['intervals']]))
        stream.force('}');return result
    return write


def compile_plan(g):
    if g['truncated'] or len(g['tokens'])>=g['max_tokens'] or unavailable(g['selection']['reason']):return dict(reason='UNKNOWN',action='UNKNOWN')
    return g['selection']


def feedback_writer(stream):
    stream.force('{"description":"');value=field(stream,SPEC['planner_words'],SPEC['planner_tokens'],'}');stream.force('}');return dict(description=value)
