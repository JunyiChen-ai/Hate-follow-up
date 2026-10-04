"""Typed local factual replacement and real source ownership; no score or GT."""
from pathlib import Path
import copy
import json
import re

SPEC = json.loads((Path(__file__).parent / 'spec.json').read_text())
VERSION = SPEC['version']
CONSTANTS = SPEC['constants']
FIELDS = tuple(SPEC['fields'])


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def model_visible(value):
    if isinstance(value, dict):
        return {k: model_visible(v) for k, v in value.items() if k != 'path'}
    if isinstance(value, list):
        return [model_visible(v) for v in value]
    return value


def catalog(window):
    """All current local frames and overlapping-safe unique literal spans."""
    body = window['body']
    words = list(re.finditer(r'\S+', body))
    prefix = f'w{window["i"]:04d}:'
    sources = {}
    for frame in window['frames']:
        handle = prefix + frame['id']
        sources[handle] = dict(handle=handle, kind='visual', owner=window['i'],
                               start=window['start'], end=window['end'], frame=frame)
    for a in range(len(words)):
        for b in range(a + 1, min(a + CONSTANTS['quote_words'], len(words)) + 1):
            start, end = words[a].start(), words[b - 1].end()
            quote = body[start:end]
            if body.find(quote) != start or body.find(quote, start + 1) != -1:
                continue
            handle = prefix + f't{a:04d}_{b:04d}'
            sources[handle] = dict(handle=handle, kind='speech', owner=window['i'],
                start=window['start'], end=window['end'], char_start=start, char_end=end, quote=quote)
    return sources


def source_table(window):
    words = list(re.finditer(r'\S+', window['body']))
    return dict(window=window['i'], start=window['start'], end=window['end'], body=window['body'],
        words=[dict(i=i, start=w.start(), end=w.end()) for i, w in enumerate(words)],
        frames=model_visible(window['frames']),
        handle_rule=f'w{window["i"]:04d}:p<frame index> or w{window["i"]:04d}:t<start word:04d>_<exclusive end word:04d>; unique contiguous 1-16-word literal speech only')


def unknown_field(name, error=None):
    assert name in FIELDS
    return dict(name=name, known=False, value='UNKNOWN', witnesses=[], error=error)


def write_field(stream, name, sources):
    stream.force('{"name":' + canonical(name) + ',"known":')
    known = json.loads(stream.choose(['false'] + (['true'] if sources else [])))
    stream.force(',"value":"')
    if known:
        stream.description()
        value = stream.events[-1]['text']
    else:
        stream.force('UNKNOWN"')
        value = 'UNKNOWN'
    stream.force(',"witnesses":[')
    witnesses = []
    if known:
        witnesses.append(json.loads(stream.choose([canonical(h) for h in sources])))
        remaining = [h for h in sources if h != witnesses[0]]
        separator = stream.choose([']'] + ([','] if remaining else []))
        if separator == ',':
            witnesses.append(json.loads(stream.choose([canonical(h) for h in remaining])))
            stream.force(']')
    else:
        stream.force(']')
    stream.force('}')
    return dict(name=name, known=known, value=value, witnesses=witnesses)


def write_draft(stream, sources):
    stream.force('{"fields":[')
    fields = []
    for i, name in enumerate(FIELDS):
        if i:
            stream.force(',')
        fields.append(write_field(stream, name, sources))
    stream.force(']}')
    return dict(fields=fields)


def write_plan(stream, available):
    stream.force('{"questions":[')
    questions = []
    if available:
        remaining = list(FIELDS)
        while True:
            stream.force('{"field":')
            name = json.loads(stream.choose([canonical(n) for n in remaining]))
            remaining.remove(name)
            stream.force(',"question":"')
            stream.description()
            questions.append(dict(field=name, question=stream.events[-1]['text']))
            stream.force('}')
            if len(questions) == CONSTANTS['max_questions']:
                stream.force(']')
                break
            if stream.choose([']', ',']) == ']':
                break
    else:
        stream.force(']')
    stream.force('}')
    return dict(questions=questions)


def completed(g):
    assert type(g['truncated']) is bool
    if g['truncated']:
        assert g['selection'] is None
        return None
    assert g['selection'] == json.loads(g['text'])
    return g['selection']


def compile_field(raw, name, sources):
    assert set(raw) == {'name', 'known', 'value', 'witnesses'} and raw['name'] == name
    assert type(raw['known']) is bool and type(raw['value']) is str
    assert len(raw['value'].split()) <= CONSTANTS['description_words']
    handles = raw['witnesses']
    assert type(handles) is list and len(handles) <= CONSTANTS['max_witnesses']
    assert len(set(handles)) == len(handles) and all(h in sources for h in handles)
    if not raw['known']:
        assert raw['value'] == 'UNKNOWN' and not handles
        return unknown_field(name)
    assert handles
    if not raw['value'].strip():
        return unknown_field(name, 'empty_known_value')
    return dict(**copy.deepcopy(raw), error=None)


def compile_draft(g, sources):
    raw = completed(g)
    if raw is None:
        return dict(fields=[unknown_field(n, 'token_cap') for n in FIELDS], error='token_cap')
    assert set(raw) == {'fields'} and len(raw['fields']) == len(FIELDS)
    return dict(fields=[compile_field(r, n, sources) for r, n in zip(raw['fields'], FIELDS)], error=None)


def compile_plan(g, available):
    raw = completed(g)
    if raw is None:
        return dict(questions=[], error='token_cap')
    assert set(raw) == {'questions'} and type(raw['questions']) is list
    questions = raw['questions']
    assert (CONSTANTS['min_questions_with_sources'] <= len(questions) <= CONSTANTS['max_questions']) if available else not questions
    names = []
    for q in questions:
        assert set(q) == {'field', 'question'} and q['field'] in FIELDS and type(q['question']) is str
        assert len(q['question'].split()) <= CONSTANTS['description_words']
        names.append(q['field'])
    assert len(set(names)) == len(names)
    if any(not q['question'].strip() for q in questions):
        return dict(questions=[], error='empty_question')
    return dict(questions=copy.deepcopy(questions), error=None)


def compile_verification(g, name, sources):
    raw = completed(g)
    return unknown_field(name, 'token_cap') if raw is None else compile_field(raw, name, sources)


def replace_fields(draft, plan, observations):
    """Fresh queried UNKNOWN replaces draft; valid unqueried fields survive."""
    assert [r['name'] for r in draft['fields']] == list(FIELDS)
    assert len(observations) == len(plan['questions'])
    fields = {r['name']: copy.deepcopy(r) for r in draft['fields']}
    changes = []
    for q, observation in zip(plan['questions'], observations):
        assert observation['name'] == q['field']
        old = fields[q['field']]
        fields[q['field']] = copy.deepcopy(observation)
        keys = ('known', 'value', 'witnesses')
        changes.append(dict(field=q['field'], previous=copy.deepcopy(old), current=copy.deepcopy(observation),
                            literal_changed=any(old[k] != observation[k] for k in keys)))
    return dict(fields=[fields[n] for n in FIELDS]), changes


def branch_record(metadata, i, kind):
    window = metadata['windows'][i]
    record = metadata['records'][i]['compiled']
    sources = catalog(window)
    handles = []
    for field in record['fields']:
        for handle in field['witnesses']:
            if handle not in handles:
                handles.append(handle)
    return dict(current_window=i, start=window['start'], end=window['end'], modality=kind,
                fields=record['fields'], literal_witness_sources=[sources[h] for h in handles])
