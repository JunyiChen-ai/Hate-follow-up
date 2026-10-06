"""Grammar-selected literal spans with explicit LOCAL and CONTEXT ownership."""
import json
from partition import SPEC,LEXICAL


def content(words,scoped):
    def entries(ids):
        return '\n'.join(f"W{i:08d} [{words[i]['start']:.4f}s,{words[i]['end']:.4f}s] {json.dumps(words[i]['text'],ensure_ascii=False)}" for i in ids) or '(none)'
    a,b=scoped['bounds']
    return [dict(type='text',text=f'Current window [{a:.4f}s,{b:.4f}s).\nLOCAL original words:\n'+entries(scoped['local_ids'])+
        '\nCONTEXT original words outside current window, in the same lexical segments:\n'+entries(scoped['context_ids'])+
        '\nSelect chronological inclusive spans by word IDs; endpoints must stay in one role, length at most32 words. Select UNKNOWN for unavailable LOCAL act; NONE for a second act or unnecessary explanation. Return only the requested fields.')]


def writer_for(words,scoped):
    def writer(stream):
        result=dict(local_spans=[],context_spans=[],unknown=False)
        for role,key in (('local_ids','local_spans'),('context_ids','context_spans')):
            pool=list(scoped[role]);previous=-1
            for number in range(SPEC['max_spans_per_role']):
                starts=[i for i in pool if i>previous and (role!='local_ids' or LEXICAL.search(words[i]['text']))]
                sentinel='UNKNOWN;' if role=='local_ids' and number==0 else 'NONE;'
                stream.force(f'{key}{number+1}_start=')
                selected=stream.choose([sentinel]+[f'W{i:08d};' for i in starts])
                if selected==sentinel:
                    if sentinel=='UNKNOWN;':result['unknown']=True;return result
                    break
                start=int(selected[1:-1]);ends=[]
                for i in range(start,min(len(words),start+SPEC['max_span_words'])):
                    if i not in pool:break
                    ends.append(i)
                assert ends and ends[0]==start
                stream.force(f'{key}{number+1}_end=')
                selected=stream.choose([f'W{i:08d};' for i in ends]);end=int(selected[1:-1])
                result[key].append([start,end+1]);previous=end
        stream.force('END;')
        return result
    return writer


def validate_selection(words,scoped,selection):
    assert set(selection)=={'local_spans','context_spans','unknown'} and type(selection['unknown']) is bool
    if selection['unknown']:assert not selection['local_spans'] and not selection['context_spans'];return
    assert selection['local_spans']
    for role,key in (('local_ids','local_spans'),('context_ids','context_spans')):
        spans=selection[key];assert len(spans)<=SPEC['max_spans_per_role'];prior=-1
        for start,end in spans:
            assert type(start) is int and type(end) is int and prior<start<end<=len(words)
            assert end-start<=SPEC['max_span_words'] and all(i in scoped[role] for i in range(start,end))
            if role=='local_ids':assert LEXICAL.search(words[start]['text'])
            prior=end-1


def literal(words,spans):
    return [dict(start_word=start,end_word=end,word_ids=list(range(start,end)),
        bounds=[words[start]['start'],words[end-1]['end']],text=''.join(w['text'] for w in words[start:end])) for start,end in spans]


def reader_record(words,scoped,selection):
    validate_selection(words,scoped,selection)
    assert not selection['unknown']
    return dict(current_window=scoped['bounds'],local_acts=literal(words,selection['local_spans']),
        explaining_context=literal(words,selection['context_spans']),role_semantics='only LOCAL is direct current-window evidence')


def question(original,record):
    return SPEC['reader_instruction']+'\nLiteral source record:\n'+json.dumps(record,ensure_ascii=False)+'\n\n'+original
