"""Real tokenizer grammar, source-role corruption, and frozen question checks."""
import argparse
import copy
import json
from pathlib import Path
from compiler import writer_for,content,validate_selection,reader_record,question
from partition import SPEC,partition,scope
from src.mllm_renderer import cpu_renderer
from src.structured_source_generation import Stream


class Choices:
    def __init__(self,choices):self.choices=iter(choices);self.events=[]
    def force(self,text):self.events.append(('force',text))
    def choose(self,options):
        value=next(self.choices);assert value in options
        self.events.append(('choose',options,value));return value


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    j=cpu_renderer();words=[dict(id=i,text=' one',start=i/20,end=(i+1)/20) for i in range(400)]
    scoped=scope(words,partition(words),0,8)
    selections=['W00000002;','W00000004;','NONE;','W00000160;','W00000165;','NONE;']
    planned=Choices(selections);value=writer_for(words,scoped)(planned);validate_selection(words,scoped,value)
    assert value==dict(local_spans=[[2,5]],context_spans=[[160,166]],unknown=False)
    tokens=[]
    for event in planned.events:
        if event[0]=='force':tokens.extend(j.tok.encode(event[1],add_special_tokens=False))
        else:
            seq=[j.tok.encode(x,add_special_tokens=False) for x in event[1]];ordered=sorted(tuple(x) for x in seq)
            assert len(set(ordered))==len(ordered) and not any(b[:len(a)]==a for a,b in zip(ordered,ordered[1:]))
            tokens.extend(j.tok.encode(event[2],add_special_tokens=False))
    stream=Stream(j,SPEC['source_generation_token_cap'],tokens=tokens)
    replay=writer_for(words,scoped)(stream);assert replay==value and stream.tokens==tokens
    literal=reader_record(words,scoped,value);assert literal['local_acts'][0]['word_ids']==[2,3,4]
    assert literal['explaining_context'][0]['bounds'][0]>=8
    original='Frozen original Yes/No question';assert question(original,literal).endswith(original)
    bad=copy.deepcopy(value);bad['context_spans']=[[2,5]]
    try:validate_selection(words,scoped,bad)
    except AssertionError:pass
    else:raise AssertionError('wrong LOCAL/CONTEXT role accepted')
    unknown=writer_for(words,scoped)(Choices(['UNKNOWN;']));assert unknown['unknown'] and not unknown['local_spans']
    validate_selection(words,scoped,unknown)
    summary=dict(PASS=True,real_tokenizer_prefixfree_grammar=True,actual_stream_replay_exact=True,tokens=len(tokens),
        wrong_role_rejected=True,original_question_verbatim=True,scope='CPU grammar/source ownership only; no GPU or performance')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('PASS real grammar/token/literal ownership')


if __name__=='__main__':main()
